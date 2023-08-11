from state import State
from collections import defaultdict
from game import init_player_dice, init_state
from trie import Trie
import time
import numpy as np
import os
import random
import pickle

STRATEGY_INTERVAL = 1000
PRUNE_THRESHOLD = 2 * 60  # 200 minutes
LCFR_TRESHOLD = 400 * 60  # 400 minutes
DISCOUNT_INTERVAL = 10 * 60  # 10 minutes
REGRET_PRUNE_THRESHOLD = -15

# random.seed(37)
# # R for regret, S for state
# R = defaultdict(lambda: 0.0)
# sigma = defaultdict(lambda: 0.0)
# S = defaultdict(lambda: 0.0)
# explored = defaultdict(lambda: False)
# phi = defaultdict(lambda: 0.0)

# Global dictionary for stroing regret, strategy, state and explore values
# first index is the regret, second index is the strategy,
# third index is the state, fourth index is the explore value,
# fifth index is the final strategy value
# GLOBAL_DICT = defaultdict(lambda: [0.0, 0.0, 0.0, False, 0.0])

global_trie = Trie()


def calculate_strategy(state: State, trie: Trie) -> dict:
    """
    :param state: state of the game

    Calculates and update the strategy based on regrets and return the trie
    """
    regret_sum = 0
    valid_action_list = state.next_valid_move()
    for action in valid_action_list:
        child = state.copy()
        child.update_history(action)
        # find in the trie
        child_node = trie.search(str(child))
        if child_node is not None:
            regret_sum += max(child_node['#'][0], 0)

    for action in valid_action_list:
        child = state.copy()
        child.update_history(action)
        child_node = trie.search(str(child))
        if regret_sum > 0:
            if child_node is not None:
                child_node['#'][1] = max(child_node['#'][0], 0) / regret_sum
            else:
                trie.insert(str(child), strategy_p=0)
        else:
            if child_node is not None:
                child_node['#'][1] = 1 / len(valid_action_list)
            else:
                trie.insert(str(child), strategy_p=1/len(valid_action_list))
    return trie


def update_strategy(state: State, trie: Trie) -> Trie:
    """
    :param state: state of the game
    :param trie: global trie for storing regret, strategy, state and explore values

    Updates the strategy based on current regret and state values
    Returns the trie
    """

    if state.is_terminal():
        return trie
    elif state.player_of_current_turn() == 1:  # agent's turn
        trie = calculate_strategy(state, trie)

        # sample the action
        action_probablities = np.array([])
        valid_action_list = state.next_valid_move()
        for valid_move in valid_action_list:
            sub_state = str(state)+str(valid_move[0])+str(valid_move[1])
            sub_state_node = trie.search(sub_state)
            if sub_state_node is None:
                # the action has not been explored, so set the probability to 0
                action_probablities = np.append(action_probablities, 0)
            else:
                action_probablities = np.append(action_probablities, sub_state_node['#'][1])
        if np.sum(action_probablities) == 0:
            # if all the actions have not been explored, sample uniformly
            action_probablities = np.ones(len(valid_action_list))

        action_probablities /= np.sum(action_probablities)
        action = valid_action_list[np.random.choice(len(valid_action_list), p=action_probablities)]

        # update the strategy
        child_state = state.copy()
        child_state.update_history(action)
        child_state_node = trie.search(str(child_state))
        if child_state_node is not None:
            child_state_node['#'][4] += 1
        else:
            trie.insert(str(child_state), strategy_p=1/len(state.next_valid_move()))
            child_state_node = trie.search(str(child_state))
            child_state_node['#'][4] += 1
        trie = update_strategy(child_state, trie)
    else:  # opponent's turn
        # iterate through all the valid actions
        for valid_move in state.next_valid_move():
            child_state = state.copy()
            child_state.update_history(valid_move)
            trie = update_strategy(child_state, trie)

    return trie


def sample_opponent_action(state: State, opponent_dice: int) -> tuple:
    """
    :param state: state of the game
    :param opponent_dice: opponent's dice number

    retrieve the opponent's state file and sample the opponent's action

    Returns the opponent's action
    """
    # replace the agent's dice with the opponent's dice
    state.dice = opponent_dice
    valid_action_list = state.next_valid_move()

    # load the dictionary from the json file
    if os.path.exists(f"output/trie_{str(state.dice)}.pkl"):
        # oppoent_trie = defaultdict(lambda: [0.0, 0.0, 0.0, False], oppoent_trie)
        with open(f"output/trie_{str(state.dice)}.pkl", "rb") as f:
            # print(f'reading existing trie of dice {state.dice}')
            oppoent_trie = pickle.load(f)
        oppoent_trie = calculate_strategy(state, oppoent_trie)

        # get the action probabilities and sample actions
        action_probablities = np.array([])
        for valid_move in valid_action_list:
            sub_state = str(state)+str(valid_move[0])+str(valid_move[1])
            sub_state_node = oppoent_trie.search(sub_state)
            if sub_state_node is not None:
                action_probablities = np.append(action_probablities, sub_state_node['#'][1])
            else:
                # the action has not been explored, so set the probability to 0
                action_probablities = np.append(action_probablities, 0)
        if np.sum(action_probablities) == 0:
            # if all the actions have not been explored, sample uniformly
            action_probablities = np.ones(len(valid_action_list))
        action_probablities /= np.sum(action_probablities)
        action = valid_action_list[np.random.choice(len(valid_action_list), p=action_probablities)]
    else:  # if the dictionary does not exist, sample uniformly
        action = valid_action_list[np.random.choice(len(valid_action_list))]

    return action


def traverse_mccfr(state: State, opponent_dice: int, trie: Trie) -> float:
    """
    :param history: state of the game
    :param opponent_dice: opponent's dice number
    :param trie: global trie for storing regret, strategy, state and explore values

    Returns 1. utility value for terminal node, or expected value for non-terminal node
            2. global trie
    """

    if state.is_terminal():  # terminal node
        return state.utility(opponent_dice), trie
    elif state.dice == 0:  # dice have not been rolled, chance node
        state = init_state(state)
        return traverse_mccfr(state, opponent_dice, trie)
    elif state.player_of_current_turn() == 1:  # agent's turn
        state_value = 0
        trie = calculate_strategy(state, trie)

        sub_state_value_dict = {}
        valid_move_list = state.next_valid_move()

        for valid_move in valid_move_list:
            child_state = state.copy()
            child_state.update_history(valid_move)
            sub_state_value = traverse_mccfr(child_state, opponent_dice, trie)[0]
            sub_state_value_dict[valid_move] = sub_state_value
            child_node = trie.search(str(child_state))
            if child_node is not None:
                state_value += child_node['#'][1] * sub_state_value
            else:
                trie.insert(str(child_state), strategy_p=1/len(valid_move_list))
                new_child_node = trie.search(str(child_state))
                state_value += new_child_node['#'][1] * sub_state_value

        for valid_move in valid_move_list:
            sub_state = str(state)+str(valid_move[0])+str(valid_move[1])
            sub_state_node = trie.search(sub_state)
            # sub_state_node is not None because we have inserted it in the previous loop
            sub_state_node['#'][0] += (sub_state_value_dict[valid_move] - state_value)

        return state_value, trie
    else:
        # sample the opponent's action
        action = sample_opponent_action(state.copy(), opponent_dice)
        child_state = state.copy()
        child_state.update_history(action)
        return traverse_mccfr(child_state, opponent_dice, trie)


def traverse_mccfr_p(state: State, opponent_dice: int, trie: Trie) -> float:
    """
    :param history: state of the game
    :param opponent_dice: opponent's dice number
    :param trie: global trie for storing regret, strategy, state and explore values

    MCCFR with pruning for very negative regrets
    Returns utility value for terminal node, or expected value for non-terminal node
    """

    if state.is_terminal():  # terminal node
        return state.utility(opponent_dice), trie
    elif state.dice == 0:  # dice have not been rolled, chance node
        state = init_state(state)
        return traverse_mccfr_p(state, opponent_dice, trie)
    elif state.player_of_current_turn() == 1:  # agent's turn
        state_value = 0
        trie = calculate_strategy(state, trie)
        sub_state_value_dict = {}
        valid_move_list = state.next_valid_move()

        for valid_move in valid_move_list:
            child_state = state.copy()
            child_state.update_history(valid_move)
            child_node = trie.search(str(child_state))
            if child_node is None:
                trie.insert(str(child_state), strategy_p=1/len(valid_move_list))
                child_node = trie.search(str(child_state))
            if child_node['#'][0] > REGRET_PRUNE_THRESHOLD:
                sub_state_value = traverse_mccfr_p(child_state, opponent_dice, trie)[0]
                sub_state_value_dict[valid_move] = sub_state_value
                state_value += child_node['#'][1] * sub_state_value

        for valid_move in valid_move_list:
            sub_state = str(state)+str(valid_move[0])+str(valid_move[1])
            sub_state_node = trie.search(sub_state)
            if sub_state_node is not None and sub_state_node['#'][0] > REGRET_PRUNE_THRESHOLD:
                sub_state_node['#'][0] += (sub_state_value_dict[valid_move] - state_value)

        return state_value, trie
    else:
        # sample the opponent's action
        action = sample_opponent_action(state.copy(), opponent_dice)
        child_state = state.copy()
        child_state.update_history(action)
        return traverse_mccfr_p(child_state, opponent_dice, trie)


def MCCFR_P(T: int, start_time: float) -> defaultdict:
    """
    :param T: number of iterations

    Returns a dictionary of the first round betting strategies
    """

    # determine if there is a previous time record file in the output directory
    if os.path.exists("output/time.pkl"):
        with open("output/time.pkl", "rb") as f:
            previous_time_record = pickle.load(f)
            previous_cumulative_time = previous_time_record['cumulative_time']
            previous_T = previous_time_record['T']
    else:
        previous_cumulative_time = 0
        previous_T = 0

    for t in range(1, T):
        # random.seed(37 + t)
        # np.random.seed(37 + t)
        time_elasped = time.time() - start_time
        if t % 100 == 0:
            print(f'iteration {t} time elapsed: {time_elasped / 3600:.2f} hours')
        # init the state
        state = init_state(State([]))
        if os.path.exists(f"output/trie_{str(state.dice)}.pkl"):
            with open(f"output/trie_{str(state.dice)}.pkl", "rb") as f:
                trie = pickle.load(f)
            # update the strategy every 5000 iterations
            # if t % STRATEGY_INTERVAL == 0:
            #     print(f'time elapsed: {time_elasped / 3600:.2f} hours')
            #     print(f'updating strategy for dice {state.dice}')
            #     trie = update_strategy(state, trie)
        else:
            trie = Trie()

        if time_elasped + previous_cumulative_time > PRUNE_THRESHOLD:
            q = np.random.rand()
            if q < 0.05:
                traverse_mccfr(state, init_player_dice(), trie)
            else:
                traverse_mccfr_p(state, init_player_dice(), trie)
        else:
            traverse_mccfr(state, init_player_dice(), trie)

        if time_elasped + previous_cumulative_time < LCFR_TRESHOLD \
              and (time_elasped + previous_cumulative_time + 1) % DISCOUNT_INTERVAL == 0:
            d = ((t + previous_T) / DISCOUNT_INTERVAL) / (((t + previous_T) / DISCOUNT_INTERVAL) + 1)

            for end_state in trie.all_end_state():
                end_state['#'][0] /= d  # discount the regret
                # end_state['#'][4] /= d  # discount the strategy

        # Serialize and save the Trie object
        with open(f"output/trie_{str(state)}.pkl", "wb") as f:
            pickle.dump(trie, f)
        
        # Serialize and save the time record
        with open("output/time.pkl", "wb") as f:
            pickle.dump({'cumulative_time': time_elasped + previous_cumulative_time,
                         'T': t + previous_T}, f)

    print(f'training episode of {T} iterations ends. time elapsed: {time_elasped / 3600:.2f} hours')
    print(f'total number of iterations: {T + previous_T}')
    print(f'total time elapsed: {(time_elasped + previous_cumulative_time) / 3600:.2f} hours')
    return trie


if __name__ == "__main__":
    MCCFR_P(1*10**3, time.time())
