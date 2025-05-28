from state import State
from collections import defaultdict
from game import init_player_dice, init_state
import time
import numpy as np
import os
import json

STRATEGY_INTERVAL = 1000
PRUNE_THRESHOLD = 200 * 60  # 200 minutes
LCFR_TRESHOLD = 400 * 60  # 400 minutes
DISCOUNT_INTERVAL = 10 * 60  # 10 minutes
REGRET_PRUNE_THRESHOLD = -300

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


def calculate_strategy(state: State, global_dict: dict) -> dict:
    """
    :param state: state of the game
    :param global_dict: global dictionary for storing regret, strategy, state and explore values

    Calculates and update the strategy based on regrets and returns the global dictionary
    """
    regret_sum = 0
    valid_action_list = state.next_valid_move()
    for action in valid_action_list:
        child = state.copy()
        child.update_history(action)
        regret_sum += max(global_dict[str(child)][0], 0)

    for action in valid_action_list:
        child = state.copy()
        child.update_history(action)
        if regret_sum > 0:
            global_dict[str(child)][1] = max(global_dict[str(child)][0], 0) / regret_sum
        else:
            global_dict[str(child)][1] = 1 / len(valid_action_list)

    return global_dict


def update_strategy(state: State, global_dict: dict) -> dict:
    """
    :param state: state of the game
    :param global_dict: global dictionary for storing regret, strategy, state and explore values

    Updates the strategy based on current regret and state values
    Returns the global dictionary
    """

    if state.is_terminal():
        return global_dict
    elif state.player_of_next_move() == 1:  # agent's turn
        global_dict = calculate_strategy(state, global_dict)

        # sample the action
        action_probablities = np.array([])
        valid_action_list = state.next_valid_move()
        for valid_move in valid_action_list:
            sub_state = str(state)+str(valid_move[0])+str(valid_move[1])
            action_probablities = np.append(action_probablities, global_dict[sub_state][1])
        action_probablities /= np.sum(action_probablities)
        action = valid_action_list[np.random.choice(len(valid_action_list), p=action_probablities)]

        # update the strategy
        child_state = state.copy()
        child_state.update_history(action)
        global_dict[str(child_state)][4] += 1
        global_dict = update_strategy(child_state, global_dict)
    else:  # opponent's turn
        # iterate through all the valid actions
        for valid_move in state.next_valid_move():
            child_state = state.copy()
            child_state.update_history(valid_move)
            global_dict = update_strategy(child_state, global_dict)

    return global_dict


def sample_opponent_action(state: State, opponent_dice: int) -> tuple:
    """
    :param state: state of the game
    :param opponent_dice: opponent's dice number

    retrieve the opponent's state file and sample the opponent's action

    Returns the opponent's action
    """
    # replace the agent's dice with the opponent's dice
    s = state.copy()
    s.dice = opponent_dice
    valid_action_list = state.next_valid_move()

    # load the dictionary from the json file
    if os.path.exists(f'output/{str(state)}.json'):
        with open(f'output/{str(state)}.json', 'r') as fp:
            # print(f'reading existing dictionary of dice {state.dice}')
            opponent_dict = json.load(fp)
            opponent_dict = defaultdict(lambda: [0.0, 0.0, 0.0, False], opponent_dict)

        opponent_dict = calculate_strategy(state, opponent_dict)

        # get the action probabilities and sample actions
        action_probablities = np.array([])
        for valid_move in valid_action_list:
            sub_state = str(state)+str(valid_move[0])+str(valid_move[1])
            action_probablities = np.append(action_probablities, opponent_dict[sub_state][1])
        action_probablities /= np.sum(action_probablities)
        action = valid_action_list[np.random.choice(len(valid_action_list), p=action_probablities)]
    else:  # if the dictionary does not exist, sample uniformly
        action = valid_action_list[np.random.choice(len(valid_action_list))]

    return action


def traverse_mccfr(state: State, opponent_dice: int, global_dict: dict) -> float:
    """
    :param history: state of the game
    :param opponent_dice: opponent's dice number
    :param global_dict: global dictionary for storing regret, strategy, state and explore values

    Returns 1. utility value for terminal node, or expected value for non-terminal node
            2. global dictionary
    """

    if state.is_terminal():  # terminal node
        return state.utility(opponent_dice), global_dict
    elif state.dice == 0:  # dice have not been rolled, chance node
        state = init_state(state)
        return traverse_mccfr(state, opponent_dice, global_dict)
    elif state.player_of_next_move() == 1:  # agent's turn
        state_value = 0
        global_dict = calculate_strategy(state, global_dict)

        for valid_move in state.next_valid_move():
            child_state = state.copy()
            child_state.update_history(valid_move)
            sub_state_value = traverse_mccfr(child_state, opponent_dice, global_dict)[0]
            global_dict[str(child_state)][2] = sub_state_value
            state_value += global_dict[str(child_state)][1] * sub_state_value

        for valid_move in state.next_valid_move():
            sub_state = str(state)+str(valid_move[0])+str(valid_move[1])
            global_dict[sub_state][0] += (global_dict[sub_state][2] - state_value)

        return state_value, global_dict
    else:
        # sample the opponent's action
        action = sample_opponent_action(state.copy(), opponent_dice)
        child_state = state.copy()
        child_state.update_history(action)
        return traverse_mccfr(child_state, opponent_dice, global_dict)


def traverse_mccfr_p(state: State, opponent_dice: int, global_dict: dict) -> float:
    """
    :param history: state of the game
    :param opponent_dice: opponent's dice number
    :param global_dict: global dictionary for storing regret, strategy, state and explore values

    MCCFR with pruning for very negative regrets
    Returns utility value for terminal node, or expected value for non-terminal node
    """

    if state.is_terminal():  # terminal node
        return state.utility(opponent_dice), global_dict
    elif state.dice == 0:  # dice have not been rolled, chance node
        state = init_state(state)
        return traverse_mccfr_p(state, opponent_dice, global_dict)
    elif state.player_of_next_move() == 1:  # agent's turn
        state_value = 0
        global_dict = calculate_strategy(state, global_dict)

        for valid_move in state.next_valid_move():
            child_state = state.copy()
            child_state.update_history(valid_move)
            if global_dict[str(child_state)][0] > REGRET_PRUNE_THRESHOLD:
                sub_state_value = traverse_mccfr_p(child_state, opponent_dice, global_dict)[0]
                global_dict[str(child_state)][3] = True
                global_dict[str(child_state)][2] = sub_state_value
                state_value += global_dict[str(child_state)][1] * sub_state_value
            else:
                global_dict[str(child_state)][3] = False

                # mark all the greater bids to be not explored
                for greater_valid_move in child_state.next_valid_move():
                    # greater_child_state = child_state.copy()
                    # greater_child_state.update_history(greater_valid_move)
                    greater_child_state = str(child_state)+str(greater_valid_move[0])+str(greater_valid_move[1])
                    global_dict[greater_child_state][3] = False

                break

        for valid_move in state.next_valid_move():
            sub_state = str(state)+str(valid_move[0])+str(valid_move[1])
            if global_dict[sub_state][3]:
                global_dict[sub_state][0] += (global_dict[sub_state][2] - state_value)

        return state_value, global_dict
    else:
        # sample the opponent's action
        action = sample_opponent_action(state.copy(), opponent_dice)
        child_state = state.copy()
        child_state.update_history(action)
        return traverse_mccfr_p(child_state, opponent_dice, global_dict)


def MCCFR_P(T: int, start_time: float) -> defaultdict:
    """
    :param T: number of iterations

    Returns a dictionary of the first round betting strategies
    """

    for t in range(1, T):
        # random.seed(37 + t)
        # np.random.seed(37 + t)

        time_elasped = time.time() - start_time

        # init the state
        state = init_state(State([]))

        # load the dictionary from the json file
        if os.path.exists(f'output/{str(state.dice)}.json'):
            with open(f'output/{str(state.dice)}.json', 'r') as fp:
                # print(f'reading existing dictionary of dice {state.dice}')
                global_dict = json.load(fp)
                global_dict = defaultdict(lambda: [0.0, 0.0, 0.0, False, 0.0], global_dict)

            # update the strategy every 5000 iterations
            if t % STRATEGY_INTERVAL == 0:
                print(f'time elapsed: {time_elasped / 3600:.2f} hours')
                print(f'updating strategy for dice {state.dice}')
                global_dict = update_strategy(state, global_dict)
        else:
            global_dict = defaultdict(lambda: [0.0, 0.0, 0.0, False, 0.0])

        if time_elasped > PRUNE_THRESHOLD:
            q = np.random.rand()
            if q < 0.05:
                traverse_mccfr(state, init_player_dice(), global_dict)
            else:
                traverse_mccfr_p(state, init_player_dice(), global_dict)
        else:
            traverse_mccfr(state, init_player_dice(), global_dict)

        if time_elasped < LCFR_TRESHOLD and (time_elasped + 1) % DISCOUNT_INTERVAL == 0:
            d = (t / DISCOUNT_INTERVAL) / (t / DISCOUNT_INTERVAL + 1)

            for k in global_dict:
                global_dict[k][0] *= d  # discount the regret
                global_dict[k][4] *= d  # discount the strategy

        # save the dictionary to a json file
        with open(f'output/{str(state.dice)}.json', 'w') as fp:
            json.dump(global_dict, fp)
            # print('save dictionary of dFice', state.dice, 'to json file')

    # print time elapsed in hours
    print(f'time elapsed: {time_elasped / 3600:.2f} hours')
    return global_dict


if __name__ == "__main__":
    MCCFR_P(1*10**5, time.time())
