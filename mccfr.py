from state import State
from collections import defaultdict
from game import init_player_dice, init_state
from trie import Trie
import time
import numpy as np
import os
import random
import pickle
import argparse
import yaml

DEFAULT_TRAINING_CONFIG = {
    "iterations": 1 * 10**6,
    "strategy_interval": 1000,
    "prune_threshold": 120 * 60,
    "lcfr_threshold": 1200 * 60,
    "discount_interval": 10 * 60,
    "discount_iteration_interval": 100,
    "regret_prune_threshold": -15,
    "save_interval": 5000,
}

STRATEGY_INTERVAL = DEFAULT_TRAINING_CONFIG["strategy_interval"]  # in iterations
PRUNE_THRESHOLD = DEFAULT_TRAINING_CONFIG["prune_threshold"]  # in seconds
LCFR_TRESHOLD = DEFAULT_TRAINING_CONFIG["lcfr_threshold"]  # in seconds
DISCOUNT_INTERVAL = DEFAULT_TRAINING_CONFIG["discount_interval"]  # in seconds
DISCOUNT_ITERATION_INTERVAL = DEFAULT_TRAINING_CONFIG["discount_iteration_interval"]  # in iterations
REGRET_PRUNE_THRESHOLD = DEFAULT_TRAINING_CONFIG["regret_prune_threshold"]
SAVE_INTERVAL = DEFAULT_TRAINING_CONFIG["save_interval"]  # in iterations


def load_training_config(config_path: str) -> dict:
    """Load MCCFR training parameters from YAML and merge with defaults."""
    config = DEFAULT_TRAINING_CONFIG.copy()
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file must contain a top-level mapping: {config_path}")
        for key in config:
            if key in loaded:
                config[key] = loaded[key]
    return config


def apply_training_config(config: dict) -> None:
    """Apply config values to module-level constants used by training code."""
    global STRATEGY_INTERVAL
    global PRUNE_THRESHOLD
    global LCFR_TRESHOLD
    global DISCOUNT_INTERVAL
    global DISCOUNT_ITERATION_INTERVAL
    global REGRET_PRUNE_THRESHOLD
    global SAVE_INTERVAL

    STRATEGY_INTERVAL = int(config["strategy_interval"])
    PRUNE_THRESHOLD = int(config["prune_threshold"])
    LCFR_TRESHOLD = int(config["lcfr_threshold"])
    DISCOUNT_INTERVAL = int(config["discount_interval"])
    DISCOUNT_ITERATION_INTERVAL = int(config["discount_iteration_interval"])
    REGRET_PRUNE_THRESHOLD = float(config["regret_prune_threshold"])
    SAVE_INTERVAL = int(config["save_interval"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/mccfr.yaml")
    parser.add_argument("--iterations", type=int, default=None)

    # Optional per-parameter CLI overrides (take precedence over YAML).
    parser.add_argument("--strategy-interval", type=int, default=None)
    parser.add_argument("--prune-threshold", type=int, default=None)
    parser.add_argument("--lcfr-threshold", type=int, default=None)
    parser.add_argument("--discount-interval", type=int, default=None)
    parser.add_argument("--discount-iteration-interval", type=int, default=None)
    parser.add_argument("--regret-prune-threshold", type=float, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    return parser.parse_args()

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

# Cache for opponent tries to avoid repeated disk I/O
_opponent_cache = {}


def clear_opponent_cache():
    """Clear the opponent trie cache (useful for testing)."""
    _opponent_cache.clear()


def get_average_strategy(state: State, trie: Trie) -> dict:
    """
    :param state: state of the game
    :param trie: global trie storing regret, strategy probability, and action counter

    Computes the average strategy from cumulative action counts (index [2]).
    In CFR, the average strategy converges to Nash equilibrium.

    Returns a dict mapping each valid action to its average strategy probability.
    """
    valid_action_list = state.next_valid_move()
    count_sum = 0.0
    counts = {}

    for action in valid_action_list:
        child = str(state) + str(action[0]) + str(action[1])
        child_node = trie.search(child)
        if child_node is not None:
            counts[action] = child_node['#'][2]
            count_sum += child_node['#'][2]
        else:
            counts[action] = 0.0

    result = {}
    if count_sum > 0:
        for action in valid_action_list:
            result[action] = counts[action] / count_sum
    else:
        uniform = 1.0 / len(valid_action_list)
        for action in valid_action_list:
            result[action] = uniform

    return result


def calculate_strategy(state: State, trie: Trie) -> dict:
    """
    :param state: state of the game

    Calculates and update the strategy based on regrets and return the trie
    """
    regret_sum = 0
    valid_action_list = state.next_valid_move()
    for action in valid_action_list:
        child = str(state) + str(action[0]) + str(action[1])
        # find in the trie
        child_node = trie.search(child)
        if child_node is not None:
            regret_sum += max(child_node['#'][0], 0)

    for action in valid_action_list:
        child = str(state) + str(action[0]) + str(action[1])
        child_node = trie.search(child)
        if regret_sum > 0:
            if child_node is not None:
                child_node['#'][1] = max(child_node['#'][0], 0) / regret_sum
            else:
                trie.insert(child, strategy_p=0)
        else:
            if child_node is not None:
                child_node['#'][1] = 1 / len(valid_action_list)
            else:
                trie.insert(child, strategy_p=1/len(valid_action_list))
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
    elif state.player_of_next_move() == 1:  # agent's turn
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
            child_state_node['#'][2] += 1 # TODO
        else:
            trie.insert(str(child_state), strategy_p=1/len(valid_action_list))
            child_state_node = trie.search(str(child_state))
            child_state_node['#'][2] += 1
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
    s = state.copy()
    s.dice = opponent_dice
    valid_action_list = s.next_valid_move()

    # load the opponent trie from cache or disk
    dice_key = s.dice
    if dice_key not in _opponent_cache:
        path = f"output/trie_{str(s.dice)}.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                _opponent_cache[dice_key] = pickle.load(f)

    if dice_key in _opponent_cache:
        oppoent_trie = _opponent_cache[dice_key]
        oppoent_trie = calculate_strategy(s, oppoent_trie)

        # get the action probabilities and sample actions
        action_probablities = np.array([])
        for valid_move in valid_action_list:
            sub_state = str(s)+str(valid_move[0])+str(valid_move[1])
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
    :param trie: global trie for storing regret value, strategy probability, and action counter

    Returns 1. utility value for terminal node, or expected state value for non-terminal node
            2. global trie
    """

    if state.is_terminal():  # terminal node
        return state.utility(opponent_dice), trie
    elif state.dice == 0:  # dice have not been rolled, chance node
        state = init_state(state)
        return traverse_mccfr(state, opponent_dice, trie)
    elif state.player_of_next_move() == 1:  # agent's turn
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
    :param trie: global trie for storing regret value, strategy probability, and action counter

    MCCFR with pruning for very negative regrets
    Returns utility value for terminal node, or expected state value for non-terminal node
    """

    if state.is_terminal():  # terminal node
        return state.utility(opponent_dice), trie
    elif state.dice == 0:  # dice have not been rolled, chance node
        state = init_state(state)
        return traverse_mccfr_p(state, opponent_dice, trie)
    elif state.player_of_next_move() == 1:  # agent's turn
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

    # In-memory cache for agent tries — only load from disk on first encounter
    agent_tries = {}

    total_cumulative_time = previous_cumulative_time
    pruning_active = total_cumulative_time > PRUNE_THRESHOLD
    lcfr_active = total_cumulative_time < LCFR_TRESHOLD
    last_report_time = 0.0
    iter_times = []  # track recent iteration durations for speed estimate

    print(f"{'='*60}")
    print(f"  MCCFR Training — {T-1} iterations")
    if previous_T > 0:
        print(f"  Resuming from iteration {previous_T} ({previous_cumulative_time/3600:.2f}h)")
    print(f"  Pruning threshold: {PRUNE_THRESHOLD/60:.0f}min | LCFR threshold: {LCFR_TRESHOLD/60:.0f}min")
    print(f"  Pruning: {'ON' if pruning_active else 'OFF'} | LCFR discounting: {'ON' if lcfr_active else 'OFF'}")
    print(f"{'='*60}")

    for t in range(1, T):
        iter_start = time.time()
        random.seed(37 + t)
        np.random.seed(37 + t)
        time_elasped = time.time() - start_time
        total_cumulative_time = time_elasped + previous_cumulative_time

        # init the state
        state = init_state(State([]))
        dice_key = state.dice
        if dice_key not in agent_tries:
            path = f"output/trie_{str(state.dice)}.pkl"
            if os.path.exists(path):
                with open(path, "rb") as f:
                    agent_tries[dice_key] = pickle.load(f)
            else:
                agent_tries[dice_key] = Trie()
        trie = agent_tries[dice_key]

        # update the strategy
        if (t+1) % STRATEGY_INTERVAL == 0:
            trie = update_strategy(state, trie)

        if total_cumulative_time > PRUNE_THRESHOLD:
            q = np.random.rand()
            if q < 0.05:
                traverse_mccfr(state, init_player_dice(), trie)
            else:
                traverse_mccfr_p(state, init_player_dice(), trie)
        else:
            traverse_mccfr(state, init_player_dice(), trie)

        if total_cumulative_time < LCFR_TRESHOLD \
                and t % DISCOUNT_ITERATION_INTERVAL == 0:
            d = (t / DISCOUNT_ITERATION_INTERVAL) / \
                ((t / DISCOUNT_ITERATION_INTERVAL) + 1)

            for path, end_state in trie.all_end_state():
                end_state['#'][0] *= d  # discount the regret
                end_state['#'][2] *= d  # discount the action counter

        # Track iteration speed
        iter_dur = time.time() - iter_start
        iter_times.append(iter_dur)
        if len(iter_times) > 100:
            iter_times.pop(0)

        # Progress report every 100 iterations
        if t % 100 == 0:
            pct = t / (T - 1) * 100
            avg_speed = len(iter_times) / sum(iter_times) if iter_times else 0
            remaining = (T - 1 - t) / avg_speed if avg_speed > 0 else 0

            # Check phase transitions
            was_pruning = pruning_active
            was_lcfr = lcfr_active
            pruning_active = total_cumulative_time > PRUNE_THRESHOLD
            lcfr_active = total_cumulative_time < LCFR_TRESHOLD

            trie_nodes = len(trie.data)
            print(f"  [{pct:5.1f}%] iter {t + previous_T:>7d} | "
                  f"{time_elasped/60:6.1f}min | "
                  f"{avg_speed:.1f} it/s | "
                  f"ETA {remaining/60:.1f}min | "
                  f"dice seen: {len(agent_tries)} | "
                  f"trie nodes: {trie_nodes:,}")

            if pruning_active and not was_pruning:
                print(f"  >>> Pruning activated at {total_cumulative_time/60:.1f}min")
            if not lcfr_active and was_lcfr:
                print(f"  >>> LCFR discounting ended at {total_cumulative_time/60:.1f}min")

        if (t+1) % SAVE_INTERVAL == 0:
            if not os.path.exists("output"):
                os.makedirs("output")

            # Serialize and save the Trie object
            with open(f"output/trie_{str(state.dice)}.pkl", "wb") as f:
                pickle.dump(trie, f)

            # Serialize and save the time record
            with open("output/time.pkl", "wb") as f:
                pickle.dump({'cumulative_time': total_cumulative_time,
                            'T': t + previous_T}, f)

        # Strategy update report
        if (t+1) % STRATEGY_INTERVAL == 0:
            print(f"  --- Strategy updated for dice {state.dice} "
                  f"(trie nodes: {len(trie.data):,})")

    print(f"{'='*60}")
    print(f"  Training complete: {T-1} iterations in {time_elasped/3600:.2f}h")
    print(f"  Total iterations (all sessions): {T - 1 + previous_T}")
    print(f"  Total time (all sessions): {total_cumulative_time/3600:.2f}h")
    print(f"  Unique dice trained: {len(agent_tries)}")
    total_nodes = sum(len(t.data) for t in agent_tries.values())
    print(f"  Total trie nodes across all dice: {total_nodes:,}")
    print(f"{'='*60}")
    return trie


if __name__ == "__main__":
    args = parse_args()
    config = load_training_config(args.config)

    cli_overrides = {
        "iterations": args.iterations,
        "strategy_interval": args.strategy_interval,
        "prune_threshold": args.prune_threshold,
        "lcfr_threshold": args.lcfr_threshold,
        "discount_interval": args.discount_interval,
        "discount_iteration_interval": args.discount_iteration_interval,
        "regret_prune_threshold": args.regret_prune_threshold,
        "save_interval": args.save_interval,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            config[key] = value

    apply_training_config(config)
    MCCFR_P(int(config["iterations"]), time.time())
