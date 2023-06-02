from collections import defaultdict
import time
import numpy as np
import random

random.seed(37)
np.random.seed(37)

STRATEGY_INTERVAL = 10000
PRUNE_THRESHOLD = 200 * 60  # 200 minutes
LCFR_TRESHOLD = 400 * 60  # 400 minutes
DISCOUNT_INTERVAL = 10 * 60  # 10 minutes
REGRET_PRUNE_THRESHOLD = -3 * 1e5

# R for regret, S for state
R = defaultdict(lambda: 0.0)
sigma = defaultdict(lambda: 0.0)
S = defaultdict(lambda: 0.0)
explored = defaultdict(lambda: False)
phi = defaultdict(lambda: 0.0)

# def update_strategy(history: str, P: int):
#     """
#     :param history: history of the game
#     :param P: player number

#     Updates the average strategy for player P
#     """
#     if _is_terminal(history) or len(history) > :
#         return


def _player_of_current_turn(history: str) -> int:
    """
    :param history: history of the game

    Returns player number of the current turn (0 or 1)
    """
    assert (len(history) >= 11 and not history.endswith('c'))

    first_to_act = history[10]

    if (len(history) - 11) % 4 == 0:
        return first_to_act
    else:
        if first_to_act == 0:
            return 1
        else:  # first_to_act == 1
            return 0


def _next_valid_move(history: str) -> list:
    """
    :param history: history of the game

    Returns a list of valid moves for the current turn
    """
    assert (len(history) >= 11)

    if history.endswith('c'):
        return []

    bid = [str(i) for i in range(3, 10)]
    value = [str(i) for i in range(1, 7)]

    if len(history) == 11:
        valid_move = [b + v for b in bid for v in value]

    else:
        last_bid = int(history[-2])
        last_value = int(history[-1])

        assert (last_bid <= 9 and last_value <= 6)

        wild_one = '1' not in history[11:]

        valid_move = []
        # get valid moves with the same bid
        if wild_one:
            greater_values = [str(i) for i in range(last_value + 1, 7)] + ['1']
        else:
            greater_values = [str(i) for i in range(last_value + 1, 7)]

        valid_move += [str(last_bid) +
                       greater_val for greater_val in greater_values]

        # get valid moves with greater bid
        greater_bid = [str(i) for i in range(last_bid + 1, 10)]

        if wild_one:
            valid_move += [b + v for b in greater_bid for v in value]
        else:
            # no 1's
            valid_move += [b + v for b in greater_bid for v in value[1:]]

        valid_move += ['c']
        # print(valid_move)
    return valid_move


def _utility(history: str, P: int) -> float:
    """
    :param history: history of the game
    :param P: player number

    Returns utility value for terminal node
    """
    assert (history.endswith('c') and len(history) >= 14)

    bid = history[-3]
    val = history[-2]

    first_to_act = int(history[10])
    wild_one = '1' not in history[11:]

    dice = {str(val): 0 for val in range(1, 7)}
    for d in history[:10]:
        dice[d] += 1

    if wild_one:
        if int(bid) > dice[val] + dice['1']:
            result = 1
        else:
            result = -1
    else:
        if int(bid) > dice[val]:
            result = 1
        else:
            result = -1
    if (len(history) + 1 - 11) % 4 == 0:
        if first_to_act == P:
            return -1.0 * result
        else:
            return 1.0 * result
    else:
        if first_to_act == P:
            return 1.0 * result
        else:
            return -1.0 * result


def _is_chance(history: str) -> bool:
    """
    :param history: history of the game

    Returns True if the game is chance node
    """
    return len(history) <= 10


def _chance_action(history: str) -> str:
    """
    :param history: history of the game

    Returns a random action for chance node
    """
    his_len = len(history)

    assert (his_len == 0 or his_len == 10 or his_len == 5)

    if his_len == 10:  # decide who goes first
        return random.choice(['0', '1'])
    else:
        return ''.join([str(random.randint(1, 6)) for _ in range(5)])


def _sample_action(history: str) -> str:
    """
    :param history: history of the game

    Returns a random action for the current turn based on the strategy profile
    """
    assert (len(history) >= 11)

    nonzero_valid_actions = []
    action_frequencies = np.array([])

    global sigma
    action_list = _next_valid_move(history)
    np.random.shuffle(np.array(action_list))
    for valid_move in action_list:
        if sigma[history + valid_move] != 0:
            nonzero_valid_actions.append(valid_move)
            action_frequencies = np.append(
                action_frequencies, sigma[history + valid_move])

    assert (len(nonzero_valid_actions) == len(action_frequencies))
    if (len(nonzero_valid_actions) == 0):
        return random.choice(action_list)

    action_frequencies = np.divide(
        action_frequencies, np.sum(action_frequencies))
    action_frequencies = np.cumsum(action_frequencies)

    r = random.random()
    for i in range(len(action_frequencies)):
        if r < action_frequencies[i]:
            return nonzero_valid_actions[i]


def calculate_strategy(history: str) -> defaultdict:
    """
    :param history: history of the game

    Calculates the strategy based on regrets
    """
    regret_sum = 0
    global R, sigma

    for valid_move in _next_valid_move(history):
        regret_sum += max(R[history + valid_move], 0)

    for valid_move in _next_valid_move(history):
        if regret_sum > 0:
            sigma[history +
                  valid_move] = max(R[history + valid_move], 0) / regret_sum
        else:
            sigma[history + valid_move] = 1 / len(_next_valid_move(history))

    return sigma[history]  # may be unnecessary


def traverse_mccfr(history: str, P: int) -> float:
    """
    :param history: history of the game
    :param P: player number

    Returns utility value for terminal node, or expected value for non-terminal node
    """

    global sigma, R, S

    if history.endswith('c'):
        return _utility(history, P)  # TODO: do i need to pass P?
    elif _is_chance(history):  # TODO: what nodes are chance nodes?
        # sample an action from the chance probabilities
        action = _chance_action(history)
        return traverse_mccfr(history + action, P)
    elif _player_of_current_turn(history) == P:
        state_value = 0
        calculate_strategy(history)

        for valid_move in _next_valid_move(history):
            sub_state_value = traverse_mccfr(history + valid_move, P)
            S[history + valid_move] = sub_state_value
            state_value += sigma[history + valid_move] * sub_state_value

        for valid_move in _next_valid_move(history):
            R[history + valid_move] += (S[history + valid_move] - state_value)

        return state_value
    else:
        calculate_strategy(history)

        # sample actions
        action = _sample_action(history)
        return traverse_mccfr(history + action, P)


def traverse_mccfr_p(history: str, P: int) -> float:
    """
    :param history: history of the game
    :param P: player number

    MCCFR with pruning for very negative regrets
    Returns utility value for terminal node, or expected value for non-terminal node
    """
    global sigma, R, S, explored

    if history.endswith('c'):
        return _utility(history, P)  # TODO: do i need to pass P?
    elif _is_chance(history):  # TODO: what nodes are chance nodes?
        # sample an action from the chance probabilities
        action = _chance_action(history)
        return traverse_mccfr_p(history + action, P)
    elif _player_of_current_turn(history) == P:
        state_value = 0
        calculate_strategy(history)

        for valid_move in _next_valid_move(history):
            if R[history + valid_move] > REGRET_PRUNE_THRESHOLD:
                sub_state_value = traverse_mccfr_p(history + valid_move, P)
                explored[history + valid_move] = True
                S[history + valid_move] = sub_state_value
                state_value += sigma[history + valid_move] * sub_state_value
            else:
                explored[history + valid_move] = False
        for valid_move in _next_valid_move(history):
            if explored[history + valid_move]:
                R[history +
                    valid_move] += (S[history + valid_move] - state_value)

        return state_value
    else:
        calculate_strategy(history)

        # sample actions
        action = _sample_action(history)
        return traverse_mccfr_p(history + action, P)


def MCCFR_P(T: int, start_time: float) -> defaultdict:
    """
    :param T: number of iterations

    Returns a dictionary of the first round betting strategies
    """

    for t in range(1, T):
        for P in [0, 1]:  # there are only two players
            # if t % STRATEGY_INTERVAL == 0:
            #     update_strategy('', P)

            time_elasped = time.time() - start_time

            if time_elasped > PRUNE_THRESHOLD:
                q = np.random.rand()
                if q < 0.05:
                    traverse_mccfr('', P)
                else:
                    traverse_mccfr_p('', P)
            else:
                traverse_mccfr('', P)

            if time_elasped < LCFR_TRESHOLD and time_elasped % DISCOUNT_INTERVAL == 0:
                d = (t / DISCOUNT_INTERVAL) / (t / DISCOUNT_INTERVAL + 1)

                global R, phi

                # TODO: probably need to improve efficiency here
                for k in R:
                    R[k] /= d

                # for k in phi:
                #     phi[k] /= d

    global sigma
    return sigma  # may be unnecessary


if __name__ == "__main__":
    MCCFR_P(10**7, time.time())
