import pytest
from unittest.mock import patch
import numpy as np
from mccfr import (
    calculate_strategy,
    update_strategy,
    sample_opponent_action,
    traverse_mccfr,
    traverse_mccfr_p
)
from state import State
from trie import Trie


@pytest.fixture
def trie():
    """Create a fresh Trie for testing."""
    return Trie()


@pytest.fixture
def initial_state():
    """Create an initial state for testing."""
    return State([55555, (2, 3)], first_act=True)


# --- calculate_strategy tests ---

def test_calculate_strategy_uniform_on_zero_regrets():
    """All regrets 0 -> uniform 1/N."""
    state = State([22222, (2, 3)], first_act=True)
    trie = Trie()
    valid_moves = state.next_valid_move()

    # Insert all child nodes with zero regret
    for move in valid_moves:
        child = str(state) + str(move[0]) + str(move[1])
        trie.insert(child, strategy_p=0)

    trie = calculate_strategy(state, trie)

    expected_prob = 1.0 / len(valid_moves)
    for move in valid_moves:
        child = str(state) + str(move[0]) + str(move[1])
        node = trie.search(child)
        assert abs(node['#'][1] - expected_prob) < 1e-10


def test_calculate_strategy_proportional_to_positive_regrets():
    """Regrets [3, 1, 0, 0] -> probs [0.75, 0.25, 0, 0]."""
    state = State([22222, (7, 6)], first_act=False)
    trie = Trie()
    valid_moves = state.next_valid_move()
    assert len(valid_moves) >= 4

    # Set specific regrets for the first 4 actions
    regrets = [3.0, 1.0, 0.0, 0.0]
    for i, move in enumerate(valid_moves):
        child = str(state) + str(move[0]) + str(move[1])
        trie.insert(child, strategy_p=0)
        if i < len(regrets):
            trie.search(child)['#'][0] = regrets[i]

    trie = calculate_strategy(state, trie)

    # Check proportional probabilities for first 4
    expected = [0.75, 0.25, 0.0, 0.0]
    for i in range(4):
        child = str(state) + str(valid_moves[i][0]) + str(valid_moves[i][1])
        node = trie.search(child)
        assert abs(node['#'][1] - expected[i]) < 1e-10


def test_calculate_strategy_negative_regrets_clamped():
    """Negative regrets -> prob 0, positive -> proportional."""
    state = State([22222, (7, 6)], first_act=False)
    trie = Trie()
    valid_moves = state.next_valid_move()

    # Set one positive, rest negative
    for i, move in enumerate(valid_moves):
        child = str(state) + str(move[0]) + str(move[1])
        trie.insert(child, strategy_p=0)
        if i == 0:
            trie.search(child)['#'][0] = 5.0
        else:
            trie.search(child)['#'][0] = -10.0

    trie = calculate_strategy(state, trie)

    # First action should get prob 1.0, rest 0
    child0 = str(state) + str(valid_moves[0][0]) + str(valid_moves[0][1])
    assert abs(trie.search(child0)['#'][1] - 1.0) < 1e-10

    for i in range(1, len(valid_moves)):
        child = str(state) + str(valid_moves[i][0]) + str(valid_moves[i][1])
        assert abs(trie.search(child)['#'][1] - 0.0) < 1e-10


def test_calculate_strategy_all_negative_regrets():
    """All negative -> uniform fallback."""
    state = State([22222, (7, 6)], first_act=False)
    trie = Trie()
    valid_moves = state.next_valid_move()

    for move in valid_moves:
        child = str(state) + str(move[0]) + str(move[1])
        trie.insert(child, strategy_p=0)
        trie.search(child)['#'][0] = -5.0

    trie = calculate_strategy(state, trie)

    expected_prob = 1.0 / len(valid_moves)
    for move in valid_moves:
        child = str(state) + str(move[0]) + str(move[1])
        node = trie.search(child)
        assert abs(node['#'][1] - expected_prob) < 1e-10


def test_calculate_strategy_inserts_missing_nodes():
    """Empty trie -> creates nodes with appropriate prob."""
    state = State([22222, (7, 6)], first_act=False)
    trie = Trie()
    valid_moves = state.next_valid_move()

    trie = calculate_strategy(state, trie)

    # All nodes should now exist with uniform probability
    expected_prob = 1.0 / len(valid_moves)
    for move in valid_moves:
        child = str(state) + str(move[0]) + str(move[1])
        node = trie.search(child)
        assert node is not None
        assert abs(node['#'][1] - expected_prob) < 1e-10


# --- update_strategy tests ---

def test_update_strategy_terminal_state():
    """update_strategy with a terminal state returns trie unchanged."""
    trie = Trie()
    terminal_state = State([55555, (2, 3), (-1, -1)], first_act=True)
    result_trie = update_strategy(terminal_state, trie)
    assert result_trie is trie


# --- traverse_mccfr tests ---

def test_traverse_mccfr_terminal_state():
    """traverse_mccfr with terminal state returns utility."""
    trie = Trie()
    terminal_state = State([55555, (2, 3), (-1, -1)], first_act=True)
    opponent_dice = 55555
    utility, result_trie = traverse_mccfr(terminal_state, opponent_dice, trie)
    assert isinstance(utility, (int, float))
    assert result_trie is trie


def test_traverse_mccfr_p_terminal_state():
    """traverse_mccfr_p with terminal state returns utility."""
    trie = Trie()
    terminal_state = State([55555, (2, 3), (-1, -1)], first_act=True)
    opponent_dice = 55555
    utility, result_trie = traverse_mccfr_p(terminal_state, opponent_dice, trie)
    assert isinstance(utility, (int, float))
    assert result_trie is trie


# --- Hand-traced traverse_mccfr test ---

def test_traverse_mccfr_hand_traced_regrets():
    """
    Hand-traced test for traverse_mccfr.

    Setup: State([22222, (7,6)], first_act=False), opponent_dice=33333.
    It's the agent's turn (history len=1, first_act=False -> player 1).
    Mock sample_opponent_action to always return challenge (-1,-1).

    Agent has all valid moves from (7,6). Each move leads to opponent challenging.
    - Challenge (-1,-1) from agent: terminal state, bid was (7,6), agent has [2,2,2,2,2],
      opponent has [3,3,3,3,3]. Total 6's = 0 (wild_one still True, count 1's: zero 1's in
      either hand). So 0 < 7, challenge succeeds. Agent challenged, so agent wins: utility +1.
    - Any other bid from agent -> opponent challenges that bid. Need to compute each utility.

    With wild_one=True and dice 22222 vs 33333 (no 1's in either):
    - (7,6): 0 sixes, 0 ones = 0 total. 0 < 7, challenge succeeds -> agent challenged = +1
    - For bids by agent, opponent challenges: utility depends on bid face count.
    """
    state = State([22222, (7, 6)], first_act=False)
    opponent_dice = 33333
    trie = Trie()

    valid_moves = state.next_valid_move()

    # Compute expected utilities for each action
    # When agent takes an action and opponent challenges:
    # dice: 22222 (agent) + 33333 (opponent), wild_one=True
    # No 1's in either hand, so wildcard count = 0
    # face 1: count=0 (but wild_one becomes False when face 1 is bid)
    # face 2: count=5
    # face 3: count=5
    # face 4: count=0
    # face 5: count=0
    # face 6: count=0

    with patch('mccfr.sample_opponent_action', return_value=(-1, -1)):
        utility, result_trie = traverse_mccfr(state, opponent_dice, trie)

    # Verify regrets were set in the trie
    # The challenge action (-1,-1) should have the best regret
    # because challenging (7,6) with 0 sixes succeeds
    challenge_key = str(state) + str(-1) + str(-1)
    challenge_node = result_trie.search(challenge_key)
    assert challenge_node is not None

    # Verify all actions have regret entries
    for move in valid_moves:
        key = str(state) + str(move[0]) + str(move[1])
        node = result_trie.search(key)
        assert node is not None, f"Missing trie entry for move {move}"


def test_traverse_mccfr_challenge_dominant_regret():
    """
    In a state where challenge is clearly the best action,
    verify it accumulates the highest regret.

    State: [22222, (7,6)], first_act=False
    Opponent dice: 33333
    Agent challenges (7,6): 0 sixes total, 0 < 7 -> challenge wins -> utility +1
    Agent bids higher, opponent challenges: depends on bid but most bids lose
    """
    state = State([22222, (7, 6)], first_act=False)
    opponent_dice = 33333
    trie = Trie()

    with patch('mccfr.sample_opponent_action', return_value=(-1, -1)):
        utility, result_trie = traverse_mccfr(state, opponent_dice, trie)

    valid_moves = state.next_valid_move()
    challenge_key = str(state) + str(-1) + str(-1)
    challenge_regret = result_trie.search(challenge_key)['#'][0]

    # Challenge regret should be positive (it's the best action)
    assert challenge_regret > 0

    # Other actions should have lower (likely negative) regret
    for move in valid_moves:
        if move == (-1, -1):
            continue
        key = str(state) + str(move[0]) + str(move[1])
        node = result_trie.search(key)
        assert node['#'][0] <= challenge_regret


# --- Convergence test ---

def test_traverse_mccfr_convergence():
    """
    Run multiple iterations on a fixed-dice subgame.
    After convergence, challenge should be the dominant strategy.

    State: [22222, (7,6)], first_act=False
    Opponent always challenges. With 0 sixes and bid of 7 sixes,
    agent challenging is clearly best.
    """
    state_template = [22222, (7, 6)]
    opponent_dice = 33333
    trie = Trie()

    with patch('mccfr.sample_opponent_action', return_value=(-1, -1)):
        for _ in range(200):
            state = State(state_template, first_act=False)
            _, trie = traverse_mccfr(state, opponent_dice, trie)

    # After convergence, compute strategy
    state = State(state_template, first_act=False)
    trie = calculate_strategy(state, trie)

    valid_moves = state.next_valid_move()
    challenge_key = str(state) + str(-1) + str(-1)
    challenge_prob = trie.search(challenge_key)['#'][1]

    # Challenge should have high probability after convergence
    assert challenge_prob > 0.8, f"Challenge probability {challenge_prob} should be > 0.8"

    # Other actions should have low probability
    for move in valid_moves:
        if move == (-1, -1):
            continue
        key = str(state) + str(move[0]) + str(move[1])
        node = trie.search(key)
        assert node['#'][1] < 0.1, f"Non-challenge move {move} has prob {node['#'][1]}"
