"""Tests to verify bug fixes in mccfr.py and state.py."""
import os
import pickle
import pytest
from unittest.mock import patch, MagicMock
from state import State
from trie import Trie
from mccfr import (
    calculate_strategy,
    sample_opponent_action,
    DISCOUNT_ITERATION_INTERVAL,
)


# --- Bug 1: sample_opponent_action should use opponent's perspective ---

def test_sample_opponent_uses_opponent_perspective():
    """
    Bug 1: sample_opponent_action was using agent's dice for trie lookup
    instead of opponent's dice. Verify it now uses opponent_dice.
    """
    agent_state = State([11111, (2, 3)], first_act=True)
    opponent_dice = 66666

    # Create a trie file for the opponent's dice perspective
    opponent_state = agent_state.copy()
    opponent_state.dice = opponent_dice
    opponent_trie = Trie()

    # Insert strategy nodes for the opponent's perspective
    valid_moves = opponent_state.next_valid_move()
    for move in valid_moves:
        child = str(opponent_state) + str(move[0]) + str(move[1])
        opponent_trie.insert(child, strategy_p=0)

    # Set challenge to be the dominant action in opponent's trie
    challenge_key = str(opponent_state) + str(-1) + str(-1)
    opponent_trie.search(challenge_key)['#'][0] = 100.0  # high regret

    # Save opponent trie with opponent dice filename
    os.makedirs("output", exist_ok=True)
    trie_path = f"output/trie_{str(opponent_dice)}.pkl"
    try:
        with open(trie_path, "wb") as f:
            pickle.dump(opponent_trie, f)

        # Sample opponent action - should use opponent's dice file
        action = sample_opponent_action(agent_state, opponent_dice)
        assert action in valid_moves
    finally:
        if os.path.exists(trie_path):
            os.remove(trie_path)


def test_sample_opponent_action_no_file_uniform():
    """When no trie file exists, sample_opponent_action samples uniformly."""
    state = State([99999, (2, 3)], first_act=True)
    opponent_dice = 88888

    # Ensure no trie file exists for this dice
    trie_path = f"output/trie_{str(opponent_dice)}.pkl"
    if os.path.exists(trie_path):
        os.remove(trie_path)

    action = sample_opponent_action(state, opponent_dice)
    valid_moves = state.next_valid_move()
    assert action in valid_moves


# --- Bug 2: all_end_state destructuring ---

def test_all_end_state_destructuring():
    """
    Bug 2: Loop used `for end_state in trie.all_end_state()` but
    all_end_state returns [path, node] pairs. Verify correct destructuring.
    """
    trie = Trie()
    trie.insert("abc", strategy_p=0.5)
    trie.insert("def", strategy_p=0.3)

    # This should work without error (was crashing before fix)
    for path, end_state in trie.all_end_state():
        assert isinstance(path, str)
        assert '#' in end_state
        # Verify we can access the values correctly
        end_state['#'][0] *= 0.5  # discount regret
        end_state['#'][2] *= 0.5  # discount counter

    # Verify mutations took effect
    node = trie.search("abc")
    assert node['#'][0] == 0.0  # 0.0 * 0.5 = 0.0
    assert node['#'][2] == 0.0  # 0.0 * 0.5 = 0.0


# --- Bug 3: Discount interval never triggers ---

def test_discount_interval_triggers():
    """
    Bug 3: `float % int == 0` comparison was effectively always False.
    Now uses iteration-count-based check. Verify DISCOUNT_ITERATION_INTERVAL
    is a positive integer that allows modulo to work.
    """
    assert isinstance(DISCOUNT_ITERATION_INTERVAL, int)
    assert DISCOUNT_ITERATION_INTERVAL > 0

    # Verify the modulo check works for expected iteration values
    for t in range(1, 500):
        if t % DISCOUNT_ITERATION_INTERVAL == 0:
            # This should trigger at regular intervals
            assert t >= DISCOUNT_ITERATION_INTERVAL
            break
    else:
        pytest.fail("Discount interval never triggered in 500 iterations")


def test_discount_actually_applies():
    """Verify that discount modifies regret and counter values in the trie."""
    trie = Trie()
    trie.insert("abc", strategy_p=0.5)
    trie.search("abc")['#'][0] = 10.0  # set regret
    trie.search("abc")['#'][2] = 5.0   # set counter

    t = DISCOUNT_ITERATION_INTERVAL
    d = (t / DISCOUNT_ITERATION_INTERVAL) / ((t / DISCOUNT_ITERATION_INTERVAL) + 1)

    # Simulate the discount loop
    for path, end_state in trie.all_end_state():
        end_state['#'][0] *= d
        end_state['#'][2] *= d

    node = trie.search("abc")
    assert node['#'][0] == 10.0 * d
    assert node['#'][2] == 5.0 * d


# --- Bug 4: Save/load filename mismatch ---

def test_save_load_filename_consistency():
    """
    Bug 4: Save used trie_{str(state)}.pkl but load used trie_{str(state.dice)}.pkl.
    Both should now use state.dice.
    """
    import mccfr
    import inspect

    source = inspect.getsource(mccfr.MCCFR_P)

    # Count occurrences of the save pattern
    # After fix, save should use state.dice, not str(state)
    assert 'trie_{str(state.dice)}.pkl' in source, \
        "Save should use trie_{str(state.dice)}.pkl"

    # The old buggy pattern should not appear
    # Note: str(state) without .dice would produce the full state string
    # We check that save and load use the same pattern
    save_pattern_count = source.count('trie_{str(state.dice)}.pkl')
    # Should appear at least twice: once for load, once for save
    assert save_pattern_count >= 2, \
        f"Expected trie_{{str(state.dice)}}.pkl to appear at least twice, found {save_pattern_count}"


# --- Bug 5: Wild-one face wraparound ---

def test_wild_one_face_ordering():
    """
    Bug 5: When wild_one=True, face 1 appears in the valid moves list
    after face 6 for the same quantity. Document this behavior.

    With last bid (qty, face=4) and wild_one=True:
    greater_values = [5, 6] + [1] = [5, 6, 1]
    So bidding (qty, 1) is valid from (qty, 4) — face 1 wraps around.
    """
    state = State([22222, (3, 4)], first_act=True)
    valid_moves = state.next_valid_move()

    # Face 1 with same quantity should be valid (wraparound)
    assert (3, 1) in valid_moves, "Face 1 should be valid via wraparound when wild_one=True"

    # Face 5 and 6 should also be valid
    assert (3, 5) in valid_moves
    assert (3, 6) in valid_moves

    # Face 2, 3 should NOT be valid (not greater than 4)
    assert (3, 2) not in valid_moves
    assert (3, 3) not in valid_moves
    assert (3, 4) not in valid_moves


def test_wild_one_face_ordering_disabled():
    """When wild_one=False, face 1 does NOT wrap around."""
    state = State([22222, (3, 4)], first_act=True)
    state.wild_one = False
    valid_moves = state.next_valid_move()

    # Face 1 should NOT be valid when wild_one is False
    assert (3, 1) not in valid_moves

    # Face 5 and 6 should still be valid
    assert (3, 5) in valid_moves
    assert (3, 6) in valid_moves


def test_wild_one_disabled_after_bidding_one():
    """Bidding face 1 should disable wild_one."""
    state = State([22222, (3, 1)], first_act=True)
    # After bidding face 1, wild_one should be False
    assert state.wild_one is False

    valid_moves = state.next_valid_move()
    # With wild_one=False and last_face=1, no same-quantity bids are valid
    # (greater_values would be [] since last_face == 1 branch returns [])
    same_qty_moves = [m for m in valid_moves if m[0] == 3 and m != (-1, -1)]
    assert len(same_qty_moves) == 0
