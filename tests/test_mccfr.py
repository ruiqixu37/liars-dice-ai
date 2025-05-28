import pytest
from mccfr import (
    calculate_strategy,
    update_strategy,
    sample_opponent_action,
    traverse_mccfr,
    traverse_mccfr_p
)
from state import State
from collections import defaultdict
import numpy as np
import os
import json

@pytest.fixture
def initial_state():
    """Create an initial state for testing"""
    return State([5, (2, 3)], first_act=True)

@pytest.fixture
def global_dict():
    """Create a global dictionary for testing"""
    return defaultdict(lambda: [0.0, 0.0, 0.0, False, 0.0])

def test_calculate_strategy(initial_state, global_dict):
    """Test that calculate_strategy returns a valid strategy"""
    # Set some initial regrets
    valid_moves = initial_state.next_valid_move()
    for move in valid_moves:
        child = initial_state.copy()
        child.update_history(move)
        global_dict[str(child)][0] = 1.0  # Set positive regret
    
    result_dict = calculate_strategy(initial_state, global_dict)
    
    # Check that strategy probabilities sum to 1
    strategy_sum = 0
    for move in valid_moves:
        child = initial_state.copy()
        child.update_history(move)
        strategy_sum += result_dict[str(child)][1]
    
    assert abs(strategy_sum - 1.0) < 1e-10
    assert all(0 <= result_dict[str(child)][1] <= 1 
              for move in valid_moves 
              for child in [initial_state.copy().update_history(move)])

def test_update_strategy_terminal_state(global_dict):
    """Test update_strategy with a terminal state"""
    terminal_state = State([5, (2, 3), (-1, -1)], first_act=True)
    result_dict = update_strategy(terminal_state, global_dict)
    assert result_dict == global_dict  # Should return unchanged dict for terminal state

def test_sample_opponent_action(initial_state):
    """Test that sample_opponent_action returns a valid action"""
    opponent_dice = 5
    action = sample_opponent_action(initial_state, opponent_dice)
    assert action in initial_state.next_valid_move()

def test_traverse_mccfr_terminal_state(global_dict):
    """Test traverse_mccfr with a terminal state"""
    terminal_state = State([5, (2, 3), (-1, -1)], first_act=True)
    opponent_dice = 5
    utility, result_dict = traverse_mccfr(terminal_state, opponent_dice, global_dict)
    assert isinstance(utility, float)
    assert isinstance(result_dict, defaultdict)

def test_traverse_mccfr_p_terminal_state(global_dict):
    """Test traverse_mccfr_p with a terminal state"""
    terminal_state = State([5, (2, 3), (-1, -1)], first_act=True)
    opponent_dice = 5
    utility, result_dict = traverse_mccfr_p(terminal_state, opponent_dice, global_dict)
    assert isinstance(utility, float)
    assert isinstance(result_dict, defaultdict)

def test_traverse_mccfr_chance_node(global_dict):
    """Test traverse_mccfr with a chance node (dice = 0)"""
    chance_state = State([0], first_act=True)
    opponent_dice = 5
    utility, result_dict = traverse_mccfr(chance_state, opponent_dice, global_dict)
    assert isinstance(utility, float)
    assert isinstance(result_dict, defaultdict)

def test_traverse_mccfr_p_chance_node(global_dict):
    """Test traverse_mccfr_p with a chance node (dice = 0)"""
    chance_state = State([0], first_act=True)
    opponent_dice = 5
    utility, result_dict = traverse_mccfr_p(chance_state, opponent_dice, global_dict)
    assert isinstance(utility, float)
    assert isinstance(result_dict, defaultdict) 