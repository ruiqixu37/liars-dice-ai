from state import State


def test_state_initialization():
    """Test that state is initialized correctly"""
    state = State([])
    assert state.history == []
    assert state.dice == 0
    assert state.first_act is None


def test_state_valid_moves():
    """Test that valid moves are generated correctly"""
    state = State([])
    valid_moves = state.next_valid_move()
    # First move should be all possible bids
    assert len(valid_moves) > 0
    assert all(isinstance(move, tuple) for move in valid_moves)
    assert all(len(move) == 2 for move in valid_moves if move != (-1, -1))


def test_state_terminal():
    """Test terminal state detection"""
    # Create a state with a challenge
    state = State([(2, 3), (-1, -1)])  # A bid followed by a challenge
    assert state.is_terminal()


def test_state_utility():
    """Test utility calculation"""
    state = State([(2, 3), (-1, -1)])  # A bid followed by a challenge
    # Test with a specific dice configuration
    test_dice = 11111  # All ones
    utility = state.utility(test_dice)
    assert utility in [0, 1]  # Should be either 0 or 1


def test_state_player_turn():
    """Test player turn alternation"""
    state = State([])
    # First player should be determined by first_act
    state.first_act = True
    assert state.player_of_current_turn()  # Should be True for first player

    # After a move, it should switch
    state.history.append((2, 3))
    assert not state.player_of_current_turn()  # Should be False for second player
