from state import State
from game import init_state


def test_state_initialization():
    """Test that state is initialized correctly"""
    state = State([])
    assert state.history == []
    assert state.dice == 0
    assert state.first_act is None


def test_state_valid_moves():
    """Test that valid moves are generated correctly"""
    state = State([])
    state = init_state(state)
    valid_moves = state.next_valid_move()
    # First move should be all possible bids
    assert len(valid_moves) > 0
    assert all(isinstance(move, tuple) for move in valid_moves)
    assert all(len(move) == 2 for move in valid_moves if move != (-1, -1))


def test_state_terminal():
    """Test terminal state detection"""
    # Create a state with a challenge
    state = State([12345, (2, 3), (-1, -1)])  # A bid followed by a challenge
    assert state.is_terminal()


def test_state_utility():
    """Test utility calculation"""
    state = State([11111, (2, 3), (-1, -1)], first_act=True)  # A bid followed by a challenge
    # Test with a specific dice configuration
    test_dice = 34562
    utility = state.utility(test_dice)
    assert utility == 1

    state = State([23456, (4, 3), (-1, -1)], first_act=True)
    # Test with a specific dice configuration
    test_dice = 23456
    utility = state.utility(test_dice)
    assert utility == -1

    state = State([23456, (2, 6), (-1, -1)], first_act=True)
    # Test with a specific dice configuration
    test_dice = 23456
    utility = state.utility(test_dice)
    assert utility == 1

    state = State([23456, (2, 6), (-1, -1)], first_act=False)
    # Test with a specific dice configuration
    test_dice = 23456
    utility = state.utility(test_dice)
    assert utility == -1


def test_state_player_turn():
    """Test player turn alternation"""
    state = State([12345])
    # First player should be determined by first_act
    state.first_act = True
    assert state.player_of_next_move() == 1

    # After a move, it should switch
    state.history.append((2, 3))
    assert state.player_of_next_move() == 0


def test_state_copy_preserves_wild_one():
    """Copy with wild_one=False should preserve the flag."""
    state = State([22222, (3, 1)], first_act=True)
    # After bidding face 1, wild_one should be False
    assert state.wild_one is False

    copied = state.copy()
    assert copied.wild_one is False
    assert copied.dice == state.dice
    assert copied.history == state.history
    assert copied.first_act == state.first_act


def test_state_copy_preserves_wild_one_true():
    """Copy with wild_one=True (default) should preserve it."""
    state = State([22222, (3, 4)], first_act=True)
    assert state.wild_one is True

    copied = state.copy()
    assert copied.wild_one is True


def test_state_str_challenge_encoding():
    """Verify __str__ handles (-1,-1) challenge — the '-' chars must work as trie keys."""
    state = State([12345, (2, 3), (-1, -1)], first_act=True)
    s = str(state)

    # Should contain the dice, first_act flag, and encoded history
    assert s.startswith("12345")
    # The challenge should be encoded as "-1-1"
    assert "-1-1" in s

    # Verify we can use this as a trie key
    from trie import Trie
    trie = Trie()
    trie.insert(s, strategy_p=0.5)
    node = trie.search(s)
    assert node is not None
    assert node['#'][1] == 0.5


def test_wild_one_face_ordering():
    """Verify face=1 is a valid bid from face=4 when wild_one=True."""
    state = State([22222, (3, 4)], first_act=True)
    assert state.wild_one is True

    valid_moves = state.next_valid_move()
    # Face 1 wraps around when wild_one=True
    assert (3, 1) in valid_moves
    assert (3, 5) in valid_moves
    assert (3, 6) in valid_moves


def test_state_copy_independence():
    """Modifying a copy should not affect the original."""
    state = State([22222, (3, 4)], first_act=True)
    copied = state.copy()
    copied.dice = 33333
    copied.history.append((-1, -1))
    copied.wild_one = False

    assert state.dice == 22222
    assert len(state.history) == 1
    assert state.wild_one is True
