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


# --- Utility with wild_one edge cases ---

def test_utility_wild_one_counts_ones_as_wildcards():
    """With wild_one=True, 1s count as the bid face. Agent=11111, opp=11111, bid (5,2).
    All 10 dice are 1s, wild_one treats them as 2s -> total=10 >= 5, challenge fails."""
    # Agent bids (5,2), opponent challenges. first_act=True -> agent bids (even), opp challenges (odd)
    state = State([11111, (5, 2), (-1, -1)], first_act=True)
    assert state.wild_one is True
    utility = state.utility(11111)
    # Opponent challenged. player_of_next_move at terminal: history len=2, first_act=True -> 2%2==0 -> agent's "next" turn
    # So player_of_next_move()==1 (True), meaning opponent made the challenge (agent_challenge=-1)
    # total 2s: 0 actual 2s, but 10 ones as wildcards = 10. 10 >= 5, challenge fails -> -1 * -1 = 1
    # Wait, that means opponent loses. Utility from agent perspective = +1? No: challenge failed means
    # the challenger loses. Opponent challenged and failed -> agent wins -> utility = -1 * agent_challenge
    # agent_challenge = -1 (opponent challenged), challenge failed -> return -1 * -1 = 1
    assert utility == 1  # opponent challenged and failed


def test_utility_bidding_face_one_disables_wild():
    """Bidding face=1 disables wild_one. Agent=11111, opp=11111, bid (5,1), challenged.
    wild_one=False, count only actual 1s = 10 >= 5, challenge fails."""
    state = State([11111, (5, 1), (-1, -1)], first_act=True)
    assert state.wild_one is False  # face=1 was bid
    utility = state.utility(11111)
    # 10 actual 1s, no wildcards needed. 10 >= 5, challenge fails.
    # Opponent challenged (same logic as above), fails -> agent wins
    assert utility == 1


def test_utility_wild_one_mixed_dice():
    """wild_one=True with mixed dice. Agent=12345, opp=12345, bid (4,3), challenged.
    Count 3s: 1+1=2. Count 1s (wildcards): 1+1=2. Total=4 >= 4, challenge fails."""
    state = State([12345, (4, 3), (-1, -1)], first_act=True)
    assert state.wild_one is True
    utility = state.utility(12345)
    # Opponent challenged, 4 >= 4, challenge fails -> agent wins
    assert utility == 1


def test_utility_wild_one_disabled_by_prior_bid():
    """A prior bid on face=1 disables wild_one for utility calculation.
    Agent=12345, opp=12345, bids: (2,1), (3,3), challenge.
    wild_one=False because (2,1) was bid. Count only 3s: 1+1=2. 2 < 3, challenge succeeds."""
    state = State([12345, (2, 1), (3, 3), (-1, -1)], first_act=True)
    assert state.wild_one is False
    utility = state.utility(12345)
    # history len=3, first_act=True -> 3%2==1 -> player_of_next_move returns False (0)
    # player_of_next_move()==0 -> agent made the challenge, agent_challenge=1
    # total 3s: 1+1=2 (no wildcards). 2 < 3, challenge succeeds -> return 1 * 1 = 1
    assert utility == 1  # agent challenged and succeeded


def test_utility_wild_one_not_double_count_face_one():
    """When bid face IS 1 and wild_one=False, 1s are counted as face matches only, not wildcards.
    Agent=11111, opp=22222, bid (3,1), challenged.
    wild_one=False. Count face 1: 5+0=5. 5 >= 3, challenge fails."""
    state = State([11111, (3, 1), (-1, -1)], first_act=True)
    assert state.wild_one is False
    utility = state.utility(22222)
    # Opponent challenged, 5 >= 3, challenge fails -> agent wins
    assert utility == 1


# --- next_valid_move boundary cases ---

def test_valid_moves_first_bid_exactly_18():
    """First bid should have exactly 18 moves: qty 2-4, face 1-6, no challenge."""
    state = State([12345], first_act=True)
    valid_moves = state.next_valid_move()
    assert len(valid_moves) == 18
    assert (-1, -1) not in valid_moves
    for q in range(2, 5):
        for f in range(1, 7):
            assert (q, f) in valid_moves


def test_valid_moves_from_7_6():
    """From (7,6): face 6 is max, no same-qty bids (except wraparound 1 if wild_one).
    Higher qty: 8 valid (min(7+3,9)=9, range(8,9)=[8]). Plus challenge."""
    state = State([12345, (7, 6)], first_act=True)
    assert state.wild_one is True
    valid_moves = state.next_valid_move()

    # Same-qty: only (7, 1) via wild_one wraparound
    assert (7, 1) in valid_moves

    # Higher qty: (8, 1) through (8, 6)
    for f in range(1, 7):
        assert (8, f) in valid_moves

    # No qty 9+
    assert not any(m[0] >= 9 for m in valid_moves if m != (-1, -1))

    # Challenge available
    assert (-1, -1) in valid_moves

    # Total: (7,1) + 6 qty-8 bids + challenge = 8
    assert len(valid_moves) == 8


def test_valid_moves_from_8_6_with_wild_one():
    """From (8,6) with wild_one=True: only (8,1) wraparound and challenge."""
    state = State([12345, (8, 6)], first_act=True)
    assert state.wild_one is True
    valid_moves = state.next_valid_move()
    assert (8, 1) in valid_moves
    assert (-1, -1) in valid_moves
    assert len(valid_moves) == 2


def test_valid_moves_from_8_6_wild_one_false():
    """From (8,6) with wild_one=False: no wraparound. Only challenge."""
    state = State([12345, (8, 6)], first_act=True)
    state.wild_one = False
    valid_moves = state.next_valid_move()
    assert valid_moves == [(-1, -1)]


def test_valid_moves_from_8_1_wild_one_false():
    """From (8,1) with wild_one=False: face=1 lowest, no same-qty. Only challenge."""
    state = State([12345, (8, 1)], first_act=True)
    state.wild_one = False
    valid_moves = state.next_valid_move()
    assert valid_moves == [(-1, -1)]


def test_valid_moves_quantity_cap_at_8():
    """Quantity should never exceed 8 in valid moves."""
    state = State([12345, (6, 3)], first_act=True)
    valid_moves = state.next_valid_move()
    for move in valid_moves:
        if move != (-1, -1):
            assert move[0] <= 8, f"Quantity {move[0]} exceeds cap of 8"
