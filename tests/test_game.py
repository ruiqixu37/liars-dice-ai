from game import Game, init_player_dice, format_dice, format_action, \
    count_matching_dice, parse_player_action, load_trie


def test_game_initialization():
    """Test that game is initialized correctly"""
    game = Game()
    assert game.state is None
    assert game.player_dice is None
    assert game.agent_trie is None
    assert game.hints is False


def test_game_initialization_with_hints():
    """Test that game can be initialized with hints enabled"""
    game = Game(hints=True)
    assert game.hints is True


def test_init_player_dice():
    """Test that player dice initialization produces valid values"""
    dice = init_player_dice()
    # Convert to list of individual dice values
    dice_values = [int(d) for d in str(dice).zfill(5)]
    assert len(dice_values) == 5
    assert all(1 <= d <= 6 for d in dice_values)


def test_format_dice():
    """Test dice formatting"""
    assert format_dice(12345) == [1, 2, 3, 4, 5]
    assert format_dice(66666) == [6, 6, 6, 6, 6]
    assert format_dice(11111) == [1, 1, 1, 1, 1]


def test_format_action():
    """Test action formatting"""
    assert format_action((3, 4)) == "bid 3 4"
    assert format_action((-1, -1)) == "challenge"


def test_count_matching_dice():
    """Test dice counting with and without wild ones"""
    # face=3, no wilds
    assert count_matching_dice(33333, 33333, 3, False) == 10
    # face=3, with wilds (1s count)
    assert count_matching_dice(11133, 11133, 3, True) == 10
    # face=1, wilds active but face is 1 so no extra
    assert count_matching_dice(11133, 11133, 1, True) == 6


def test_parse_player_action():
    """Test parsing player input"""
    assert parse_player_action("bid 3 4") == (3, 4)
    assert parse_player_action("challenge") == (-1, -1)
    assert parse_player_action("CHALLENGE") == (-1, -1)
    assert parse_player_action("BID 3 4") == (3, 4)
    assert parse_player_action("garbage") is None
    assert parse_player_action("bid abc") is None
    assert parse_player_action("") is None


def test_load_trie_missing():
    """Test that loading a nonexistent trie returns None"""
    assert load_trie(99999) is None
