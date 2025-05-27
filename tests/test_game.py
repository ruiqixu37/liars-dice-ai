from game import Game, init_player_dice


def test_game_initialization():
    """Test that game is initialized correctly"""
    game = Game()
    assert game.num_players == 2
    assert len(game.players) == 0
    assert game.total_dice == {val: 0 for val in range(1, 7)}
    assert game.wild_one is True
    assert game.state is None
    assert game.player_dice is None
    assert game.strategy is None


def test_init_player_dice():
    """Test that player dice initialization produces valid values"""
    dice = init_player_dice()
    # Convert to list of individual dice values
    dice_values = [int(d) for d in str(dice).zfill(5)]
    assert len(dice_values) == 5
    assert all(1 <= d <= 6 for d in dice_values)


def test_game_challenge():
    """Test the challenge functionality"""
    game = Game()
    # Test with a specific configuration
    game.total_dice = {1: 2, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1}

    # Test a valid challenge (bid is too high)
    assert game.challenge(41)  # Bidding 4 ones when there are only 2

    # Test an invalid challenge (bid is correct)
    assert not game.challenge(21)  # Bidding 2 ones when there are 2


def test_game_restart():
    """Test game restart functionality"""
    game = Game()
    # Add a player
    from player import Player
    from dice import Dice
    player = Player(Dice())
    game.add_player(player)

    # Store initial dice values
    initial_dice = player.dice.dice.copy()

    # Restart the game
    game.restart()

    # Check that dice were rolled
    assert player.dice.dice != initial_dice
    assert game.wild_one is True
