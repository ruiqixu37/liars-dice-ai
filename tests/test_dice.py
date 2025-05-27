from dice import Dice


def test_dice_initialization():
    """Test that dice are initialized with correct number of dice"""
    dice = Dice()
    assert len(dice.dice) == 5
    assert all(1 <= d <= 6 for d in dice.dice)


def test_dice_roll():
    """Test that rolling dice produces valid values"""
    dice = Dice()
    dice.roll()
    assert len(dice.dice) == 5
    assert all(1 <= d <= 6 for d in dice.dice)
    # Note: We can't assert that the dice changed because there's a small chance they could roll
    # the same values


def test_dice_representation():
    """Test the string representation of dice"""
    dice = Dice()
    dice_str = str(dice)
    assert isinstance(dice_str, str)
    assert len(dice_str) == 5  # Should show 5 numbers
