from dice import Dice

class Player:
    """
    Class for liar's dice agents

    each person has five dices 
    """

class Player:
    """
    Class for liar's dice agents

    by default each player has five dices 
    """
    def __init__(self, dice: Dice, name = "anony"):
        self.name = name
        self.dice = dice
        self.score = 0

    def __repr__(self):
        return f'{self.name} has {self.score} points'
    
    def roll(self):
        self.dice.roll()