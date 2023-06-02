import random


class Dice:
    """
    A dice class
    """

    def __init__(self, num=5, pattern=None):
        self.num = num
        self.dice = []

        if pattern is None:
            for i in range(num):
                self.dice.append(random.randint(1, 6))
        else:
            for c in pattern:
                self.dice.append(int(c))

    def __repr__(self):
        return ''.join(str(die) for die in self.dice)

    def roll(self):
        for i in range(self.num):
            self.dice[i] = random.randint(1, 6)
        return self.dice
