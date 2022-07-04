from re import T
from dice import Dice
from player import Player
import random

random.seed(37)

class Game:
    """
    engine for the liar's dice game 
    """
    def __init__(self, num_players = 2):
        self.num_players = num_players
        self.players = []
        self.total_dice = {val:0 for val in range(1,7)}
        self.wild_one = True
    
    def add_player(self, player: Player):
        self.players.append(player)
    
    def refresh_ttl_dice(self):
        """
        refresh the total dice in the current game
        """
        self.total_dice = {val:0 for val in range(1,7)}
        for p in self.players:
            for d in p.dice.dice:
                self.total_dice[d] += 1

    def challenge(self, bid: int):
        """
        challenge a player's bid
        bid is represented as a two-digit integer, where the first digit is the 
        number of dices, and the second digit is the value of the dices

        this implies the mamximum bid is 99 and the maximum number of dices for 
        a bid is 9.

        return True if the bid is not accepted, False otherwise
        """
        num_dice_bid = bid // 10
        value_dice = bid % 10

        # the value of the bid is 1
        if value_dice == 1:
            num_dice_game = self.total_dice[1]
        else:
            if self.wild_one == True:
                num_dice_game = self.total_dice[1] + self.total_dice[value_dice]
            else:
                num_dice_game = self.total_dice[value_dice]
        
        return num_dice_game < num_dice_bid
    
    def restart(self):
        """
        restart the game, holding players fixed
        """
        self.wild_one = True
        for p in self.players:
            p.dice.roll()
        self.refresh_ttl_dice()
        
def pick_valid_move(node:str) -> str:
    if node.endswith('c'): 
        return ''
    
    bid = [str(i) for i in range(3, 10)]
    value = [str(i) for i in range(1, 7)]

    if len(node) == 10:
        valid_move = [b + v for b in bid for v in value]
        
    else:
        last_bid = int(node[-2])
        last_value = int(node[-1])
        
        assert(last_bid <= 9 and last_value <= 6)
        
        wild_one = not '1' in node[10:]
        
        valid_move = []
        # get valid moves with the same bid
        if wild_one:
            greater_values = [str(i) for i in range(last_value + 1, 7)] + ['1']
        else:
            greater_values = [str(i) for i in range(last_value + 1, 7)]
            
        valid_move += [str(last_bid) + greater_val for greater_val in greater_values]
        
        # get valid moves with greater bid
        greater_bid = [str(i) for i in range(last_bid + 1, 10)]
        
        if wild_one:
            valid_move += [b + v for b in greater_bid for v in value]
        else:
            valid_move += [b + v for b in greater_bid for v in value[1:]] # no 1's
        
        valid_move += ['c']
        # print(valid_move)
    return random.choice(valid_move)

def result(node:str) -> str:
    assert(node.endswith('c') and len(node) >= 12)
    
    bid = node[-3]
    val = node[-2]
    
    wild_one = not '1' in node[10:]
    
    dice = {str(val):0 for val in range(1,7)}
    for d in node[:10]:
        dice[d] += 1
        
    # wins if the bid is fake
    if wild_one:
        return int(bid) > dice[val] + dice['1']
    else:
        return int(bid) > dice[val]

# if __name__ == "__main__":
#     node = '123451234534c'
#     valid_move = result(node)
#     print(valid_move)

#     d1 = Dice(pattern='33451')
#     d2 = Dice(pattern='33451')
#     p1 = Player(dice = d1, name = "p1")
#     p2 = Player(dice = d2, name = "p2")
#     game = Game()
#     game.add_player(p1)
#     game.add_player(p2)

#     print(p1.dice)
#     print(p2.dice)

#     game.refresh_ttl_dice()

#     bid = random.randint(0,9) * 10 + random.randint(1,6)
#     print(f"the bid is {bid // 10} {bid % 10}'s")

#     print(f"the number of dice in the game is: {game.total_dice}")

#     print(f"the challenge result is: {game.challenge(bid)}")