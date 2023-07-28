from player import Player
from state import State
import random
import json

# random.seed(37)


def init_player_dice() -> int:
    """
    Initialize the dice for the player
    give the player five dices. each digit represents a dice from 1 to 6
    e.g. 12345 means 1, 2, 3, 4, 5
    """
    return random.randint(1, 6) * 10000 + random.randint(1, 6) * 1000 + random.randint(1, 6) * 100 \
        + random.randint(1, 6) * 10 + random.randint(1, 6)


def init_state(state: State) -> State:
    """
    Initialize the state of the game
    """
    if state.dice == 0:
        state.dice = init_player_dice()

    if state.first_act is None:
        state.first_act = random.choice([True, False])

    return state


class Game:
    """
    engine for the liar's dice game
    """

    def __init__(self, num_players=2):
        self.num_players = num_players
        self.players = []
        self.total_dice = {val: 0 for val in range(1, 7)}
        self.wild_one = True
        self.state = None  # agent's state
        self.player_dice = None  # player's dice
        self.strategy = None  # agent's strategy

    def start_game(self):
        """
        init state for the agent and assign dice to the player
        """
        self.state = init_state(State([]))

        # read the strategy from the file
        with open(f"output/{str(self.state.dice)}.json", "r") as fp:
            print(f'agent strategy {str(self.state.dice)} is loaded')
            self.strategy = json.load(fp)

        print("Game started!")
        self.player_dice = init_player_dice()
        print(f"Your dice is: {self.player_dice}")

    def play(self):
        """
        Simulate the game. The agent and the player take turns to play the game.
        Use prompt to get the player's input.
        """

        if self.state.first_act:
            print("Agent goes first.")
        else:
            print("Player goes first.")
        print('-' * 20)
        while not self.state.is_terminal():
            if self.state.player_of_current_turn():
                # agent's turn
                action_list = self.state.next_valid_move()
                action = random.choice(action_list)
                if action != (-1, -1):
                    print(f"agent's action is: bid {action[0]} {action[1]}'s")
                else:
                    print("agent's action is: challenge!")
                self.state.history.append(action)
            else:
                # player's turn, use prompt to get the player's input
                valid_moves = self.state.next_valid_move()
                action = input("Your action in the format of 'bid 2 3' or 'challenge': ")
                if action == "challenge":
                    action = (-1, -1)
                else:
                    action = tuple(map(int, action.split()[1:]))
                while action not in valid_moves:
                    print("Invalid action, please try again!")
                    action = input("Your action in the format of 'bid 2 3' or 'challenge': ")
                    if action == "challenge":
                        action = (-1, -1)
                    else:
                        action = tuple(map(int, action.split()[1:]))
                self.state.history.append(action)

        # terminal node, determine the winner
        print('-' * 20)
        print(f"agent's dice is: {self.state.dice}")
        if self.state.utility(self.player_dice) == 1:
            print("You win!")
        else:
            print("You lose!")

    def add_player(self, player: Player):
        self.players.append(player)

    def refresh_ttl_dice(self):
        """
        refresh the total dice in the current game
        """
        self.total_dice = {val: 0 for val in range(1, 7)}
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
            if self.wild_one:
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


if __name__ == "__main__":
    g = Game()
    g.start_game()
    g.play()
