from copy import copy


class State:
    def __init__(self, state, first_act=None):
        """
        state: tuple, (dice, (quantity, face), (quantity, face), ..., (-1, -1)))
        -1 stands for challenge
        first_act: bool, True if the agent is the first to act
        """
        if len(state) >= 1:
            self.dice = state[0]
        else:
            self.dice = 0
        self.wild_one = True
        self.children = []
        self.first_act = first_act
        self.history = list(state[1:])

    def __str__(self) -> str:
        dice = str(self.dice)
        history = ''.join([str(bid[0])+str(bid[1]) for bid in self.history])
        return f'{dice}{history}'

    def next_valid_move(self):
        """
        Returns a list of valid moves for the current turn

        To be memory efficient, a couple of priors are applied:
        1. the first move can bet a quanity of 2 to 4 with any face
        2. a move can bet a quantity of up to 8
        3. any move's quantity can be at most 2 greater than the last move's quantity

        Then, per the rules of the game, the moves should:
        1. be greater than the last move in quantity, or greater in face if the quantity is the same
        """
        assert (self.dice != 0)

        if len(self.history) == 0:
            return [(i, j) for i in range(2, 5) for j in range(1, 7)]

        if self.history[-1] == (-1, -1):
            return []

        last_quantity, last_face = self.history[-1]

        valid_moves = []

        # get valid moves with the same bid
        if self.wild_one:
            greater_values = [i for i in range(last_face + 1, 7)] + [1]
        else:
            if last_face != 1:
                greater_values = [i for i in range(last_face + 1, 7)]
            else:
                greater_values = []

        valid_moves += [(last_quantity, face) for face in greater_values]

        # get valid moves with greater bid
        greater_bid = [i for i in range(last_quantity + 1, min(last_quantity + 3, 9))]

        valid_moves += [(bid, face) for bid in greater_bid for face in range(1, 7)]

        valid_moves += [(-1, -1)]  # challenge

        return valid_moves

    def player_of_next_move(self):
        """
        Returns the player number of the next_move
        1 for agent, 0 for opponent
        """
        assert self.dice != 0 and self.first_act is not None

        if self.first_act:
            return len(self.history) % 2 == 0
        else:
            return len(self.history) % 2 == 1

    def update_child(self, move):
        """
        :param move: tuple, (quantity, face)
        """
        assert move in self.next_valid_move()

        if move not in self.children:
            self.children.append(move)

    def update_history(self, move):
        """
        :param move: tuple, (quantity, face)
        """
        assert move in self.next_valid_move()

        if move[1] == 1:
            self.wild_one = False

        self.history.append(move)

    def utility(self, opponent_dice):
        """
        param: opponent_dice, int, opponent's dice number

        return: int, 1 if the agent wins, -1 if the agent loses
        """

        assert self.history[-1] == (-1, -1)  # last move is challenge

        last_bid = self.history[-2]
        last_bid_quantity, last_bid_face = last_bid

        total_face = 0

        # get the total number of last bid face in the game
        # including the opponent's dice
        # dice is a 5-digit number, each digit represents a dice
        # e.g. 12345 means 1, 2, 3, 4, 5
        # iterate through each digit and count the number of last bid face
        for digit in str(self.dice):
            if int(digit) == last_bid_face:
                total_face += 1
            elif self.wild_one and int(digit) == 1:
                total_face += 1

        for digit in str(opponent_dice):
            if int(digit) == last_bid_face:
                total_face += 1
            elif self.wild_one and int(digit) == 1:
                total_face += 1

        # determine who challenged based on self.first_act
        # flip player_of_next_move() because it returns the "next" player
        # but we want the "current" player
        if self.player_of_next_move() == 0:  # agent made the challenge
            agent_challenge = 1
        else:  # opponent made the challenge
            agent_challenge = -1

        if total_face >= last_bid_quantity:  # challenge failed
            return -1 * agent_challenge
        else:
            return 1 * agent_challenge

    def copy(self):
        """
        get a copy of the current state
        """
        new_state = State((self.dice, *self.history), self.first_act)
        new_state.children = copy(self.children)
        new_state.wild_one = self.wild_one

        return new_state

    def is_terminal(self):
        """
        Determine if the current state is a terminal state
        """
        return len(self.history) >= 2 and self.history[-1] == (-1, -1)
