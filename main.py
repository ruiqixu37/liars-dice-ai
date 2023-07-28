from collections import defaultdict


def next_valid_move(last_face, last_quantity):
    """
    Returns a list of valid moves for the current turn

    To be memory efficient, a couple of rules are applied:
    1. the first move can bet a quanity of 2 to 4 with any face
    2. a move can bet a quantity of up to 8
    3. any move's quantity can be at most 2 greater than the last move's quantity

    Then, per the rules of the game, the moves should:
    1. be greater than the last move in quantity, or greater in face if the quantity is the same
    """

    valid_moves = []

    # get valid moves with the same bid
    greater_values = [i for i in range(last_face + 1, 7)]

    valid_moves += [(last_quantity, face) for face in greater_values]

    # get valid moves with greater bid
    greater_bid = [i for i in range(last_quantity + 1, min(last_quantity + 3, 9))]

    valid_moves += [(bid, face) for bid in greater_bid for face in range(2, 7)]

    return valid_moves


if __name__ == '__main__':

    # init default dict
    d = defaultdict(lambda: 1)

    # iterate over all possible moves reversely and use
    # dynamic programming to calculate the number of possible moves
    for last_quantity in range(7, 2, -1):
        for last_face in range(6, 0, -1):
            moves = next_valid_move(last_face, last_quantity)
            for move in moves:
                d[(last_quantity, last_face)] += d[move]

    print(d[(3, 1)])
