import pytest
from state import State
from game import Game
# from mcts import MCTSNode
# from mccfr import MCCFR


@pytest.fixture
def empty_state():
    """Fixture for an empty game state"""
    return State([])


@pytest.fixture
def game_instance():
    """Fixture for a game instance"""
    return Game()


# @pytest.fixture
# def mcts_node(empty_state):
#     """Fixture for an MCTS node"""
#     return MCTSNode(empty_state)


# @pytest.fixture
# def mccfr_instance():
#     """Fixture for an MCCFR instance"""
#     return MCCFR()
