from game import Game
import game
from dice import Dice
from player import Player
import pandas as pd
import numpy as np
import random
import time 

class MCTS_Trainer:
    def __init__(self, game: Game, time, comp_power: int):
        assert(isinstance(game, Game))
        assert(game.players != [])
        self.game = game
        self.trees = pd.DataFrame(columns=['parent', 'children', 'wins', 
                                            'visits', 'wins', 'uct_value'])
        self.time = time
        self.comp_power = comp_power

    def resources_left(time, comp_power):
        return time.time() < time and comp_power > 0

    def monte_carlo_tree_search(self, root):
        while self.resources_left(self.time, self.comp_power):
            leaf = self.traverse(root)
            simulation_result = self.rollout(leaf)
            self.backpropagate(leaf, simulation_result)
            
        return self.best_child(root)

    def fully_expanded(self, node):
        return self.trees.loc[node]['children'] == []

    def best_uct(self, node) -> str:
        parent = self.trees.loc[node].copy()
        children = parent['children'].values # type is numpy.ndarray of str

        # filter children nodes
        children = self.trees.loc[children]
        best_child = children.loc[children['uct_value'] == max(children['uct_value'])] # type is pandas.DataFrame

        return best_child['id'].values[0]

    def pick_unvisited(children): # TODO: need to check later
        for child in children:
            if child.visits == 0:
                return child
        return None

    # function for node traversal
    def traverse(self, node):
        while self.fully_expanded(node):
            node = self.best_uct(node)
        
        # TODO: what is the following line for?            
        # in case no children are present / node is terminal 
        return self.pick_unvisited(node.children) or node

    def non_terminal(node):
        return not node.endswith('c') # c stands for challenge

    # function for the result of the simulation
    def rollout(self, node):
        while self.non_terminal(node):
            node = self.rollout_policy(node)
        return result(node)

    def pick_random(children):
        return random.choice(children)

    # function for randomly selecting a child node
    def rollout_policy(self, node): 
        # TODO: need to check later
        
        # check if node has children
        if self.fully_expanded(node):
            # randomly create a valid child for node
            child = game.pick_valid_move(node)
            
            # TODO: when to update the visits and values??
            # fill data to tree table
            self.trees.loc[child] = {'parent': node, 'children': [], 
                                     'wins': 0, 'visits': 0, 'value': 0}
        
        return self.pick_random(node.children)

    def is_root(node): # TODO: need to check later
        return node.parent == None

    def update_stats(node, result): # TODO: need to check later
        node.visits += 1
        node.wins += result
        return node.wins / node.visits

    # function for backpropagation
    def backpropagate(self, node, result):
        if self.is_root(node): return
        node.stats = self.update_stats(node, result)
        self.backpropagate(node.parent)

    # function for selecting the best child
    # node with highest number of visits
    def best_child(self, node):
        # pick child with highest number of visits
        return max(node.children, key=lambda child: child.visits)


if __name__ == '__main__':
    d1 = Dice(pattern='33451')
    d2 = Dice(pattern='33451')
    p1 = Player(dice = d1, name = "p1")
    p2 = Player(dice = d2, name = "p2")
    game = Game()
    game.add_player(p1)
    game.add_player(p2)

    train_time = 60 * 10 # 10 minutes
    trainer = MCTS_Trainer(game, time.time() + train_time, comp_power = 10) 
