from game import Game
from game import pick_valid_move, result
from dice import Dice
from player import Player
import pandas as pd
import numpy as np
import random
import time 

random.seed(37)

class MCTS_Trainer:
    def __init__(self, game: Game, time, comp_power: int):
        assert(isinstance(game, Game))
        assert(game.players != [])
        self.game = game
        self.trees = pd.DataFrame(columns=['parent', 'children', 'wins', 
                                            'visits'])
        self.time = time
        self.comp_power = comp_power

    def resources_left(self, end_time, comp_power):
        return time.time() < end_time and comp_power > 0

    def monte_carlo_tree_search(self, root):
        self.init_tree(root)
        
        while self.resources_left(self.time, self.comp_power):
            leaf = self.traverse(root)
            extended_leaf, simulation_result = self.rollout(leaf)
            self.backpropagate(extended_leaf, simulation_result)
            
        return self.best_child(root)
    
    def init_tree(self, root):
        self.trees.loc[root] = {'parent': None, 'children': [], 
                         'wins': 0, 'visits': 0}
    
    def fully_expanded(self, node):
        return self.trees.loc[node]['children'] == []

    def best_uct(self, node) -> str:
        parent = self.trees.loc[node].copy()
        parent_visit = parent['visits']
        
        children = parent['children'] # type is numpy.ndarray of str

        # filter children nodes
        children = self.trees.loc[children]
        children = children.loc[children['visits'] > 0]

        # explore ratio = 0.1
        uct_values = children['wins'] / children['visits'] + \
            0.1 * np.sqrt(np.log(parent_visit) / children['visits'].astype('float'))
        
        best_children = uct_values[uct_values == max(uct_values)].index
       
        best_child = random.choice(best_children)

        return best_child

    def pick_unvisited(self, parent):
        children = self.trees.loc[parent]['children'] # type is list of str

        # filter children nodes
        children = self.trees.loc[children].index
        
        if len(children) == 0:
            return None
        else:
            return random.choice(children)

    # function for node traversal
    def traverse(self, node):
        while not self.fully_expanded(node):
            # explore ratio = 0.1
            random_num = np.random.rand()
            if random_num < 0.1:
                child = pick_valid_move(node)
            
                # TODO: when to update the visits and values??
                # fill data to tree table
                self.trees.loc[node + child] = {'parent': node, 'children': [], 
                                         'wins': 0, 'visits': 0}
                self.trees.loc[node]['children'].append(node + child)
                
                return (node + child)
            else:
                node = self.best_uct(node)
        # in case no children are present / node is terminal 
        return self.pick_unvisited(node) or node

    def non_terminal(self, node):
        return not node.endswith('c') # c stands for challenge

    # function for the result of the simulation
    def rollout(self, node):
        while self.non_terminal(node):
            node = self.rollout_policy(node)
        return node, result(node)

    def pick_random(self, children):
        return random.choice(children)

    # function for randomly selecting a child node
    def rollout_policy(self, node): 
        # TODO: need to check later
        
        # check if node has children
        if self.fully_expanded(node):
            # randomly create a valid child for node
            child = pick_valid_move(node)
            
            # TODO: when to update the visits and values??
            # fill data to tree table
            self.trees.loc[node + child] = {'parent': node, 'children': [], 
                                     'wins': 0, 'visits': 0}
            self.trees.loc[node]['children'].append(node + child)
        
        return self.pick_random(self.trees.loc[node]['children'])

    def is_root(self, node): # TODO: need to check later
        return self.trees.loc[node]['parent'] == '' or \
            self.trees.loc[node]['parent'] is None

    def update_stats(self, node, result): # TODO: need to check later
        self.trees.loc[node]['visits'] += 1
        self.trees.loc[node]['wins'] += result

    # function for backpropagation
    def backpropagate(self, node, result):
        while not self.is_root(node):
            self.update_stats(node, result)
            node = self.trees.loc[node]['parent']
        
        # also update the root
        self.update_stats(node, result)

    # function for selecting the best child
    # node with highest number of visits
    def best_child(self, node):
        # pick child with highest number of visits
        # todo: palce holder value 
        return None


if __name__ == '__main__':
    d1 = Dice(pattern='33451')
    d2 = Dice(pattern='33451')
    p1 = Player(dice = d1, name = "p1")
    p2 = Player(dice = d2, name = "p2")
    game = Game()
    game.add_player(p1)
    game.add_player(p2)

    train_time = 30 * 60 # 30 minutes
    trainer = MCTS_Trainer(game, time.time() + train_time, comp_power = 10) 
    trainer.monte_carlo_tree_search('3345133451')