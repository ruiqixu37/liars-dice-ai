from state import State
import argparse
import os
import pickle
import random

import numpy as np


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


def format_dice(dice_int):
    """Format a 5-digit dice integer as a readable list, e.g. [1, 2, 3, 4, 5]."""
    digits = []
    d = dice_int
    while d > 0:
        digits.append(d % 10)
        d //= 10
    digits.reverse()
    return digits


def format_action(action):
    """Format an action tuple as a readable string."""
    if action == (-1, -1):
        return "challenge"
    return f"bid {action[0]} {action[1]}"


def count_matching_dice(dice1, dice2, face, wild_one):
    """Count total dice matching a face across both players' dice."""
    total = 0
    for dice_int in (dice1, dice2):
        d = dice_int
        while d > 0:
            digit = d % 10
            d //= 10
            if digit == face:
                total += 1
            elif wild_one and digit == 1 and face != 1:
                total += 1
    return total


def display_strategy(strategy, label="AI Strategy"):
    """Display a strategy probability distribution as an ASCII bar chart."""
    # Filter actions with probability > 0.01 and sort descending
    items = [(a, p) for a, p in strategy.items() if p > 0.01]
    items.sort(key=lambda x: x[1], reverse=True)

    if not items:
        return

    max_label_len = max(len(format_action(a)) for a, _ in items)
    max_bar_width = 20

    print(f"\n  {label}:")
    for action, prob in items:
        bar_len = int(prob * max_bar_width)
        bar = "\u2588" * bar_len
        label_str = format_action(action).ljust(max_label_len)
        print(f"    {label_str}  {bar}  {prob:.2f}")
    print()


def load_trie(dice_int):
    """Load a trained trie for a given dice value, or return None."""
    path = f"output/trie_{dice_int}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def parse_player_action(text):
    """Parse player input into an action tuple. Returns None on parse error."""
    text = text.strip().lower()
    if text == "challenge":
        return (-1, -1)
    if text.startswith("bid "):
        parts = text.split()
        if len(parts) == 3:
            try:
                return (int(parts[1]), int(parts[2]))
            except ValueError:
                return None
    return None


class Game:
    """Engine for the Liar's Dice CLI game."""

    def __init__(self, hints=False):
        self.hints = hints
        self.state = None       # agent's state (dice = agent's dice)
        self.player_dice = None
        self.agent_trie = None

    def start_game(self):
        """Initialize agent state, player dice, and load agent strategy."""
        self.state = init_state(State([]))
        self.player_dice = init_player_dice()

        # Load agent strategy
        self.agent_trie = load_trie(self.state.dice)
        if self.agent_trie is None:
            print("Warning: No trained strategy found for agent's dice. Agent will play randomly.")
        else:
            print(f"Agent strategy loaded (dice {self.state.dice}).")

        print(f"\nYour dice: {format_dice(self.player_dice)}")
        print(f"Wild ones: {'active' if self.state.wild_one else 'inactive'}")
        print()

    def _agent_action(self):
        """Select an action for the agent using trained strategy or uniform random."""
        from mccfr import get_average_strategy

        valid_moves = self.state.next_valid_move()

        if self.agent_trie is not None:
            strategy = get_average_strategy(self.state, self.agent_trie)
            display_strategy(strategy)
            actions = list(strategy.keys())
            probs = np.array([strategy[a] for a in actions])
            # Renormalize in case of floating point drift
            probs /= probs.sum()
            action = actions[np.random.choice(len(actions), p=probs)]
        else:
            action = valid_moves[np.random.choice(len(valid_moves))]

        return action

    def _show_hints(self):
        """Show hint strategy for the player's current position."""
        from mccfr import get_average_strategy

        player_trie = load_trie(self.player_dice)
        if player_trie is None:
            print("  Hint unavailable -- no trained strategy for this dice combination")
            return

        # Build a mirrored state: player's dice, flipped first_act, same history
        mirrored = State([self.player_dice] + list(self.state.history),
                         first_act=not self.state.first_act)

        strategy = get_average_strategy(mirrored, player_trie)

        # Show top 5 actions
        items = sorted(strategy.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\n  Suggested moves:")
        for action, prob in items:
            print(f"    {format_action(action):15s}  {prob:.2f}")
        print()

    def _show_bid_history(self):
        """Print the current bid history."""
        if not self.state.history:
            print("  No bids yet.")
            return
        print("  Bid history:")
        for i, bid in enumerate(self.state.history):
            # Determine who made this bid
            if self.state.first_act:
                who = "AI" if i % 2 == 0 else "You"
            else:
                who = "You" if i % 2 == 0 else "AI"
            print(f"    {who}: {format_action(bid)}")

    def _show_game_result(self):
        """Display end-of-game result with full details."""
        print("=" * 40)
        print("GAME OVER")
        print("=" * 40)

        agent_dice_list = format_dice(self.state.dice)
        player_dice_list = format_dice(self.player_dice)
        print(f"  AI's dice:   {agent_dice_list}")
        print(f"  Your dice:   {player_dice_list}")

        last_bid = self.state.history[-2]
        last_bid_qty, last_bid_face = last_bid
        print(f"  Last bid:    {format_action(last_bid)}")

        # Determine wild_one status at end of game
        wild = self.state.wild_one
        total = count_matching_dice(self.state.dice, self.player_dice,
                                    last_bid_face, wild)
        wild_note = " (ones are wild)" if wild and last_bid_face != 1 else ""
        print(f"  Matching dice (face {last_bid_face}): {total}{wild_note}")

        # Who challenged?
        challenger_idx = len(self.state.history) - 1
        if self.state.first_act:
            challenger = "AI" if challenger_idx % 2 == 0 else "You"
        else:
            challenger = "You" if challenger_idx % 2 == 0 else "AI"

        bid_succeeded = total >= last_bid_qty
        print(f"  {challenger} challenged. The bid {'was met' if bid_succeeded else 'was NOT met'}.")

        utility = self.state.utility(self.player_dice)
        if utility == 1:
            print("\n  ** Agent wins. You lose! **")
        else:
            print("\n  ** You win! **")
        print()

    def play(self):
        """Main game loop."""
        if self.state.first_act:
            print("AI goes first.\n")
        else:
            print("You go first.\n")

        while not self.state.is_terminal():
            print("-" * 40)
            self._show_bid_history()
            print(f"  Wild ones: {'active' if self.state.wild_one else 'inactive'}")
            print()

            if self.state.player_of_next_move():
                # Agent's turn
                print("  AI is thinking...")
                action = self._agent_action()
                print(f"  AI plays: {format_action(action)}")
                self.state.update_history(action)
            else:
                # Player's turn
                valid_moves = self.state.next_valid_move()

                if self.hints:
                    self._show_hints()

                # Show valid move summary
                bids = [m for m in valid_moves if m != (-1, -1)]
                has_challenge = (-1, -1) in valid_moves
                if bids:
                    min_bid = bids[0]
                    max_bid = bids[-1]
                    print(f"  Valid bids: {format_action(min_bid)} to {format_action(max_bid)}")
                if has_challenge:
                    print("  You can also: challenge")
                print()

                action = None
                while action not in valid_moves:
                    try:
                        raw = input("  Your move (e.g. 'bid 3 4' or 'challenge'): ")
                    except (EOFError, KeyboardInterrupt):
                        print("\nGame aborted.")
                        return
                    action = parse_player_action(raw)
                    if action is None or action not in valid_moves:
                        print("  Invalid move, please try again!")
                        action = None

                self.state.update_history(action)

        self._show_game_result()


def main():
    parser = argparse.ArgumentParser(description="Play Liar's Dice against a trained AI")
    parser.add_argument("--hints", action="store_true",
                        help="Enable hint mode to see suggested moves")
    args = parser.parse_args()

    while True:
        game = Game(hints=args.hints)
        game.start_game()
        game.play()

        try:
            again = input("Play again? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            again = "n"
        if again != "y":
            print("Thanks for playing!")
            break


if __name__ == "__main__":
    main()
