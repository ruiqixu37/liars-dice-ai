# liars-dice-ai

A Liar's Dice AI using [Monte Carlo Counterfactual Regret Minimization (MCCFR)](https://proceedings.neurips.cc/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf), inspired by [Pluribus](https://www.science.org/doi/10.1126/science.aay2400).

## Game Rules

Two players each roll 5 hidden dice. Players take turns bidding on the total count of a face value across *all* dice. A bid is a pair *(quantity, face)* — e.g. "3 fours" claims there are at least 3 dice showing 4 among both players combined. Each bid must be strictly higher than the previous one. Instead of bidding, a player may **challenge** the last bid.

When challenged, all dice are revealed:
- If the actual count meets or exceeds the bid, the **challenger loses**.
- If the actual count is less than the bid, the **bidder loses**.

**Wild ones**: Dice showing 1 count as wildcards for any face — unless someone has bid on face 1, which disables the wildcard rule for the rest of the round.

## Architecture

```
state.py     — State representation and game logic (valid moves, terminal check, utility)
mccfr.py     — MCCFR training loop (traverse, pruning, LCFR discounting, strategy updates)
trie.py      — Flat-dict trie for storing regret, strategy probabilities, and action counters
game.py      — CLI for playing against the trained AI
```

**State encoding**: `State([dice, (qty, face), ...])` where `dice` is a 5-digit integer (e.g. `23415`) and the remaining elements are the bid history. A `first_act` flag tracks who goes first.

**Training**: MCCFR iteratively plays the game against itself using external sampling. Each iteration picks a random dice roll, traverses the game tree, and updates regret values. Over many iterations the average strategy converges to an approximate Nash equilibrium. Key phases:
- **LCFR discounting** (early training) — discounts old regrets and action counters so early noise fades.
- **Pruning** (after warmup) — skips branches with very negative regret to speed up traversal.
- Strategies are saved per-dice as serialized Trie files (`output/trie_{dice}.pkl`).

## Setup

```bash
conda create -n liarsdice python=3.11
conda activate liarsdice
pip install numpy pytest pyyaml
```

## Training

```bash
# Train using config/mccfr.yaml (default iterations is set in config)
python mccfr.py

# Optionally point to a custom config file
python mccfr.py --config path/to/mccfr.yaml

# Optionally override config values from CLI
python mccfr.py --iterations 10000 --prune-threshold 5400 --save-interval 50
```

Training prints progress every 100 iterations — iteration speed, ETA, unique dice seen, trie size, and phase transitions (pruning activation, LCFR end). Models are saved to `output/` every `save_interval` iterations (default `5000`) and training can be resumed by running the command again.

## Playing

```bash
# Play against the AI
python game.py

# Play with hint mode (shows suggested moves from the trained strategy)
python game.py --hints
```

The CLI shows:
- Your dice and bid history each turn
- Wild ones status
- Valid move range on your turn
- AI strategy visualization (probability distribution over its top actions)
- End-of-game summary with both dice, matching count, and result explanation

With `--hints`, the AI looks up the trained strategy for your dice and suggests the top moves. If no strategy has been trained for your exact dice roll, it prints a notice.

## Tests

```bash
pytest tests/
```
