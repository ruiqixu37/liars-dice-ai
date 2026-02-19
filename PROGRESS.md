# Test Implementation Progress

## TODO 1: Average strategy extraction and verification
- [x] Add `get_average_strategy()` to `mccfr.py`
- [x] Test: probabilities sum to 1.0
- [x] Test: zero counts fallback to uniform
- [x] Test: proportional to counts
- [x] Test: empty trie fallback to uniform
- [x] Test: converges to dominant action after training

## TODO 2: Utility with wild_one edge cases
- [x] Test: wild_one=True, all 1s count as wildcards
- [x] Test: bidding face=1 disables wild_one
- [x] Test: mixed dice with wild_one=True
- [x] Test: prior (X,1) bid disables wild_one for subsequent utility
- [x] Test: face=1 bid doesn't double-count 1s

## TODO 3: Multi-step traversal
- [x] Test: 2-move agent sequence has regret entries at both decision points
- [x] Test: 3-move sequence (agent→opponent→agent) propagates utility back

## TODO 4: next_valid_move boundary cases
- [x] Test: first move has exactly 18 moves (qty 2-4, face 1-6)
- [x] Test: from (7,6) — qty 8 valid, (7,1) wraparound, no qty 9+
- [x] Test: from (8,6) with wild_one — only (8,1) and challenge
- [x] Test: from (8,6) wild_one=False — only challenge
- [x] Test: from (8,1) wild_one=False — only challenge
- [x] Test: quantity never exceeds 8

## TODO 5: Mixed-strategy convergence
- [x] Test: regret differentiation after training with random opponents
- [x] Test: at least one positive regret, strategy not all-zero

## TODO 6: MCCFR_P integration (save/load cycle)
- [x] Test: save creates trie files and time.pkl
- [x] Test: saved trie has non-zero regret entries
- [x] Test: resume accumulates time and iterations

## Summary
- **Before**: 44 tests
- **After**: 64 tests (20 new)
- **New code**: `get_average_strategy()` function in `mccfr.py`
- All 64 tests pass in ~15 seconds
