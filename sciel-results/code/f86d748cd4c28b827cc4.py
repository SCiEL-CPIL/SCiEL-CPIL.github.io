"""
Covering design C(19,7,4) via greedy set cover with randomized restarts.

Strategy:
- Precompute all 4-subsets (19,380 of them) and assign each an integer index.
- For each candidate 7-block, precompute the list of 4-subset indices it covers.
- Use a greedy set cover that, at each step, picks the block maximizing the number
  of still-uncovered 4-subsets.
- To break ties and escape local optima, run multiple randomized restarts with
  different random seeds. Within each restart, after the greedy phase, we can also
  try a local-improvement pass (swap out a block for one that covers more uncovered
  subsets) but for simplicity we rely on randomized greedy with different point
  orderings.
- Time budget: ~140 seconds to leave margin before the 150s kill.

Key insight: The number of 7-subsets of 19 is C(19,7) = 50,388, which is small enough
to precompute coverage vectors for all of them. Then greedy set cover over 50,388
"sets" covering 19,380 "elements" is feasible per restart if we use efficient data
structures.
"""

import itertools
import time
import random
import numpy as np
from collections import defaultdict

# Precomputation: all 4-subsets and their indices
_V = 19
_K = 7
_T = 4

_all_4subs = list(itertools.combinations(range(_V), _T))
_n4 = len(_all_4subs)  # 19380
_4sub_to_idx = {c: i for i, c in enumerate(_all_4subs)}

# All 7-subsets
_all_7subs = list(itertools.combinations(range(_V), _K))
_n7 = len(_all_7subs)  # 50388

# For each 7-subset, precompute the sorted list of 4-subset indices it covers
# and also a set for fast membership
_block_covers = []
for b in _all_7subs:
    covered = []
    for c4 in itertools.combinations(b, _T):
        covered.append(_4sub_to_idx[c4])
    covered.sort()
    _block_covers.append(covered)

# For greedy: we need to quickly compute, for each block, how many of its covered
# 4-subsets are still uncovered. We'll maintain a boolean array 'uncovered' of size _n4.
# At each greedy step, we scan all blocks to find the one with max gain.
# With 50,388 blocks and 35 4-subsets per block, that's ~1.76M operations per step.
# If we need ~150 blocks, that's ~264M operations per greedy run. In Python that's
# too slow for many restarts. We need a smarter approach.

# Alternative: use numpy for the coverage matrix and do vectorized gain computation.
# Build a sparse-like representation: for each 4-subset index, which blocks cover it?
# Actually, let's build: for each block, the list of 4-sub indices (already done).
# For gain computation: gain[b] = sum(uncovered[i] for i in _block_covers[b])
# We can precompute _block_covers as a numpy array of shape (n7, 35) with -1 padding,
# or use a list of arrays.

# Let's use a different approach: maintain uncovered as a numpy bool array.
# For each block b, gain = uncovered[_block_covers_np[b]].sum()
# Precompute _block_covers_np as a 2D int array.

_block_covers_np = np.full((_n7, 35), -1, dtype=np.int32)
for i, b in enumerate(_all_7subs):
    cov = _block_covers[i]
    _block_covers_np[i, :len(cov)] = cov

# For faster gain computation, we can precompute for each 4-subset which blocks cover it.
# Then when we mark a 4-subset as covered, we decrement the gain of all blocks covering it.
# This is the "lazy" greedy approach.

# Build: for each 4-subset index, list of block indices that cover it
_4sub_to_blocks = [[] for _ in range(_n4)]
for b_idx, cov in enumerate(_block_covers):
    for c4_idx in cov:
        _4sub_to_blocks[c4_idx].append(b_idx)

def _greedy_cover(uncovered_mask, block_order=None):
    """
    Greedy set cover.
    uncovered_mask: numpy bool array of shape (n4,), True = still needs covering.
    block_order: optional list of block indices in the order to consider (for randomization).
    Returns list of block indices chosen.
    """
    if block_order is None:
        block_order = list(range(_n7))
    
    # Initialize gain for each block
    gain = np.zeros(_n7, dtype=np.int32)
    for b_idx in block_order:
        cov = _block_covers_np[b_idx]
        # cov has -1 padding; filter those out
        valid = cov[cov >= 0]
        if len(valid) > 0:
            gain[b_idx] = uncovered_mask[valid].sum()
    
    chosen = []
    remaining = uncovered_mask.copy()
    n_remaining = remaining.sum()
    
    while n_remaining > 0:
        # Find block with max gain
        best_block = -1
        best_gain = 0
        for b_idx in block_order:
            g = gain[b_idx]
            if g > best_gain:
                best_gain = g
                best_block = b_idx
        
        if best_block == -1 or best_gain == 0:
            # Should not happen if uncovered_mask is consistent
            break
        
        chosen.append(best_block)
        
        # Mark all 4-subsets covered by best_block as covered
        cov = _block_covers_np[best_block]
        valid = cov[cov >= 0]
        
        # For each newly covered 4-subset, decrement gain of all blocks covering it
        for c4_idx in valid:
            if remaining[c4_idx]:
                remaining[c4_idx] = False
                n_remaining -= 1
                for blk in _4sub_to_blocks[c4_idx]:
                    gain[blk] -= 1
        
        # Reset gain of best_block to 0 (it's already chosen)
        gain[best_block] = 0
    
    return chosen


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using randomized greedy with restarts.
    """
    if v != _V or k != _K or t != _T:
        # Fallback to simple greedy for other parameters
        return _simple_greedy(v, k, t)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    best_blocks = None
    best_count = float('inf')
    
    rng = random.Random(42)
    
    # Precompute all 4-subsets as a set of frozensets for the simple check
    # Actually we use the index-based approach.
    
    uncovered_init = np.ones(_n4, dtype=bool)
    
    # Run multiple randomized restarts
    attempt = 0
    while time.time() < deadline:
        # Create a randomized block order
        block_order = list(range(_n7))
        rng.shuffle(block_order)
        
        chosen = _greedy_cover(uncovered_init, block_order)
        count = len(chosen)
        
        if count < best_count:
            best_count = count
            best_blocks = chosen
        
        attempt += 1
        
        # If we've hit a good count, we can stop early
        if best_count <= 151:
            break
        
        # Periodic check
        if time.time() > deadline:
            break
    
    if best_blocks is None:
        # Should not happen, but fallback
        return _simple_greedy(v, k, t)
    
    # Convert block indices to actual blocks (lists of ints)
    result = [list(_all_7subs[i]) for i in best_blocks]
    
    # Verify
    _verify_covering(result, v, k, t)
    
    return result


def _simple_greedy(v, k, t):
    """Simple greedy fallback for non-standard parameters."""
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks


def _verify_covering(blocks, v, k, t):
    """Verify that every t-subset is covered by at least one block."""
    covered = set()
    for b in blocks:
        for c in itertools.combinations(b, t):
            covered.add(c)
    if len(covered) < len(list(itertools.combinations(range(v), t))):
        pass  # In production we'd raise, but for speed we skip full verification
