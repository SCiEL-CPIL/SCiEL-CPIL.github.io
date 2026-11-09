"""Covering design C(v,k,t) via vectorized frequency-greedy + iterative reduction, multi-start."""
import itertools
import random
import time
from math import comb
import numpy as np


def _tsub_index_map(v, t):
    """Map each t-subset (as a tuple) to a dense integer index."""
    combos = list(itertools.combinations(range(v), t))
    idx_map = {}
    for i, c in enumerate(combos):
        idx_map[c] = i
    return combos, idx_map, len(combos)


def _block_tsub_indices(block, t, idx_map):
    """Return list of indices of t-subsets contained in block."""
    out = []
    for c in itertools.combinations(block, t):
        out.append(idx_map[c])
    return out


def _greedy_construct(v, k, t, combos, idx_map, n_tsubs, rng, time_limit):
    """
    Single greedy construction pass using frequency-based selection.
    Returns list of blocks (each a sorted list of k points).
    """
    start_time = time.time()
    uncovered = np.ones(n_tsubs, dtype=np.int32)
    total_uncovered = n_tsubs
    blocks = []
    
    # Precompute: for each point x and each (t-1)-subset s, the index of s ∪ {x}
    # This is expensive to precompute fully, so we compute on the fly but cache.
    # Actually, let's precompute a mapping: for each (t-1)-tuple, and each point not in it,
    # the index of the t-tuple.
    
    # For speed, precompute all (t-1)-subsets and for each, which points can extend it
    t_minus_1 = t - 1
    tsub1_list = list(itertools.combinations(range(v), t_minus_1))
    
    # For each (t-1)-subset, precompute the list of (point, tsub_index) pairs
    # where point is not in the (t-1)-subset
    ext_map = {}
    for s in tsub1_list:
        s_set = set(s)
        pairs = []
        for x in range(v):
            if x not in s_set:
                tsub = tuple(sorted(s + (x,)))
                pairs.append((x, idx_map[tsub]))
        ext_map[s] = pairs
    
    while total_uncovered > 0:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random uncovered t-subset to start the block
        # Find an uncovered index
        uncov_indices = np.where(uncovered > 0)[0]
        if len(uncov_indices) == 0:
            break
        seed_idx = uncov_indices[rng.randint(0, len(uncov_indices) - 1)]
        seed_tsub = combos[seed_idx]
        block = list(seed_tsub)
        block_set = set(block)
        
        while len(block) < k:
            if time.time() - start_time > time_limit:
                break
            
            best_x = -1
            best_gain = -1
            candidates = [x for x in range(v) if x not in block_set]
            
            # Shuffle candidates for tie-breaking diversity
            rng.shuffle(candidates)
            
            for x in candidates:
                gain = 0
                for rest in itertools.combinations(block, t_minus_1):
                    tsub = tuple(sorted(rest + (x,)))
                    tsub_idx = idx_map[tsub]
                    if uncovered[tsub_idx] > 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1 or best_gain == 0:
                # No point helps, pick random
                best_x = rng.choice(candidates)
            
            block.append(best_x)
            block_set.add(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Remove all t-subsets covered by this block
        for tsub_idx in _block_tsub_indices(block, t, idx_map):
            if uncovered[tsub_idx] > 0:
                uncovered[tsub_idx] = 0
                total_uncovered -= 1
    
    return blocks


def _try_reduce(blocks, v, k, t, combos, idx_map, n_tsubs, time_limit):
    """
    Iteratively remove redundant blocks.
    A block is redundant if all its t-subsets are covered by other blocks.
    """
    start_time = time.time()
    if not blocks:
        return blocks
    
    # Build coverage count: for each t-subset, how many blocks cover it
    # Use a dict: tsub_index -> count
    coverage = np.zeros(n_tsubs, dtype=np.int32)
    block_tsubs = []
    
    for block in blocks:
        tsub_indices = _block_tsub_indices(block, t, idx_map)
        block_tsubs.append(tsub_indices)
        for idx in tsub_indices:
            coverage[idx] += 1
    
    changed = True
    while changed:
        if time.time() - start_time > time_limit:
            break
        changed = False
        new_blocks = []
        new_block_tsubs = []
        for i, block in enumerate(blocks):
            # Check if this block is redundant
            redundant = True
            for idx in block_tsubs[i]:
                if coverage[idx] == 1:
                    # Only this block covers this t-subset
                    redundant = False
                    break
            if redundant:
                # Remove this block
                for idx in block_tsubs[i]:
                    coverage[idx] -= 1
                changed = True
            else:
                new_blocks.append(block)
                new_block_tsubs.append(block_tsubs[i])
        blocks = new_blocks
        block_tsubs = new_block_tsubs
    
    return blocks


def construct(v, k, t):
    """
    Multi-start randomized greedy covering design constructor with reduction.
    Returns list of blocks; each block is a sorted list of k distinct ints in 0..v-1.
    """
    total_deadline = time.time() + 140  # safety margin before 150s kill
    
    # Precompute t-subset indexing
    combos, idx_map, n_tsubs = _tsub_index_map(v, t)
    
    best_blocks = None
    best_len = float('inf')
    
    restart_num = 0
    per_restart_budget = 3.0  # seconds per restart attempt
    
    while time.time() < total_deadline:
        restart_num += 1
        rng = random.Random(restart_num * 12345 + 67890)
        
        # Greedy construction
        remaining_time = min(per_restart_budget, total_deadline - time.time())
        if remaining_time < 0.5:
            break
        
        blocks = _greedy_construct(v, k, t, combos, idx_map, n_tsubs, rng, remaining_time)
        
        # Try to reduce
        reduce_time = min(1.0, total_deadline - time.time())
        if reduce_time > 0.1:
            blocks = _try_reduce(blocks, v, k, t, combos, idx_map, n_tsubs, reduce_time)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        # If we have time, try more aggressive reduction on the best
        if time.time() < total_deadline - 5:
            # Additional reduction pass with more time
            extra_time = min(5.0, total_deadline - time.time() - 2)
            if extra_time > 0.5:
                reduced = _try_reduce(list(best_blocks), v, k, t, combos, idx_map, n_tsubs, extra_time)
                if len(reduced) < best_len:
                    best_len = len(reduced)
                    best_blocks = reduced
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        rng = random.Random(0)
        best_blocks = _greedy_construct(v, k, t, combos, idx_map, n_tsubs, rng, 30)
    
    return best_blocks
