"""Covering design C(v,k,t) via greedy set-cover with efficient bitset tracking."""
import itertools
import random
import sys
import numpy as np

def construct(v, k, t):
    if v <= 25 and k <= 10:
        return _construct_optimized(v, k, t)
    else:
        return _basic_greedy(v, k, t)


def _basic_greedy(v, k, t):
    """Original greedy: pick uncovered t-subset, extend to k-subset maximizing coverage."""
    all_tsubsets = set(itertools.combinations(range(v), t))
    uncovered = set(all_tsubsets)
    blocks = []
    
    while uncovered:
        # Pick the first uncovered t-subset as seed
        seed_subset = next(iter(uncovered))
        block = list(seed_subset)
        
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                for combo in itertools.combinations(block, t - 1):
                    new_tsub = tuple(sorted(combo + (x,)))
                    if new_tsub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        # Remove all t-subsets of this block from uncovered
        for tsub in itertools.combinations(block, t):
            uncovered.discard(tsub)
    
    return blocks


def _construct_optimized(v, k, t):
    """
    Optimized construction:
    1. NumPy-based Greedy Set Cover for initial solution.
    2. Local Search: Redundancy removal.
    """
    rng = random.Random(42)
    
    all_ksubsets = list(itertools.combinations(range(v), k))
    tsubsets = list(itertools.combinations(range(v), t))
    n_tsubs = len(tsubsets)
    n_ksubs = len(all_ksubsets)
    
    tsub_to_idx = {ts: i for i, ts in enumerate(tsubsets)}
    
    incidence = np.zeros((n_ksubs, n_tsubs), dtype=np.uint8)
    for i, ksub in enumerate(all_ksubsets):
        for tsub in itertools.combinations(ksub, t):
            j = tsub_to_idx[tsub]
            incidence[i, j] = 1
    
    # 1. Greedy Set Cover
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    blocks_indices = []
    
    while uncovered.any():
        covered_counts = incidence @ uncovered
        best_idx = int(np.argmax(covered_counts))
        if covered_counts[best_idx] == 0:
            break
        blocks_indices.append(best_idx)
        uncovered[incidence[best_idx] == 1] = 0
    
    # 2. Local Search: Redundancy Removal
    tsub_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in blocks_indices:
        tsub_counts[incidence[idx] == 1] += 1
        
    improved = True
    while improved:
        improved = False
        for i in range(len(blocks_indices) - 1, -1, -1):
            idx = blocks_indices[i]
            covered_by_idx = incidence[idx] == 1
            if np.all(tsub_counts[covered_by_idx] > 1):
                tsub_counts[covered_by_idx] -= 1
                blocks_indices.pop(i)
                improved = True
                
    return [list(all_ksubsets[i]) for i in blocks_indices]



