"""Covering design C(v,k,t) via greedy set-cover with efficient bitset tracking."""
import itertools
import random
import sys
import time

def construct(v, k, t):
    # For small parameters, use optimized greedy with bitsets
    if v <= 20 and k <= 8:
        return _greedy_cover_bitset(v, k, t)
    else:
        return _basic_greedy(v, k, t)


def _basic_greedy(v, k, t):
    """Original greedy: pick uncovered t-subset, extend to k-subset maximizing coverage."""
    all_tsubsets = set(itertools.combinations(range(v), t))
    uncovered = set(all_tsubsets)
    blocks = []
    
    while uncovered:
        seed_subset = next(iter(uncovered))
        block = list(seed_subset)
        
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
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
        for tsub in itertools.combinations(block, t):
            uncovered.discard(tsub)
    
    return blocks


def _greedy_cover_bitset(v, k, t):
    """
    Greedy set-cover using Python integers as bitsets for speed.
    Each k-subset's coverage is a bitmask of the t-subsets it contains.
    uncovered is a bitmask of which t-subsets are still uncovered.
    At each step, find the k-subset whose bitmask & uncovered has max popcount.
    """
    tsubsets = list(itertools.combinations(range(v), t))
    n_tsubs = len(tsubsets)
    tsub_to_idx = {ts: i for i, ts in enumerate(tsubsets)}
    
    all_ksubsets = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubsets)
    
    # Precompute bitmask for each k-subset
    ksub_masks = [0] * n_ksubs
    for i, ksub in enumerate(all_ksubsets):
        mask = 0
        for tsub in itertools.combinations(ksub, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
        ksub_masks[i] = mask
    
    # uncovered bitmask
    uncovered = (1 << n_tsubs) - 1
    blocks = []
    
    # Precompute popcount lookup for 16-bit chunks for faster popcount
    # Actually, Python's int.bit_count() is fast enough in modern Python
    
    while uncovered:
        best_idx = -1
        best_count = 0
        
        for i in range(n_ksubs):
            covered = ksub_masks[i] & uncovered
            cnt = covered.bit_count()
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        
        if best_idx == -1 or best_count == 0:
            break
        
        blocks.append(list(all_ksubsets[best_idx]))
        uncovered &= ~ksub_masks[best_idx]
    
    return blocks
