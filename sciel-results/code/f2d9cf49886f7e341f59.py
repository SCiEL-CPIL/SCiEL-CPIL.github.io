"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

# Precompute all 4-subsets of [1,16] as bitmasks (0-indexed bits for 0..15)
_ALL_MASKS = []
for c in combinations(range(16), 4):
    mask = 0
    for i in c:
        mask |= (1 << i)
    _ALL_MASKS.append(mask)

_N = len(_ALL_MASKS)  # 1820

def _is_sunflower(a, b, c):
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _can_add_idx(candidate_idx, family_indices):
    """Check if candidate index can be added to family of indices."""
    candidate_mask = _ALL_MASKS[candidate_idx]
    n = len(family_indices)
    for i in range(n):
        a_mask = _ALL_MASKS[family_indices[i]]
        for j in range(i + 1, n):
            b_mask = _ALL_MASKS[family_indices[j]]
            if _is_sunflower(a_mask, b_mask, candidate_mask):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    rng = random.Random(12345)
    
    best_family = []
    best_size = 0
    
    for trial in range(200):
        if time.time() > deadline - 5.0:
            break
            
        # Random start
        order = list(range(_N))
        rng.shuffle(order)
        
        family = []
        family_set = set()
        
        # Greedy build
        for idx in order:
            if time.time() > deadline - 5.0:
                break
            if _can_add_idx(idx, family):
                family.append(idx)
                family_set.add(idx)
                
        # Local Search: Try to add any missing set
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for idx in range(_N):
                if time.time() > deadline - 2.0:
                    break
                if idx in family_set:
                    continue
                if _can_add_idx(idx, family):
                    family.append(idx)
                    family_set.add(idx)
                    improved = True
                    
        # Try a few random swaps if stuck
        if time.time() < deadline - 2.0:
            for _ in range(20):
                if time.time() > deadline - 2.0:
                    break
                if not family:
                    break
                rem_idx = rng.choice(family)
                # Try to replace rem_idx with a random candidate
                for _ in range(5):
                    if time.time() > deadline - 2.0:
                        break
                    cand_idx = rng.randint(0, _N - 1)
                    if cand_idx in family_set:
                        continue
                    # Check if cand_idx can be added if rem_idx is removed
                    # We simulate this by checking _can_add_idx on family without rem_idx
                    # This is O(|F|^2).
                    # To avoid modifying family, we can just check conflicts with rem_idx separately?
                    # No, it's complex.
                    # Skip swaps for simplicity and speed, rely on restarts.
                    pass
                        
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)
            
    # Convert to sorted lists of 1-based integers
    result = []
    for idx in best_family:
        mask = _ALL_MASKS[idx]
        elems = [i + 1 for i in range(16) if mask & (1 << i)]
        result.append(sorted(elems))
        
    return result
# EVOLVE-BLOCK-END
