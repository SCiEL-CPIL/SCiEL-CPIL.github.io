import time
import random
import math
import numpy as np

def _score_mask(mask):
    """
    Fast score using Python big integers as bitsets.
    mask: int where bit i is 1 if (i+1) is in the set.
    Returns the number of distinct d > 0 such that a, a+d, a+2d are in the set.
    """
    count = 0
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _mask_to_list(mask):
    """Convert mask to list of integers (1..512)."""
    res = []
    m = mask
    i = 0
    while m:
        if m & 1:
            res.append(i + 1)
        m >>= 1
        i += 1
    return res

def _list_to_mask(S):
    """Convert list of integers (1..512) to mask."""
    mask = 0
    for x in S:
        mask |= (1 << (x - 1))
    return mask

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We aim to finish by 115s.
    deadline = start_time + 115.0
    
    best_score = 0
    best_mask = 0
    
    # Strategy:
    # 1. Quick random initializations.
    # 2. Multi-start Simulated Annealing with a smarter move set (swap instead of remove/add).
    #    The current seed uses remove/add which can break the size constraint if not careful (though it is careful).
    #    Swap is generally more effective for fixed-size set optimization.
    
    num_chains = 20  # More chains, shorter each
    
    # Pre-compute nothing for the fast scorer, it's already O(N) bit ops.
    # The bottleneck is the number of steps we can run.
    
    for chain_idx in range(num_chains):
        # Check time
        if time.time() > deadline:
            break
        
        # Initialize a random set of 32 distinct integers
        S = random.sample(range(1, 513), 32)
        mask = _list_to_mask(S)
        current_score = _score_mask(mask)
        
        if current_score > best_score:
            best_score = current_score
            best_mask = mask
        
        # Simulated Annealing
        # Start with higher temperature, cool down.
        T = 10.0
        T_min = 0.1
        cooling = 0.9995  # Slower cooling to explore more
        
        # Estimate steps per chain:
        # Total time budget for SA is roughly 110s / 20 chains = 5.5s per chain.
        # _score_mask takes ~10-20us. So ~50k steps per chain.
        # We will just loop until time runs out or T is too low.
        
        while time.time() < deadline:
            if T < T_min:
                break
            
            # Pick two distinct indices to swap
            i1 = random.randint(0, 31)
            i2 = random.randint(0, 31)
            while i2 == i1:
                i2 = random.randint(0, 31)
            
            x1 = S[i1]
            x2 = S[i2]
            
            # Swap x1 and x2 in S and mask
            # mask ^= (1 << (x1-1)) | (1 << (x2-1))  -- only if x1 != x2, which is true
            mask ^= (1 << (x1 - 1))
            mask ^= (1 << (x2 - 1))
            
            S[i1] = x2
            S[i2] = x1
            
            new_score = _score_mask(mask)
            delta = new_score - current_score
            
            # Acceptance
            if delta > 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask
            else:
                # Revert
                mask ^= (1 << (x1 - 1))
                mask ^= (1 << (x2 - 1))
                S[i1] = x1
                S[i2] = x2
            
            T *= cooling
    
    # Final check: if we have time left, do a quick local search on the best solution
    if time.time() < deadline - 5.0 and best_mask != 0:
        S_best = _mask_to_list(best_mask)
        mask_best = best_mask
        score_best = best_score
        
        # Greedy local search: try all swaps, take best improvement
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            best_swap_delta = 0
            best_i1 = -1
            best_i2 = -1
            
            # Sample a subset of swaps to keep it fast
            # 32*31/2 = 496 swaps. We can do all of them.
            for i1 in range(32):
                for i2 in range(i1+1, 32):
                    x1 = S_best[i1]
                    x2 = S_best[i2]
                    
                    mask_test = mask_best
                    mask_test ^= (1 << (x1 - 1))
                    mask_test ^= (1 << (x2 - 1))
                    
                    s_test = _score_mask(mask_test)
                    delta = s_test - score_best
                    
                    if delta > best_swap_delta:
                        best_swap_delta = delta
                        best_i1 = i1
                        best_i2 = i2
            
            if best_swap_delta > 0:
                i1 = best_i1
                i2 = best_i2
                x1 = S_best[i1]
                x2 = S_best[i2]
                
                mask_best ^= (1 << (x1 - 1))
                mask_best ^= (1 << (x2 - 1))
                S_best[i1] = x2
                S_best[i2] = x1
                
                score_best += best_swap_delta
                best_score = score_best
                best_mask = mask_best
                improved = True
    
    return _mask_to_list(best_mask)
