"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START

def _compute_score(U):
    """Compute the score for a set U containing 0."""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    max_u = U[-1]
    if max_u <= 0:
        return 0.0
    
    # Use set operations for better performance with moderate n
    sums = set()
    diffs = set()
    for i in range(n):
        ui = U[i]
        for j in range(n):
            uj = U[j]
            sums.add(ui + uj)
            diffs.add(ui - uj)
    
    n_sums = len(sums)
    n_diffs = len(diffs)
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return 0.0
    score = 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    return score

def _golomb_ruler(n):
    """Generate a Golomb ruler of length n using known optimal/near-optimal values."""
    # Known optimal Golomb rulers for n marks
    # n, marks
    rulers = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 18, 23, 29],
        8: [0, 1, 4, 10, 18, 23, 29, 36],
        9: [0, 1, 4, 10, 18, 23, 29, 36, 44],
        10: [0, 1, 4, 10, 18, 23, 29, 36, 44, 52],
        11: [0, 1, 4, 10, 18, 23, 29, 36, 44, 52, 60],
        12: [0, 1, 4, 10, 18, 23, 29, 36, 44, 52, 60, 68],
        13: [0, 1, 4, 10, 18, 23, 29, 36, 44, 52, 60, 68, 76],
        14: [0, 1, 4, 10, 18, 23, 29, 36, 44, 52, 60, 68, 76, 84],
        15: [0, 1, 4, 10, 18, 23, 29, 36, 44, 52, 60, 68, 76, 84, 92]
    }
    if n in rulers:
        return rulers[n]
    
    # Fallback: generate a simple Golomb ruler
    U = [0]
    current = 0
    for i in range(1, n):
        current += i * 2 + 1
        U.append(current)
    return U

def _local_search_small(U, max_iterations=20000, time_limit=5.0):
    """Aggressive local search for small sets to minimize diameter and maximize score."""
    U = sorted(set(U))
    best_score = _compute_score(U)
    best_U = list(U)
    start_time = time.time()
    
    # We will try to perturb elements to reduce max element while keeping ratio high
    # Or just maximize score directly
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        n = len(best_U)
        if n < 2:
            break
            
        # Strategy: Try to reduce the largest element if possible, or optimize others
        # We focus on small n, so we can try all single-element moves
        
        # Pick a random index to perturb
        idx = random.randint(1, n - 1) # Don't move 0
        old_val = best_U[idx]
        
        # Determine search range for new value
        # We want to keep the set compact, so search near current value
        # Or search specifically to reduce max element
        if idx == n - 1:
            # Try to reduce the max element
            search_start = max(1, old_val - 20)
            search_end = old_val - 1
        else:
            search_start = max(1, old_val - 10)
            search_end = old_val + 10
            
        # If we are stuck, try a larger jump
        if random.random() < 0.1:
            search_start = 1
            search_end = max(50, best_U[-1] + 10)
            
        # Sample a few new values
        for _ in range(5):
            new_val = random.randint(search_start, search_end)
            if new_val in best_U:
                continue
                
            candidate = list(best_U)
            candidate[idx] = new_val
            candidate = sorted(set(candidate))
            
            if 0 not in candidate or len(candidate) != n:
                continue
                
            score = _compute_score(candidate)
            if score > best_score:
                best_score = score
                best_U = candidate
                
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115 second limit for safety
    
    best_U = [0, 1, 3]
    best_score = _compute_score(best_U)
    
    # 1. Evaluate known Golomb rulers
    for n in range(3, 16):
        if time.time() > deadline - 10:
            break
        seed = _golomb_ruler(n)
        score = _compute_score(seed)
        if score > best_score:
            best_score = score
            best_U = list(seed)
            
    # 2. Refine the best seed with local search
    # Since small sets are the winners, we run local search on the best few seeds
    # We will run local search on the best 5 seeds found so far
    
    # Re-evaluate seeds to keep top 5
    seeds_to_refine = []
    for n in range(3, 16):
        seed = _golomb_ruler(n)
        score = _compute_score(seed)
        seeds_to_refine.append((score, seed))
        
    seeds_to_refine.sort(key=lambda x: x[0], reverse=True)
    
    for score, seed in seeds_to_refine[:5]:
        if time.time() > deadline - 20:
            break
        U, score = _local_search_small(seed, max_iterations=50000, time_limit=15)
        if score > best_score:
            best_score = score
            best_U = U
            
    # 3. Try to find even better sets via random restarts with local search
    # Focus on small sizes 4, 5, 6
    for n_target in [4, 5, 6, 7]:
        if time.time() > deadline - 10:
            break
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            # Generate a random "near-Golomb" set
            # Start with a Golomb ruler and perturb it
            base = _golomb_ruler(n_target)
            U = list(base)
            # Perturb
            for i in range(1, len(U)):
                if random.random() < 0.5:
                    U[i] += random.randint(-5, 5)
            U = sorted(set(U))
            if 0 not in U:
                U.append(0)
            U = sorted(U)
            
            U, score = _local_search_small(U, max_iterations=20000, time_limit=2)
            if score > best_score:
                best_score = score
                best_U = U
                
    # Ensure 0 is in the set
    if 0 not in best_U:
        best_U.append(0)
        
    return sorted(best_U)

# EVOLVE-BLOCK-END
