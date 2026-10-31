import time
import math
import random
import numpy as np

def _score_of(U):
    """Calculate the score for a set U."""
    if not U or 0 not in U:
        return 0.0
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    
    max_u = U[-1]
    
    # Calculate sumset and difference set sizes
    sumset = set()
    diffset = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(n):
            sumset.add(ui + U[j])
            diffset.add(ui - U[j])
            
    size_sum = len(sumset)
    size_diff = len(diffset)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    denom_arg = 2 * max_u + 1
    if denom_arg <= 1:
        return 0.0
        
    denom = math.log(denom_arg)
    if denom == 0:
        return 0.0
        
    return 1 + math.log(ratio) / denom

def _score_of_fast(U):
    """
    Optimized score calculation using sorted properties.
    Note: For the sizes of sets we deal with (n < 100), the overhead of 
    complex optimizations might not beat the simple set implementation, 
    but we can use a slightly faster loop structure.
    """
    if not U or 0 not in U:
        return 0.0
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    
    max_u = U[-1]
    
    sumset = set()
    diffset = set()
    
    # Local variable binding for speed
    add_s = sumset.add
    add_d = diffset.add
    
    for i in range(n):
        ui = U[i]
        for j in range(n):
            add_s(ui + U[j])
            add_d(ui - U[j])
            
    size_sum = len(sumset)
    size_diff = len(diffset)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    denom_arg = 2 * max_u + 1
    if denom_arg <= 1:
        return 0.0
        
    denom = math.log(denom_arg)
    if denom == 0:
        return 0.0
        
    return 1 + math.log(ratio) / denom

def _local_search(U, max_steps=2000, time_limit=10.0, allow_change_size=True):
    """
    Perform local search. 
    Moves:
    1. Shift an element.
    2. Add a new element.
    3. Remove an element.
    4. 'Two-block' perturbation: try to split the set into two arithmetic-like groups.
    """
    best_U = sorted(set(U))
    best_score = _score_of_fast(best_U)
    
    start_time = time.time()
    n = len(best_U)
    
    # Pre-calculate some parameters
    # We want to encourage multiplicatively independent structures.
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
            
        # Choose a move type
        r = random.random()
        
        if r < 0.6:
            # Shift a random element
            if n <= 1:
                continue
            i = random.randint(1, n - 1)
            
            # Determine range for new value
            low = best_U[i-1] + 1
            # Allow expansion if it's the last element
            if i == n - 1:
                high = best_U[i] + random.randint(1, 20)
            else:
                high = best_U[i+1] - 1
                if low > high:
                    # Try to swap with neighbor or skip
                    continue
            
            new_val = random.randint(low, high)
            
            new_U = best_U.copy()
            new_U[i] = new_val
            new_U.sort()
            
            # Check uniqueness
            if len(new_U) != n:
                continue
                
        elif r < 0.8 and allow_change_size:
            # Add a new element
            if n > 30: # Cap size to keep search space manageable
                continue
            max_val = best_U[-1]
            # Try adding near the end or in gaps
            # Heuristic: add a value that is a multiple of a prime or something?
            # Just random for now, biased towards larger values to increase denominator?
            # Actually, increasing max_u increases denominator. 
            # We want to increase ratio (diff/sum). 
            # Adding elements usually increases both.
            # Let's try adding a value that is "far" from existing ones to maximize diffset spread.
            # Or add a small value to increase density?
            
            # Strategy: Add a value that creates new differences.
            # Pick a random position in the range [1, max_val * 2]
            new_val = random.randint(1, max_val * 2)
            
            if new_val in best_U:
                continue
                
            new_U = best_U + [new_val]
            new_U.sort()
            
        elif r < 0.9 and allow_change_size:
            # Remove an element
            if n <= 2:
                continue
            # Don't remove 0
            i = random.randint(1, n - 1)
            new_U = best_U[:i] + best_U[i+1:]
            
        else:
            # Two-block perturbation
            # Try to modify the set to look like A union B where A is small and dense, B is large and sparse
            # This is a heuristic jump.
            if n < 4:
                continue
                
            # Pick a pivot
            pivot_idx = random.randint(1, n-1)
            # Try to shift the upper part by a large amount
            shift = random.randint(10, 100)
            new_U = best_U.copy()
            for k in range(pivot_idx, n):
                new_U[k] += shift
            new_U.sort()
            
            if len(new_U) != n:
                continue

        new_score = _score_of_fast(new_U)
        
        if new_score > best_score:
            best_score = new_score
            best_U = new_U
            n = len(best_U)
        else:
            # Simulated annealing style acceptance to escape local optima
            # Accept with probability p if worse
            # p = exp((new_score - best_score) / T)
            # T should decay or be small.
            # Since scores are close, let's use a fixed small probability for small degradations
            if new_score > best_score - 0.001:
                if random.random() < 0.05:
                    best_U = new_U
                    best_score = new_score
                    n = len(best_U)

    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit to be safe (150s kill)
    
    best_global_U = [0, 1, 3]
    best_global_score = _score_of_fast(best_global_U)
    
    # Known optimal Golomb rulers and other good candidates
    # The "two-block" idea suggests sets like {0, 1, ..., k} union {M, M+1, ..., M+k}
    # But Golomb rulers are better for difference sets.
    
    golomb_rulers = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 25],
        [0, 1, 4, 10, 12, 17, 25, 34],
        [0, 1, 4, 10, 12, 17, 25, 34, 42],
        [0, 1, 4, 10, 12, 17, 25, 34, 42, 52],
        [0, 1, 4, 10, 12, 17, 25, 34, 42, 52, 60],
        [0, 1, 4, 10, 12, 17, 25, 34, 42, 52, 60, 72],
    ]
    
    # Generate some "two-block" candidates explicitly
    # A = {0, 1, 2}, B = {M, M+1, M+2}
    for M in [10, 20, 50, 100]:
        U_candidate = [0, 1, 2, M, M+1, M+2]
        s = _score_of_fast(U_candidate)
        if s > best_global_score:
            best_global_score = s
            best_global_U = U_candidate

    # Evaluate initial candidates
    for U in golomb_rulers:
        if time.time() > deadline:
            break
        s = _score_of_fast(U)
        if s > best_global_score:
            best_global_score = s
            best_global_U = U
    
    # Local search on each Golomb ruler
    # We will prioritize searching the best few
    candidates_for_search = []
    for U in golomb_rulers:
        s = _score_of_fast(U)
        candidates_for_search.append((s, U))
    
    # Sort by score descending
    candidates_for_search.sort(key=lambda x: x[0], reverse=True)
    
    # Search the top 5 candidates
    for score, U in candidates_for_search[:5]:
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        t_limit = min(10.0, remaining / 3) # Divide remaining time among searches
        
        U_opt, s_opt = _local_search(U, max_steps=1000, time_limit=t_limit)
        
        if s_opt > best_global_score:
            best_global_score = s_opt
            best_global_U = U_opt

    # Try some random restarts with different structures
    # 1. Random Golomb-like
    # 2. Two-block structures
    
    for restart in range(10):
        if time.time() > deadline:
            break
            
        remaining = deadline - time.time()
        if remaining < 10:
            break
            
        # Random initial set
        n = random.randint(4, 12)
        U_rand = [0]
        curr = 0
        for _ in range(n - 1):
            curr += random.randint(1, 15)
            U_rand.append(curr)
            
        # Or try a two-block structure
        if random.random() < 0.5:
            n1 = random.randint(2, 5)
            n2 = random.randint(2, 5)
            M = random.randint(20, 100)
            A = list(range(n1))
            B = [M + i for i in range(n2)]
            U_rand = A + B
            U_rand.sort()
            
        t_limit = min(5.0, remaining / 4)
        U_opt, s_opt = _local_search(U_rand, max_steps=500, time_limit=t_limit)
        
        if s_opt > best_global_score:
            best_global_score = s_opt
            best_global_U = U_opt
            
    return best_global_U
