import time
import random
import math
from itertools import combinations

def score_of(U):
    if not U:
        return 0.0
    U = list(set(U))
    if 0 not in U:
        U.append(0)
    U.sort()
    max_u = U[-1]
    if max_u == 0:
        return 0.0
    diffs = set()
    sums = set()
    for a in U:
        for b in U:
            diffs.add(a - b)
            sums.add(a + b)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 1.0 # Or slightly above if we want to encourage growth, but mathematically if ratio<1 score<1? 
                   # Actually log(ratio) is negative. 
                   # We want to maximize. 
                   # If ratio < 1, score < 1. 
                   # We should avoid this.
    val = 1 + math.log(ratio) / math.log(2 * max_u + 1)
    return val

def generate_golomb(n):
    """Generate a Golomb ruler of order n using greedy method."""
    A = [0]
    diffs = set()
    for _ in range(n - 1):
        # Find smallest x > A[-1] such that no collision
        x = A[-1] + 1
        while True:
            valid = True
            for a in A:
                d = x - a
                if d in diffs:
                    valid = False
                    break
            if valid:
                A.append(x)
                for a in A[:-1]:
                    diffs.add(x - a)
                break
            x += 1
    return A

def local_search(U, time_limit):
    """Improve the score of U using local search."""
    U = list(U)
    best_score = score_of(U)
    start_time = time.time()
    n = len(U)
    max_u = max(U)
    
    # Randomize order of indices to try
    indices = list(range(n))
    
    while time.time() - start_time < time_limit:
        random.shuffle(indices)
        improved = False
        for i in indices:
            if time.time() - start_time > time_limit:
                break
            if i == 0: # Keep 0 fixed? Or allow moving? 
                # Task says U contains 0. We can keep 0 fixed to simplify, 
                # or move it and subtract min(U) at the end.
                # Moving 0 is equivalent to shifting the whole set, which doesn't change 
                # |U-U| or |U+U|, but changes max(U) if we don't re-normalize.
                # Score depends on max(U). Shifting down reduces max(U), increasing score.
                # So we should always normalize U so min(U)=0.
                continue
            
            original_val = U[i]
            # Try a few random new positions
            # Range: roughly [0, 2*max_u]
            for _ in range(10):
                new_val = random.randint(0, int(2 * max_u) + 10)
                if new_val == original_val:
                    continue
                U[i] = new_val
                # Normalize
                min_u = min(U)
                if min_u != 0:
                    U = [x - min_u for x in U]
                # Ensure 0 is in U (it is, since min was subtracted)
                # Calculate score
                s = score_of(U)
                if s > best_score:
                    best_score = s
                    improved = True
                    max_u = max(U)
                else:
                    # Revert
                    U[i] = original_val
                    # Re-normalize if needed for next iteration? 
                    # We reverted U[i], but U might have been shifted.
                    # It's safer to keep U normalized in the loop.
                    min_u = min(U)
                    if min_u != 0:
                        U = [x - min_u for x in U]
        
        if not improved:
            # Try larger jumps or different strategy if stuck?
            # For now, just loop again with new random shuffle
            pass
            
    return U, best_score

def construct():
    deadline = time.time() + 110 # 10s margin
    best_set = [0, 1, 3]
    best_score = score_of(best_set)
    
    n = 4
    while time.time() < deadline:
        # Generate initial Golomb ruler
        U = generate_golomb(n)
        
        # Run local search for a short duration
        # Time slice: 5 seconds per n? 
        # If n is small, search is fast. If n is large, score_of is slower.
        # Let's use 2 seconds per n to allow more n's.
        search_time = 2.0
        if time.time() + search_time > deadline - 5:
            search_time = max(0, deadline - 5 - time.time())
            if search_time <= 0:
                break
                
        U_opt, s_opt = local_search(U, search_time)
        
        if s_opt > best_score:
            best_score = s_opt
            best_set = U_opt
            
        n += 1
        # Cap n to prevent score_of from becoming too slow
        if n > 30: 
             # For n=30, n^2=900, score_of is fast.
             # We can go higher.
             pass
             
    return best_set
