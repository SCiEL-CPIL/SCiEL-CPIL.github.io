import time
from math import log
import itertools
import numpy as np
import random

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
        
    # Use NumPy for faster computation
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute all pairwise sums (i <= j to avoid duplicates, but we want unique sums)
    # Actually, we need all unique sums from i,j where 0<=i<=j<n
    # But for the sumset, we just need all unique values of u_i + u_j
    # Using outer sum and taking unique
    sums_arr = U_arr[:, None] + U_arr[None, :]
    unique_sums = np.unique(sums_arr)
    
    # Compute all pairwise differences
    diffs_arr = U_arr[:, None] - U_arr[None, :]
    unique_diffs = np.unique(diffs_arr)
    
    ratio = len(unique_diffs) / len(unique_sums)
    if ratio <= 1:
        return -1
        
    max_u = U_arr[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def construct():
    deadline = time.time() + 110  # Safety margin
    
    best_score = -1
    best_set = [0, 1, 4, 9, 11]
    
    # Update best if initial is good
    s = score_set(best_set)
    if s > best_score:
        best_score = s
    
    # Phase 1: Exhaustive search for small sets with moderate max_val
    max_val = 50
    
    for n in range(3, 7): # Sizes 3 to 6
        if time.time() > deadline:
            break
            
        for combo in itertools.combinations(range(1, max_val + 1), n - 1):
            if time.time() > deadline:
                break
                
            U = (0,) + combo
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    # Phase 2: Local search to refine the best set found
    if time.time() < deadline - 15:
        # Try adding elements
        current_set = set(best_set)
        max_current = max(best_set)
        
        # Try adding elements up to 300
        for new_elem in range(1, 300):
            if time.time() > deadline:
                break
            if new_elem in current_set:
                continue
            new_set = current_set | {new_elem}
            s = score_set(list(new_set))
            if s > best_score:
                best_score = s
                best_set = list(new_set)
                current_set = new_set
                max_current = max(best_set)
        
        # Try removing elements (keep at least 2 elements)
        if len(best_set) > 2:
            for elem in list(best_set):
                if time.time() > deadline:
                    break
                if elem == 0:
                    continue  # Must keep 0
                new_set = set(best_set) - {elem}
                if len(new_set) < 2:
                    continue
                s = score_set(list(new_set))
                if s > best_score:
                    best_score = s
                    best_set = list(new_set)
    
    # Phase 3: Simulated annealing for better exploration
    if time.time() < deadline - 10:
        random.seed(42)
        current = set(best_set)
        current_score = best_score
        T = 0.1
        T_min = 0.001
        cooling = 0.995
        
        while T > T_min and time.time() < deadline - 5:
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.5:
                # Add a random element
                max_elem = max(current) if current else 0
                new_elem = random.randint(1, max_elem * 2 + 100)
                if new_elem not in current:
                    candidate = current | {new_elem}
            elif move_type < 0.8:
                # Remove a random element (not 0)
                removable = [x for x in current if x != 0]
                if removable and len(current) > 2:
                    remove_elem = random.choice(removable)
                    candidate = current - {remove_elem}
                else:
                    continue
            else:
                # Swap: remove one, add another
                removable = [x for x in current if x != 0]
                if removable:
                    remove_elem = random.choice(removable)
                    max_elem = max(current)
                    new_elem = random.randint(1, max_elem * 2 + 100)
                    if new_elem not in current:
                        candidate = (current - {remove_elem}) | {new_elem}
                    else:
                        continue
                else:
                    continue
            
            candidate_score = score_set(list(candidate))
            
            if candidate_score > current_score or random.random() < np.exp((candidate_score - current_score) / T):
                current = candidate
                current_score = candidate_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current)
            
            T *= cooling
    
    return best_set
