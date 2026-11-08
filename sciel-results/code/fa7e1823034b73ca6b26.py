import time
import math
from itertools import combinations

def compute_score(U):
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s limit
    
    best_U = [0, 1, 3, 6, 9]
    best_score = compute_score(best_U)
    
    def try_set(U):
        nonlocal best_U, best_score
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]

    # Exhaustive search over subsets of {1,...,N} of various sizes
    # We balance pool size and set size to maximize coverage within time
    
    # Size 3: Pool 50
    pool3 = list(range(1, 51))
    for combo in combinations(pool3, 3):
        if time.time() > deadline:
            break
        U = [0] + list(combo)
        try_set(U)
    if time.time() > deadline:
        return best_U

    # Size 4: Pool 40
    pool4 = list(range(1, 41))
    for combo in combinations(pool4, 4):
        if time.time() > deadline:
            break
        U = [0] + list(combo)
        try_set(U)
    if time.time() > deadline:
        return best_U

    # Size 5: Pool 30
    pool5 = list(range(1, 31))
    for combo in combinations(pool5, 5):
        if time.time() > deadline:
            break
        U = [0] + list(combo)
        try_set(U)
    if time.time() > deadline:
        return best_U

    # Size 6: Pool 25
    pool6 = list(range(1, 26))
    for combo in combinations(pool6, 6):
        if time.time() > deadline:
            break
        U = [0] + list(combo)
        try_set(U)
    if time.time() > deadline:
        return best_U

    # Size 7: Pool 20
    pool7 = list(range(1, 21))
    for combo in combinations(pool7, 7):
        if time.time() > deadline:
            break
        U = [0] + list(combo)
        try_set(U)
    if time.time() > deadline:
        return best_U

    # Size 8: Pool 15
    pool8 = list(range(1, 16))
    for combo in combinations(pool8, 8):
        if time.time() > deadline:
            break
        U = [0] + list(combo)
        try_set(U)
    if time.time() > deadline:
        return best_U

    # Local Search Phase
    # Try to improve the best set found by adding, removing, or swapping elements
    current_U = best_U[:]
    current_score = best_score
    improved = True
    pool_local = list(range(1, 101))
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding an element
        for x in pool_local:
            if x in current_U:
                continue
            new_U = current_U + [x]
            s = compute_score(new_U)
            if s > current_score:
                current_U = new_U
                current_score = s
                improved = True
                if s > best_score:
                    best_score = s
                    best_U = current_U[:]
                break # Restart loop with new base
        
        if time.time() > deadline:
            break
            
        # Try removing an element
        for i in range(len(current_U)):
            if current_U[i] == 0:
                continue # Must keep 0
            new_U = current_U[:i] + current_U[i+1:]
            s = compute_score(new_U)
            if s > current_score:
                current_U = new_U
                current_score = s
                improved = True
                if s > best_score:
                    best_score = s
                    best_U = current_U[:]
                break # Restart loop with new base
        
        if time.time() > deadline:
            break

        # Try swapping an element
        for i in range(len(current_U)):
            if current_U[i] == 0:
                continue # Don't swap 0
            for x in pool_local:
                if x in current_U:
                    continue
                new_U = current_U[:]
                new_U[i] = x
                s = compute_score(new_U)
                if s > current_score:
                    current_U = new_U
                    current_score = s
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_U = current_U[:]
                    break # Restart loop with new base
            if improved:
                break
                
    return best_U
