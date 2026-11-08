import time
from math import log, floor
import random
from itertools import combinations

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def geometric_set(base, num_terms):
    """Generate a geometric sequence set: {0, base, base^2, ..., base^(num_terms-1)}"""
    U = [0]
    val = 1
    for _ in range(num_terms):
        val *= base
        U.append(val)
    return U

def generate_candidates(n, max_val, num_candidates=50):
    """Generate diverse candidate sets for local search initialization."""
    candidates = []
    candidates_pool = list(range(1, max_val + 1))
    
    # Add geometric sets
    for base in range(2, min(10, max_val // 2 + 1)):
        if len(candidates_pool) >= n - 1:
            geo = geometric_set(base, n - 1)
            if max(geo) <= max_val:
                candidates.append(geo)
    
    # Add random sets
    for _ in range(num_candidates):
        if len(candidates_pool) >= n - 1:
            combo = random.sample(candidates_pool, n - 1)
            candidates.append([0] + sorted(combo))
    
    # Add arithmetic progression-like sets
    for start in range(1, min(5, max_val // 2 + 1)):
        for step in range(1, min(5, max_val // 2 + 1)):
            U = [0]
            val = start
            for _ in range(n - 1):
                if val <= max_val:
                    U.append(val)
                val += step
            if len(U) == n:
                candidates.append(U)
    
    return candidates

def exhaustive_search_small(n, max_val, deadline):
    """
    Exhaustive search for sets of size n containing 0, with max element <= max_val.
    """
    best_score = -1
    best_set = None
    
    if n - 1 > max_val:
        return best_score, best_set
    
    candidates = list(range(1, max_val + 1))
    
    for combo in combinations(candidates, n - 1):
        if time.time() > deadline:
            break
        U = [0] + list(combo)
        # U is already sorted because combo is from sorted candidates
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    return best_score, best_set

def local_search_dynamic(U, deadline, max_val_limit=100, max_n_limit=20):
    """
    Dynamic local search with adaptive move sizes.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    no_improvement_count = 0
    max_no_improvement = 30
    temperature = 1.0
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to ADD an element
            if len(U) < max_n_limit:
                # Try multiple candidates with varying distances
                num_attempts = min(50, max(10, int(50 * temperature)))
                for _ in range(num_attempts):
                    # Biased towards smaller values for denser sets
                    bias = random.random() ** 0.5
                    c = int(random.randint(1, max_val_limit) * bias) + 1
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        temperature = max(0.1, temperature * 0.9)
                        break
        elif move_type < 0.6:
            # Try to REMOVE an element
            if len(U) > 2:
                # Remove elements that are "outliers" or cause collisions
                removable = [x for x in U if x != 0]
                if removable:
                    # Score each removable element by how much removing it helps
                    best_removal = None
                    best_removal_score = current_score
                    for c in removable:
                        if time.time() > deadline - 3:
                            break
                        new_U = [x for x in U if x != c]
                        new_score = score_set(new_U)
                        if new_score > best_removal_score:
                            best_removal_score = new_score
                            best_removal = c
                    
                    if best_removal is not None and best_removal_score > current_score:
                        U = [x for x in U if x != best_removal]
                        current_score = best_removal_score
                        improved = True
                        temperature = max(0.1, temperature * 0.9)
        else:
            # Try to REPLACE an element
            if len(U) > 1:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                num_attempts = min(50, max(10, int(50 * temperature)))
                for _ in range(num_attempts):
                    # Try values near old_val and random values
                    if random.random() < 0.5:
                        delta = random.randint(-max(1, old_val // 3), max(1, old_val // 3))
                        new_val = max(1, min(max_val_limit, old_val + delta))
                    else:
                        new_val = random.randint(1, max_val_limit)
                    
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        temperature = max(0.1, temperature * 0.9)
                        break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
            temperature = min(1.0, temperature * 1.1)
                
    return best_local_score, best_local_set

def random_search(n, max_val, iterations, deadline):
    """
    Random search for sets of size n containing 0, with max element <= max_val.
    """
    best_score = -1
    best_set = None
    candidates = list(range(1, max_val + 1))
    
    for _ in range(iterations):
        if time.time() > deadline:
            break
        # Pick n-1 random distinct elements
        combo = random.sample(candidates, n - 1)
        U = [0] + sorted(combo)
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U
            
    return best_score, best_set

def local_search(U, deadline, max_val_limit=100, max_n_limit=15):
    """
    Local search to refine a set.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    no_improvement_count = 0
    max_no_improvement = 20
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to ADD an element
            if len(U) < max_n_limit:
                # Try a few random candidates
                for _ in range(20):
                    c = random.randint(1, max_val_limit)
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        break
        elif move_type < 0.7:
            # Try to REMOVE an element
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
        else:
            # Try to REPLACE an element
            if len(U) > 1:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                for _ in range(20):
                    new_val = random.randint(1, max_val_limit)
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def construct():
    start_time = time.time()
    deadline = start_time + 110

    best_set = [0, 1, 3]
    best_score = score_set(best_set)

    # 1. Exhaustive search for small n
    # n=3, M=50: C(50,2) = 1225
    # n=4, M=40: C(40,3) = 9880
    # n=5, M=35: C(35,4) = 52360
    # n=6, M=30: C(30,5) = 142506
    # n=7, M=25: C(25,6) = 177100
    
    search_configs = [
        (3, 50, 5),
        (4, 40, 10),
        (5, 35, 20),
        (6, 30, 40),
        (7, 25, 60),
    ]
    
    for n, m, time_limit in search_configs:
        if time.time() > deadline - time_limit - 10:
            break
        ex_score, ex_set = exhaustive_search_small(n, m, deadline - time_limit - 10)
        if ex_set and ex_score > best_score:
            best_score = ex_score
            best_set = ex_set

    # 2. Random search for slightly larger n
    random_configs = [
        (7, 40, 10000),
        (8, 35, 10000),
        (9, 30, 10000),
        (10, 25, 10000),
    ]
    
    for n, m, iters in random_configs:
        if time.time() > deadline - 20:
            break
        r_score, r_set = random_search(n, m, iters, deadline - 20)
        if r_set and r_score > best_score:
            best_score = r_score
            best_set = r_set

    # 3. Local search on the best set
    local_score, local_set = local_search(best_set, deadline, max_val_limit=100, max_n_limit=15)
    if local_score > best_score:
        best_score = local_score
        best_set = local_set

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
