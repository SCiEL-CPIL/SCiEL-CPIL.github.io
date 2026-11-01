import itertools
import math
import time

def calculate_score(U):
    U = sorted(set(U))
    if not U or 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a+b)
            diffset.add(a-b)
    r = len(diffset) / len(sumset)
    if r <= 1:
        return 1.0
    return 1 + math.log(r) / math.log(2*max_u + 1)

def construct():
    start_time = time.time()
    deadline = 100 # 100 seconds safety margin
    
    best_score = 1.0
    best_set = [0, 1, 3]
    
    # Search strategy:
    # 1. Try known good structures (powers of 2, etc.)
    # 2. Random search with local optimization
    
    # Initial candidates
    candidates = []
    for n in range(1, 15):
        U = [0] + [2**i for i in range(1, n)]
        candidates.append(U)
        
    # Search small max values
    for max_u in range(1, 25):
        if time.time() > deadline:
            break
        # Try all subsets of {0..max_u} of size 3 to 6
        # This is too slow for max_u=25 if we do all subsets.
        # Instead, fix 0 and max_u, and pick k elements from middle.
        for size in range(3, 7):
            if time.time() > deadline:
                break
            middle = list(range(1, max_u))
            # Limit combinations if too many
            if len(middle) > 20:
                # Random sampling
                import random
                random.seed(42)
                samples = [random.sample(middle, size-2) for _ in range(100)]
            else:
                samples = list(itertools.combinations(middle, size-2))
            
            for comb in samples:
                U = [0] + list(comb) + [max_u]
                s = calculate_score(U)
                if s > best_score:
                    best_score = s
                    best_set = U
    
    # Local optimization on best_set
    # Try adding/removing elements
    current_set = list(best_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        max_u = max(current_set)
        # Try adding elements up to max_u * 2
        for new_elem in range(1, max_u * 2 + 1):
            if new_elem in current_set:
                continue
            U_new = sorted(current_set + [new_elem])
            s = calculate_score(U_new)
            if s > best_score:
                best_score = s
                best_set = U_new
                current_set = U_new
                improved = True
                break
        
        # Try removing elements (except 0)
        for elem in list(current_set):
            if elem == 0:
                continue
            U_new = [x for x in current_set if x != elem]
            if len(U_new) < 2:
                continue
            s = calculate_score(U_new)
            if s > best_score:
                best_score = s
                best_set = U_new
                current_set = U_new
                improved = True
                break
                
    return best_set

# To ensure the function runs fast and returns a valid result
# I will hardcode the result of a pre-computation if possible,
# or run the search quickly.
# Given the 120s limit, the search above should finish quickly for small max_u.
