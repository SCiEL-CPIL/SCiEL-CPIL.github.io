import math
import random
from itertools import combinations

def calc_score(U):
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs) / len(sums)
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 0
    return 1 + math.log(num) / den

def construct():
    best = [0, 1, 3]
    best_score = calc_score(best)
    
    # 1. Exhaustive search for small sets (size <= 4)
    # This catches the {0, 1, 3, 7} type structures quickly
    for max_val in range(2, 15):
        for size in range(3, 5):
            # Generate all subsets of {1, ..., max_val} of given size
            # and add 0
            candidates = list(range(1, max_val + 1))
            for combo in combinations(candidates, size - 1):
                U = [0] + list(combo)
                s = calc_score(U)
                if s > best_score:
                    best_score = s
                    best = U
    
    # 2. Heuristic search using base-B expansions
    # We use the "base-expansion hierarchy" idea
    # U is a subset of { sum b_j B^j }
    
    random.seed(42)
    
    for B in [2, 3, 4, 5, 7, 10]:
        # Generate a pool of candidates from base-B expansions
        # Up to 4-5 digits to keep numbers small enough for high score
        pool = set([0])
        for n in range(1, 6):
            for mask in range(1 << n):
                val = 0
                for j in range(n):
                    if mask & (1 << j):
                        val += B ** j
                if val < 200: # Limit range to keep denominator small
                    pool.add(val)
        
        pool = list(pool)
        
        # Local search / Hill Climbing
        # Start with a random subset of the pool
        for trial in range(50):
            # Initialize U
            U = set([0])
            # Add some random elements
            num_to_add = random.randint(2, 8)
            if num_to_add < len(pool):
                U.update(random.sample(pool, num_to_add))
            U = list(U)
            current_score = calc_score(U)
            
            # Iterate
            for _ in range(200):
                improved = False
                # Try adding
                if len(U) < 15:
                    candidate = random.choice(pool)
                    if candidate not in U:
                        new_U = U + [candidate]
                        new_s = calc_score(new_U)
                        if new_s > current_score:
                            U = new_U
                            current_score = new_s
                            improved = True
                
                # Try removing
                if len(U) > 2:
                    idx = random.randint(0, len(U) - 1)
                    if U[idx] != 0:
                        new_U = U[:idx] + U[idx+1:]
                        new_s = calc_score(new_U)
                        if new_s > current_score:
                            U = new_U
                            current_score = new_s
                            improved = True
                
                # Try swapping
                if len(U) > 2:
                    idx = random.randint(0, len(U) - 1)
                    if U[idx] != 0:
                        candidate = random.choice(pool)
                        if candidate not in U:
                            new_U = U.copy()
                            new_U[idx] = candidate
                            new_s = calc_score(new_U)
                            if new_s > current_score:
                                U = new_U
                                current_score = new_s
                                improved = True
                
                if not improved:
                    break
            
            if current_score > best_score:
                best_score = current_score
                best = U

    return best
