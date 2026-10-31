import time
import math
import random

def calculate_score(U):
    if not U:
        return 1.0
    U = list(set(U))
    if 0 not in U:
        return 1.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    sum_set = set()
    diff_set = set()
    
    # Optimization: Use numpy if possible? 
    # For N < 1000, Python sets are fine.
    for u1 in U:
        for u2 in U:
            sum_set.add(u1 + u2)
            diff_set.add(u1 - u2)
            
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 1.0
        
    ratio = size_diff / size_sum
    if ratio <= 1:
        # If ratio <= 1, the log term is <= 0.
        # The score would be <= 1.
        # We want to maximize, so we prefer ratio > 1.
        pass
        
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    score = 1.0 + math.log(ratio) / denom
    return score

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_score = 1.0
    best_set = [0, 1, 3]
    
    # Strategy 1: Greedy Construction
    # We try to build a set of size N by adding one element at a time.
    # We try different target sizes.
    
    def greedy_build(target_size, max_val, seed):
        random.seed(seed)
        current = {0}
        # Candidate pool: integers from 1 to max_val
        # To make it efficient, we don't test ALL candidates at every step.
        # We test a random subset or a structured subset.
        
        for _ in range(target_size - 1):
            if time.time() > deadline:
                break
                
            best_candidate = None
            best_candidate_score = calculate_score(list(current))
            
            # Generate candidates
            # Bias towards numbers that might create unique differences
            # e.g., numbers far from existing ones, or specific structures
            candidates = set()
            
            # 1. Random candidates
            for _ in range(50):
                candidates.add(random.randint(1, max_val))
            
            # 2. Structured candidates (e.g., current max + 1, current max * 2, etc.)
            curr_max = max(current)
            candidates.add(curr_max + 1)
            candidates.add(curr_max * 2)
            candidates.add(curr_max * 2 + 1)
            candidates.add(1)
            
            for c in candidates:
                if c in current:
                    continue
                new_set = current | {c}
                # Quick check: if max increases significantly, denominator increases.
                # Calculate score
                s = calculate_score(list(new_set))
                if s > best_candidate_score:
                    best_candidate_score = s
                    best_candidate = c
            
            if best_candidate is not None:
                current.add(best_candidate)
            else:
                # No improvement found in this batch, maybe break or try harder
                # For simplicity, break if no improvement
                break
                
        return list(current)

    # Run greedy builds with different parameters
    seeds = [42, 123, 456, 789, 1000]
    target_sizes = [10, 20, 50, 100, 200]
    max_vals = [100, 500, 2000, 10000]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        for ts in target_sizes:
            if time.time() > deadline:
                break
            for mv in max_vals:
                if time.time() > deadline:
                    break
                try:
                    cand_set = greedy_build(ts, mv, seed)
                    s = calculate_score(cand_set)
                    if s > best_score:
                        best_score = s
                        best_set = cand_set
                except Exception:
                    pass

    # Strategy 2: Local Search / Hill Climbing on random sets
    # Start with a random set and try to improve it.
    
    def local_search(initial_set, iterations=1000):
        current = set(initial_set)
        if 0 not in current:
            current.add(0)
        current_score = calculate_score(list(current))
        
        for _ in range(iterations):
            if time.time() > deadline:
                break
            
            improved = False
            # Pick a random element to modify
            if len(current) <= 1:
                break
                
            u = random.choice(list(current))
            
            # Generate a new value for u
            # Bias towards nearby values or values that might increase spread
            new_u = u + random.randint(-5, 5)
            if new_u < 0:
                new_u = u + random.randint(1, 10)
            if new_u == u:
                continue
            if new_u in current:
                continue
                
            new_set = current - {u} | {new_u}
            new_score = calculate_score(list(new_set))
            
            if new_score > current_score:
                current = new_set
                current_score = new_score
                improved = True
                
            if not improved and random.random() < 0.1:
                # Occasionally add a new element or remove one?
                # Keep size constant for now to simplify
                pass
                
        return list(current), current_score

    # Run local search on a few random starts
    for seed in seeds:
        if time.time() > deadline:
            break
        random.seed(seed)
        # Generate random sets of various sizes
        for size in [10, 20, 50, 100]:
            if time.time() > deadline:
                break
            # Random set containing 0
            init_set = {0}
            for _ in range(size - 1):
                init_set.add(random.randint(1, 1000))
            
            refined_set, score = local_search(init_set)
            if score > best_score:
                best_score = score
                best_set = refined_set

    return best_set
