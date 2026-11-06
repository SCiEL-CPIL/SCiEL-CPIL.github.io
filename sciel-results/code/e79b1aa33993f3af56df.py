"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Domain definitions
    SMALL_RANGE = range(2, 129)  # [2, 128]
    LARGE_RANGE = range(129, 257) # [129, 256]
    
    small_numbers = list(SMALL_RANGE)
    large_numbers = list(LARGE_RANGE)
    
    # Precompute weights
    weight = {x: 1.0 / x for x in small_numbers}
    weight.update({x: 1.0 / x for x in large_numbers})
    
    # Precompute conflicts within small numbers
    # conf_small[a] is a set of b in small_numbers such that lcm(a, b) <= 256
    conf_small = {}
    for a in small_numbers:
        confs = set()
        for b in small_numbers:
            if a == b:
                continue
            if (a * b) // gcd(a, b) <= 256:
                confs.add(b)
        conf_small[a] = confs
        
    # Precompute conflicts between small and large
    # conf_large[a] is a set of b in large_numbers such that lcm(a, b) <= 256
    conf_large = {}
    for a in small_numbers:
        confs = set()
        for b in large_numbers:
            if (a * b) // gcd(a, b) <= 256:
                confs.add(b)
        conf_large[a] = confs
        
    # Precompute reciprocal sums for large numbers to speed up evaluation if needed
    # But we will use incremental updates.
    
    # Helper to check if x is compatible with current set A
    # x is in small_numbers
    def is_compatible_small(x, A):
        for a in A:
            if x in conf_small[a]:
                return False
        return True

    # State:
    # A: set of small numbers
    # conf_count: dict mapping large number -> count of conflicts from A
    # current_score: float
    
    def get_initial_solution():
        # Greedy: sort by weight desc, add if compatible
        A = set()
        conf_count = {b: 0 for b in large_numbers}
        score = 0.0
        
        for x in sorted(small_numbers, key=lambda y: -weight[y]):
            if is_compatible_small(x, A):
                A.add(x)
                score += weight[x]
                for b in conf_large[x]:
                    conf_count[b] += 1
                    if conf_count[b] == 1:
                        score -= weight[b]
        
        # Add all compatible large numbers
        for b in large_numbers:
            if conf_count[b] == 0:
                score += weight[b]
                
        return A, conf_count, score

    def evaluate_move_add(A, conf_count, score, x):
        # Check compatibility
        if not is_compatible_small(x, A):
            return None
            
        new_A = A | {x}
        new_conf_count = conf_count.copy()
        delta_score = weight[x]
        
        for b in conf_large[x]:
            if new_conf_count[b] == 0:
                delta_score -= weight[b]
            new_conf_count[b] += 1
            
        new_score = score + delta_score
        return new_A, new_conf_count, new_score

    def evaluate_move_remove(A, conf_count, score, x):
        if x not in A:
            return None
            
        new_A = A - {x}
        new_conf_count = conf_count.copy()
        delta_score = -weight[x]
        
        for b in conf_large[x]:
            new_conf_count[b] -= 1
            if new_conf_count[b] == 0:
                delta_score += weight[b]
            
        new_score = score + delta_score
        return new_A, new_conf_count, new_score

    def evaluate_swap(A, conf_count, score, x_out, x_in):
        if x_out not in A or x_in in A:
            return None
        if not is_compatible_small(x_in, A - {x_out}):
            return None
            
        new_A = (A - {x_out}) | {x_in}
        new_conf_count = conf_count.copy()
        delta_score = weight[x_in] - weight[x_out]
        
        # Remove x_out effects
        for b in conf_large[x_out]:
            new_conf_count[b] -= 1
            if new_conf_count[b] == 0:
                delta_score += weight[b]
                
        # Add x_in effects
        for b in conf_large[x_in]:
            if new_conf_count[b] == 0:
                delta_score -= weight[b]
            new_conf_count[b] += 1
            
        new_score = score + delta_score
        return new_A, new_conf_count, new_score

    # Main search loop
    best_A, best_conf_count, best_score = get_initial_solution()
    
    # Local Search
    # We will try to improve best_score
    
    # To avoid getting stuck, we can use multiple restarts with randomization
    # Or just strong local search on the best found so far.
    
    # Given the structure, a good initial solution is crucial.
    # Let's run a few random greedy starts and pick the best.
    
    random.seed(42)
    
    for restart in range(10):
        if time.time() > deadline - 5.0:
            break
            
        # Randomized Greedy
        A = set()
        conf_count = {b: 0 for b in large_numbers}
        score = 0.0
        
        # Shuffle small numbers, but bias towards small values (high weight)
        # We can sort by weight * random factor
        shuffled_small = small_numbers[:]
        random.shuffle(shuffled_small)
        # Sort by weight descending, but with some noise? 
        # Actually, standard greedy on weight is strong. Let's just use random order for diversity.
        # Or sort by weight and skip with some probability?
        
        # Let's stick to weight-sorted greedy for the first few, then random.
        if restart < 5:
            candidates = sorted(small_numbers, key=lambda y: -weight[y])
        else:
            candidates = shuffled_small
            
        for x in candidates:
            if is_compatible_small(x, A):
                A.add(x)
                score += weight[x]
                for b in conf_large[x]:
                    conf_count[b] += 1
                    if conf_count[b] == 1:
                        score -= weight[b]
        
        for b in large_numbers:
            if conf_count[b] == 0:
                score += weight[b]
                
        if score > best_score:
            best_score = score
            best_A = A
            best_conf_count = conf_count

    # Local Improvement on best solution found
    A = set(best_A)
    conf_count = best_conf_count.copy()
    score = best_score
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        
        # 1. Try Additions
        best_add_move = None
        best_add_score = score
        for x in small_numbers:
            if x in A:
                continue
            res = evaluate_move_add(A, conf_count, score, x)
            if res:
                new_A, new_cc, new_score = res
                if new_score > best_add_score + 1e-9:
                    best_add_score = new_score
                    best_add_move = ('add', x, new_A, new_cc)
        
        if best_add_move:
            _, x, A, conf_count = best_add_move
            score = best_add_score
            improved = True
            continue # Restart loop to find more additions
            
        # 2. Try Removals
        best_rem_move = None
        best_rem_score = score
        for x in A:
            res = evaluate_move_remove(A, conf_count, score, x)
            if res:
                new_A, new_cc, new_score = res
                if new_score > best_rem_score + 1e-9:
                    best_rem_score = new_score
                    best_rem_move = ('remove', x, new_A, new_cc)
        
        if best_rem_move:
            _, x, A, conf_count = best_rem_move
            score = best_rem_score
            improved = True
            continue
            
        # 3. Try Swaps
        best_swap_move = None
        best_swap_score = score
        for x_out in A:
            for x_in in small_numbers:
                if x_in in A:
                    continue
                res = evaluate_swap(A, conf_count, score, x_out, x_in)
                if res:
                    new_A, new_cc, new_score = res
                    if new_score > best_swap_score + 1e-9:
                        best_swap_score = new_score
                        best_swap_move = ('swap', x_out, x_in, new_A, new_cc)
                        
        if best_swap_move:
            _, x_out, x_in, A, conf_count = best_swap_move
            score = best_swap_score
            improved = True
            continue
            
    # Final construction
    if not A:
        # Fallback: just pick 2
        return [2]
        
    final_set = set(A)
    # Add large numbers
    for b in large_numbers:
        if conf_count[b] == 0:
            final_set.add(b)
            
    return list(final_set)

# EVOLVE-BLOCK-END
