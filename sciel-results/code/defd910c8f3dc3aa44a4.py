import random
import time
import math
from itertools import combinations

def is_sidon_fast(subset):
    sums = [False] * 257
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if sums[s]:
                return False
            sums[s] = True
    return True

def get_max_sidon_size(candidate):
    n = len(candidate)
    for k in range(n, 2, -1):
        for comb in combinations(range(n), k):
            subset = [candidate[i] for i in comb]
            if is_sidon_fast(subset):
                return k
    return 2

def count_collisions(candidate):
    # Count number of pairs of pairs (a,b) and (c,d) with a+b = c+d and {a,b} != {c,d}
    # Pairs are (i,j) with 0 <= i <= j < 12
    sums = {}
    n = len(candidate)
    for i in range(n):
        for j in range(i, n):
            s = candidate[i] + candidate[j]
            if s in sums:
                sums[s] += 1
            else:
                sums[s] = 1
    
    total_collisions = 0
    for s, count in sums.items():
        if count > 1:
            # Number of pairs of pairs with this sum
            # We want to count distinct pairs of pairs.
            # If a sum appears k times, there are C(k,2) pairs of pairs.
            total_collisions += count * (count - 1) // 2
    return total_collisions

def construct():
    # Start with a dense set to have many collisions
    current = list(range(12))
    current_conflicts = count_collisions(current)
    current_exact = get_max_sidon_size(current)
    
    best = current[:]
    best_conflicts = current_conflicts
    best_exact = current_exact
    
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    
    start_time = time.time()
    timeout = 55.0
    
    iter_count = 0
    
    while T > T_min and (time.time() - start_time) < timeout:
        # Generate neighbor
        candidate = current[:]
        
        # Replace 1 or 2 elements
        num_changes = 1 if random.random() < 0.8 else 2
        indices = random.sample(range(12), num_changes)
        for i in indices:
            while True:
                v = random.randint(0, 128)
                if v not in candidate:
                    break
            candidate[i] = v
        
        candidate_conflicts = count_collisions(candidate)
        
        # SA step: Maximize conflicts
        # Delta = current - candidate (positive if candidate is worse, i.e., fewer conflicts)
        delta = current_conflicts - candidate_conflicts
        
        accepted = False
        if candidate_conflicts >= current_conflicts:
            accepted = True
        else:
            # delta > 0
            prob = math.exp(-delta / T)
            if random.random() < prob:
                accepted = True
        
        if accepted:
            current = candidate
            current_conflicts = candidate_conflicts
            
            # Check exact size periodically
            # Or if we found a new conflict record
            iter_count += 1
            if iter_count % 50 == 0 or candidate_conflicts > best_conflicts:
                # Compute exact size
                candidate_exact = get_max_sidon_size(candidate)
                if candidate_exact < best_exact:
                    best_exact = candidate_exact
                    best = candidate[:]
                    best_conflicts = candidate_conflicts
                else:
                    # Even if not better exact, update best_conflicts if we want to track the "densest" set
                    # But we only care about exact.
                    # However, using best_conflicts to trigger checks might be useful.
                    if candidate_conflicts > best_conflicts:
                         best_conflicts = candidate_conflicts
                         # Don't update best set if exact is worse, but keep tracking conflicts?
                         # Actually, if exact is worse, we don't care about that set.
                         # But maybe a slightly less dense set has better exact? Unlikely.
                         # Let's just track the best exact.
                         pass
        
        T *= cooling
        
    # Final check
    # It's possible the very last move was good but not checked
    if current != best:
         final_exact = get_max_sidon_size(current)
         if final_exact < best_exact:
             best = current[:]
             best_exact = final_exact

    return best
