import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_bad_triple(a, b, c):
    return _lcm(a, b) == _lcm(a, c) == _lcm(b, c)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Precompute LCM table for all pairs in [1, 128]
    N = 128
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Precompute bad triples: for each pair (a,b), find all c such that (a,b,c) is bad
    # bad_pair[a][b] = set of c values that form a bad triple with a and b
    bad_pair = [[set() for _ in range(N + 1)] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[a][b]
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_pair[a][b].add(c)
                    bad_pair[b][a].add(c)
    
    def _can_add_fast(x, S):
        """Check if x can be added to S without creating a bad triple."""
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                if x in bad_pair[a][b]:
                    return False
        return True
    
    def _local_search(S, remaining_pool, time_limit):
        """Try to improve S by swapping elements."""
        S = set(S)
        S_list = sorted(S)
        best_size = len(S)
        best_set = set(S)
        
        # Build a list of elements not in S that could potentially be added
        candidates = [x for x in range(1, N + 1) if x not in S]
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try removing one element and adding one or more
            for idx, remove_x in enumerate(S_list):
                if time.time() > time_limit:
                    break
                S_trial = S - {remove_x}
                # Try to add candidates
                added_any = False
                for add_x in candidates:
                    if time.time() > time_limit:
                        break
                    if _can_add_fast(add_x, S_trial):
                        S_trial.add(add_x)
                        added_any = True
                        if len(S_trial) > best_size:
                            best_size = len(S_trial)
                            best_set = set(S_trial)
                            S = set(S_trial)
                            S_list = sorted(S)
                            candidates = [x for x in range(1, N + 1) if x not in S]
                            improved = True
                            break
                if improved:
                    break
        return best_set
    
    # 1. Build the base set of prime powers and 1
    base_set = set()
    base_set.add(1)
    
    # Add prime powers
    for p in range(2, 129):
        is_prime = True
        if p < 2:
            is_prime = False
        else:
            for i in range(2, int(p**0.5) + 1):
                if p % i == 0:
                    is_prime = False
                    break
        if is_prime:
            power = p
            while power <= 128:
                base_set.add(power)
                power *= p
                
    # 2. Add remaining numbers greedily
    remaining = [x for x in range(2, 129) if x not in base_set]
    
    S = set(base_set)
    
    # Greedy pass 1: Increasing order
    for x in sorted(remaining):
        if _can_add_fast(x, S):
            S.add(x)
            
    best_set = set(S)
    best_size = len(S)
    
    # Try random orderings
    all_candidates = list(remaining)
    
    for _ in range(20):
        if time.time() > deadline:
            break
        random.shuffle(all_candidates)
        S_current = set(base_set)
        for x in all_candidates:
            if _can_add_fast(x, S_current):
                S_current.add(x)
        if len(S_current) > best_size:
            best_size = len(S_current)
            best_set = set(S_current)
    
    # Local search to try to improve
    if time.time() < deadline - 5:
        improved_set = _local_search(best_set, remaining, deadline - 2)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
    
    # Final validation
    S_list = sorted(best_set)
    valid = True
    for a, b, c in combinations(S_list, 3):
        if _is_bad_triple(a, b, c):
            valid = False
            break
            
    if not valid:
        return list(base_set)
        
    if not best_set:
        return [1]
        
    return list(best_set)
