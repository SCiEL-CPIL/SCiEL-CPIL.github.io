import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 256
    candidates = list(range(2, N + 1))
    
    # Precompute conflict information for fast validation
    # conflict[a][b] = set of c such that (a,b,c) is a bad triple
    # A triple (a,b,c) is bad if a|b*c or b|a*c or c|a*b
    
    # For speed, precompute for each pair (a,b) which c values conflict
    # But this is O(N^3) = 256^3 which is too much.
    
    # Alternative: for each a, precompute which pairs (b,c) have a|b*c
    # divisors_of_product[a] = set of (b,c) pairs where a divides b*c
    
    # Even better: use a different approach.
    # For a candidate set S, we need to check that for all distinct a,b,c in S:
    # a does not divide b*c
    
    # Key insight: if we sort the set, we can check conflicts more efficiently.
    # But let's first try a faster greedy with better heuristics.
    
    # Precompute number of divisors for each number
    div_counts = [0] * (N + 1)
    for i in range(2, N + 1):
        for j in range(i, N + 1, i):
            div_counts[j] += 1
    
    # Precompute: for each a, the set of b such that b | a (i.e., b divides a)
    # This helps check if b | a*c quickly
    divisors_of = [set() for _ in range(N + 1)]
    for b in range(2, N + 1):
        for a in range(b, N + 1, b):
            divisors_of[a].add(b)
    
    def is_valid_addition_fast(a, current_set):
        """Check if adding 'a' to 'current_set' creates any bad triple."""
        if len(current_set) < 2:
            return True
            
        cs = current_set
        n = len(cs)
        
        for i in range(n):
            b = cs[i]
            for j in range(i + 1, n):
                c = cs[j]
                # Check a | b*c
                if (b * c) % a == 0:
                    return False
                # Check b | a*c
                if (a * c) % b == 0:
                    return False
                # Check c | a*b
                if (a * b) % c == 0:
                    return False
        return True

    def greedy_search(order, use_fast=True):
        s = []
        for a in order:
            if is_valid_addition_fast(a, s):
                s.append(a)
        return s

    def local_search_improve(s):
        """Try to improve set s by swapping elements or adding/removing."""
        s = list(s)
        current_size = len(s)
        improved = True
        
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Try adding elements that weren't in the set
            for a in candidates:
                if a in s:
                    continue
                if is_valid_addition_fast(a, s):
                    s.append(a)
                    improved = True
                    break
            
            # Try swapping: remove one element, add another
            if not improved:
                for i, b in enumerate(s):
                    s_temp = s[:i] + s[i+1:]
                    for a in candidates:
                        if a in s_temp:
                            continue
                        if is_valid_addition_fast(a, s_temp):
                            s = s_temp + [a]
                            improved = True
                            break
                    if improved:
                        break
        
        return s

    # Define different sorting strategies
    strategies = []
    
    # 1. Descending value (large numbers have fewer divisibility issues)
    strategies.append(sorted(candidates, reverse=True))
    
    # 2. Ascending value
    strategies.append(sorted(candidates))
    
    # 3. Primes first (low number of divisors)
    strategies.append(sorted(candidates, key=lambda x: div_counts[x]))
    
    # 4. Numbers with few divisors that are large
    strategies.append(sorted(candidates, key=lambda x: (div_counts[x], -x)))
    
    # 5. Randomized with bias toward large numbers
    for _ in range(10):
        weighted_order = []
        for a in candidates:
            weight = a / N  # Larger numbers get higher weight
            if random.random() < weight:
                weighted_order.append(a)
        if len(weighted_order) < len(candidates):
            remaining = [a for a in candidates if a not in weighted_order]
            random.shuffle(remaining)
            weighted_order.extend(remaining)
        random.shuffle(weighted_order)
        strategies.append(weighted_order)
    
    best_set = []
    
    # Run deterministic strategies
    for order in strategies[:4]:
        s = greedy_search(order)
        if len(s) > len(best_set):
            best_set = s
    
    # Run randomized strategies
    for order in strategies[4:]:
        s = greedy_search(order)
        if len(s) > len(best_set):
            best_set = s
    
    # Try local search improvement on the best set found
    if time.time() < deadline - 10.0:
        improved = local_search_improve(best_set)
        if len(improved) > len(best_set):
            best_set = improved
    
    # More randomized restarts with local search if time permits
    while time.time() < deadline - 5.0:
        # Random shuffle with bias
        order = candidates[:]
        random.shuffle(order)
        s = greedy_search(order)
        
        # Try local search on this result
        if time.time() < deadline - 5.0:
            s = local_search_improve(s)
        
        if len(s) > len(best_set):
            best_set = s
    
    # Ensure we return a list of Python ints
    return [int(x) for x in best_set]
