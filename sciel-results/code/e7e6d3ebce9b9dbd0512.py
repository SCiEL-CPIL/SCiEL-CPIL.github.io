import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # LCM of 1..22
    L = reduce(lcm, range(1, 23))
    candidates = list(range(1, 23))
    values = {i: L // i for i in candidates}
    
    # Seed set
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    
    def greedy_build(order):
        S = []
        sums = {0}
        for i in order:
            v = values[i]
            # Incremental update of subset sums
            # new_sums = sums U (sums + v)
            new_sums = set(sums)
            for s in sums:
                new_sums.add(s + v)
            
            # Check if all sums are distinct
            # If distinct, len(new_sums) should be 2 * len(sums)
            if len(new_sums) == 2 * len(sums):
                S.append(i)
                sums = new_sums
                
        return S

    # Deterministic orders
    # Ascending
    S = greedy_build(candidates)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
        if best_size == 22: return best_set

    # Descending
    S = greedy_build(sorted(candidates, reverse=True))
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
        if best_size == 22: return best_set

    # Random restarts
    # We can also try to "warm start" by using the best set found so far as a base?
    # No, greedy is order-dependent.
    
    # To maximize the chance of finding a large set, we can also try
    # to prioritize elements that were in the best set found so far?
    # Let's just do pure random restarts for now.
    
    while time.time() < deadline:
        order = candidates[:]
        random.shuffle(order)
        
        S = greedy_build(order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            if best_size == 22:
                return best_set
                
    return best_set
