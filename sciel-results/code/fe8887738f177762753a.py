# EVOLVE-BLOCK-START
from math import gcd
import time

def construct():
    N = 256
    start_time = time.time()
    
    # Precompute prime factors
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    pf = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        pf[i] = prime_factors(i)
    pf[1] = set()
    
    # Conflict sets
    conflict = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        for j in range(i + 1, N + 1):
            if pf[i] & pf[j]:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Order: 1 first (always safe), then others by value descending
    order = [1] + sorted(range(2, N + 1), key=lambda x: -x)
    
    best_sum = 0
    best_set = set()
    node_count = 0
    
    def bbs(idx, selected_set, current_sum):
        nonlocal best_sum, best_set, node_count
        node_count += 1
        
        if time.time() - start_time > 55:
            return
        
        # Upper bound: current + sum of all remaining that don't conflict with selected
        bound = current_sum
        for i in range(idx, N + 1):
            num = order[i]
            if not (conflict[num] & selected_set):
                bound += num
        
        if bound <= best_sum:
            return
        
        if idx == N + 1:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(selected_set)
            return
        
        num = order[idx]
        
        # Try include
        if not (conflict[num] & selected_set):
            selected_set.add(num)
            bbs(idx + 1, selected_set, current_sum + num)
            selected_set.remove(num)
        
        # Try exclude
        bbs(idx + 1, selected_set, current_sum)
    
    bbs(0, set(), 0)
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
