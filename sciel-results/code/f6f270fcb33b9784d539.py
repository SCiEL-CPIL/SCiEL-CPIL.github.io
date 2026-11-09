"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    def lcm_lookup(a, b):
        return lcm_table[a][b]
    
    def is_conflict(x, selected):
        for i in range(len(selected)):
            a = selected[i]
            lab = lcm_lookup(a, x)
            for j in range(i + 1, len(selected)):
                b = selected[j]
                lac = lcm_lookup(a, b)
                lbc = lcm_lookup(b, x)
                if lab == lac == lbc:
                    return True
        return False
    
    def omega(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count
    
    def largest_prime_factor(x):
        max_p = 1
        d = 2
        while d * d <= x:
            while x % d == 0:
                max_p = d
                x //= d
            d += 1
        if x > 1:
            max_p = x
        return max_p
    
    best_result = None
    best_size = 0
    
    random.seed(42)
    
    orderings = []
    orderings.append(list(range(1, n + 1)))
    orderings.append(list(range(n, 0, -1)))
    orderings.append(sorted(range(1, n + 1), key=omega))
    orderings.append(sorted(range(1, n + 1), key=omega, reverse=True))
    orderings.append(sorted(range(1, n + 1), key=largest_prime_factor))
    orderings.append(sorted(range(1, n + 1), key=largest_prime_factor, reverse=True))
    for p in [2, 3, 5, 7]:
        orderings.append(sorted(range(1, n + 1), key=lambda x: x % p))
    
    for _ in range(100):
        perm = list(range(1, n + 1))
        random.shuffle(perm)
        orderings.append(perm)
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        
        selected = []
        for x in ordering:
            if time.time() > deadline:
                break
            if not is_conflict(x, selected):
                selected.append(x)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_result = list(selected)
            if best_size >= n:
                break
    
    # Local search: try to add more elements
    if best_result and time.time() < deadline - 3.0:
        current = list(best_result)
        current_set = set(current)
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for x in range(1, n + 1):
                if x in current_set:
                    continue
                if not is_conflict(x, current):
                    current.append(x)
                    current_set.add(x)
                    improved = True
                    break
        if len(current) > best_size:
            best_size = len(current)
            best_result = current
    
    if not best_result:
        best_result = [1]
    
    return best_result
# EVOLVE-BLOCK-END
