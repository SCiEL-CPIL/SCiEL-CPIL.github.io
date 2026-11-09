"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill
    
    # Precompute LCM table for all pairs up to 512
    # This is 512*512 = 262144 entries, very fast
    N = 512
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            val = i * j // gcd(i, j)
            lcm_table[i][j] = val
            lcm_table[j][i] = val
    
    # Start with all numbers 1..22
    result = set(range(1, 23))
    
    # Greedy: try to add each number from 23 to 512
    # Try multiple orderings and keep the best
    best_result = set(result)
    best_size = len(result)
    
    # Ordering 1: ascending
    candidates = list(range(23, N + 1))
    
    for ordering_name, order in [("asc", candidates), ("desc", list(reversed(candidates)))]:
        if time.time() > deadline:
            break
        current = set(result)
        for n in order:
            if time.time() > deadline:
                break
            compatible = True
            for d in current:
                if lcm_table[n][d] > 512:
                    compatible = False
                    break
            if compatible:
                current.add(n)
        
        if len(current) > best_size:
            best_size = len(current)
            best_result = set(current)
    
    return sorted(best_result)
# EVOLVE-BLOCK-END
