"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    N = 256
    
    # Precompute prime factors for each number
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]:
            for j in range(i, N + 1, i):
                prime_factors[j].add(i)
    
    def conflicts(a, b):
        return len(prime_factors[a] & prime_factors[b]) > 0
    
    def greedy_fill(selected, deadline):
        """Greedily try to add numbers to selected, starting from largest."""
        changed = True
        while changed:
            changed = False
            for num in range(N, 1, -1):
                if time.time() > deadline - 5:
                    return
                if num in selected:
                    continue
                conflicting = [x for x in selected if x != 1 and conflicts(num, x)]
                if not conflicting:
                    selected.add(num)
                    changed = True
                elif num > sum(conflicting):
                    for x in conflicting:
                        selected.remove(x)
                    selected.add(num)
                    changed = True
    
    # Start with 1 + all primes
    selected = set([1])
    for i in range(2, N + 1):
        if len(prime_factors[i]) == 1:  # prime
            selected.add(i)
    
    greedy_fill(selected, deadline)
    best_sum = sum(selected)
    best_set = set(selected)
    
    # Local search: try removing each element and re-optimizing
    if time.time() < deadline - 15:
        for remove_candidate in sorted(best_set, key=lambda x: -x):
            if time.time() > deadline - 10:
                break
            if remove_candidate == 1:
                continue
            trial = set(best_set)
            trial.remove(remove_candidate)
            greedy_fill(trial, deadline)
            if sum(trial) > best_sum:
                best_sum = sum(trial)
                best_set = set(trial)
    
    # Convert to list of Python ints
    result = [int(x) for x in best_set]
    return result

# EVOLVE-BLOCK-END
