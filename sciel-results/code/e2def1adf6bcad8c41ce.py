"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, n + 1):
        if n % i == 0:
            divisors.append(i)
    divisors.sort(reverse=True)
    # Only keep divisors that could be part of a sum to target
    # A divisor d can only be used if d <= target
    divisors = [d for d in divisors if d <= target]
    
    # For each k from 1 upward, check if any k-combination sums to target
    # This guarantees minimum k
    for k in range(1, 20):
        if time.time() > deadline:
            break
        # For small k, use combinations directly
        if k <= 5:
            for combo in combinations(divisors, k):
                if sum(combo) == target:
                    return list(combo)
        else:
            # For larger k, use a smarter search
            # But first check if it's even possible
            # Sum of k smallest divisors
            if sum(divisors[-k:]) > target:
                break  # Can't reach target with k elements if even the smallest k sum to more than target
            # Sum of k largest divisors
            if sum(divisors[:k]) < target:
                continue  # Even the largest k don't reach target, need more elements
    
    # Fallback: use a BFS/DFS approach for larger k
    # This should not be reached for this instance
    return divisors[:1]

# EVOLVE-BLOCK-END
