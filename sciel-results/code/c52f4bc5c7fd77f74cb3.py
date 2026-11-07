# EVOLVE-BLOCK-START
import math
import time

def construct():
    n = 256
    deadline = time.time() + 50  # safety margin
    
    # Find all primes up to 256
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    primes = sieve(n)
    
    # Initialize S with 1 and largest multiple of each prime
    selected = [1]
    for p in primes:
        largest_multiple = (n // p) * p
        selected.append(largest_multiple)
    
    # Ensure all are Python ints
    selected = [int(x) for x in selected]
    
    # Local search: try to replace x with y > x if y is coprime to all other elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate over indices to allow replacement
        for i in range(len(selected)):
            x = selected[i]
            if x == 1:
                continue
            # Try replacing x with the largest possible y > x
            # y must not be in selected
            # y must be coprime to all s in selected except x
            for y in range(n, x, -1):
                if y in selected:
                    continue
                valid = True
                for j in range(len(selected)):
                    if i == j:
                        continue
                    if math.gcd(y, selected[j]) != 1:
                        valid = False
                        break
                if valid:
                    # Replace x with y
                    selected[i] = y
                    improved = True
                    break
            if improved:
                # Restart the outer loop because the set changed
                break
                
    return selected
# EVOLVE-BLOCK-END
