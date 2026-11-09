"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import heapq
import math

def _get_primes(n=1000):
    """Return list of primes <= n in increasing order."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    primes = _get_primes(1000)
    n_primes = len(primes)  # 168
    max_check = 100000
    
    # Initialize residues randomly
    residues = [random.randint(0, p - 1) for p in primes]
    
    # Initialize counts
    counts = np.zeros(max_check + 2, dtype=np.int32)
    
    # Build initial counts
    for i, p in enumerate(primes):
        r = residues[i]
        if r == 0:
            first = p
        else:
            first = r
        for x in range(first, max_check + 1, p):
            counts[x] += 1
            
    # Build initial bad points heap
    bad_points = []
    in_bad = np.zeros(max_check + 2, dtype=bool)
    
    for x in range(1, max_check + 1):
        if counts[x] < 2:
            heapq.heappush(bad_points, x)
            in_bad[x] = True
            
    def get_L():
        while bad_points:
            x = bad_points[0]
            if in_bad[x] and counts[x] < 2:
                return x - 1
            else:
                heapq.heappop(bad_points)
        return max_check

    current_L = get_L()
    
    # Simulated Annealing parameters
    T = 5.0
    T_min = 0.01
    alpha = 0.9995
    iterations = 50000
    
    for _ in range(iterations):
        if T < T_min:
            break
            
        # Pick a random prime index
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = residues[i]
        
        # Pick a new random residue
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue
            
        # Determine ranges
        if old_r == 0:
            first_old = p
        else:
            first_old = old_r
            
        if new_r == 0:
            first_new = p
        else:
            first_new = new_r
            
        # Apply changes: Decrement old, Increment new
        # We will store the old counts if we need to revert, but updating in-place and reverting is symmetric.
        
        # Decrement old_r
        for x in range(first_old, max_check + 1, p):
            old_count = counts[x]
            counts[x] -= 1
            new_count = counts[x]
            
            if old_count >= 2 and new_count < 2:
                # Became bad
                if not in_bad[x]:
                    heapq.heappush(bad_points, x)
                    in_bad[x] = True
            elif old_count < 2 and new_count >= 2:
                # Became good
                if in_bad[x]:
                    in_bad[x] = False
            
        # Increment new_r
        for x in range(first_new, max_check + 1, p):
            old_count = counts[x]
            counts[x] += 1
            new_count = counts[x]
            
            if old_count >= 2 and new_count < 2:
                # Impossible with increment
                pass
            elif old_count < 2 and new_count >= 2:
                # Became good
                if in_bad[x]:
                    in_bad[x] = False

        new_L = get_L()
        
        # Acceptance
        if new_L >= current_L:
            current_L = new_L
            residues[i] = new_r
        else:
            delta = new_L - current_L
            # delta is negative
            prob = math.exp(delta / T)
            if random.random() < prob:
                current_L = new_L
                residues[i] = new_r
            else:
                # Revert
                # Revert Increment new_r (Decrement)
                for x in range(first_new, max_check + 1, p):
                    old_count = counts[x]
                    counts[x] -= 1
                    new_count = counts[x]
                    
                    if old_count >= 2 and new_count < 2:
                        if not in_bad[x]:
                            heapq.heappush(bad_points, x)
                            in_bad[x] = True
                    elif old_count < 2 and new_count >= 2:
                        if in_bad[x]:
                            in_bad[x] = False

                # Revert Decrement old_r (Increment)
                for x in range(first_old, max_check + 1, p):
                    old_count = counts[x]
                    counts[x] += 1
                    new_count = counts[x]
                    
                    if old_count >= 2 and new_count < 2:
                        pass
                    elif old_count < 2 and new_count >= 2:
                        if in_bad[x]:
                            in_bad[x] = False

        T *= alpha

    return residues
# EVOLVE-BLOCK-END
