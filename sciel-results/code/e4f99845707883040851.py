import math
import time

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    limit_time = start_time + 290.0  # 10s margin before 300s limit

    # Start with all points in [0, N]
    active_points = list(range(N + 1))
    active_set = set(active_points)
    
    # Generate primes up to N
    primes = _primes(N)
    
    # Iterate through each prime in increasing order
    for p in primes:
        if time.time() > limit_time:
            break
        
        # Early termination: if p > len(active_points), no residue class mod p
        # can be fully covered, so we can stop
        if p > len(active_points):
            break
        
        # Count active points in each residue class mod p
        residue_counts = [0] * p
        for x in active_points:
            residue_counts[x % p] += 1
        
        # Check if all classes are non-empty
        if min(residue_counts) > 0:
            # Find the class with the fewest elements
            min_idx = min(range(p), key=lambda i: residue_counts[i])
            
            # Deactivate all points in the smallest class
            # Build set of points to remove for fast lookup
            to_remove = set()
            for x in active_points:
                if x % p == min_idx:
                    to_remove.add(x)
            
            # Remove those points
            active_points = [x for x in active_points if x not in to_remove]
            active_set -= to_remove

    return sorted(active_points)
# EVOLVE-BLOCK-END
