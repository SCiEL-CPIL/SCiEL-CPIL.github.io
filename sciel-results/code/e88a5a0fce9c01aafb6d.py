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
    start_time = time.time()
    limit_time = start_time + 290.0

    primes = _primes(N)
    
    # Start with all points
    active = [True] * (N + 1)
    active_count = N + 1
    
    # Greedy: for each prime p, if all residue classes are present, remove the smallest class
    for p in primes:
        if time.time() > limit_time:
            break
        if p > active_count:
            break
        
        # Count active points in each residue class mod p
        residue_counts = [0] * p
        for x in range(N + 1):
            if active[x]:
                residue_counts[x % p] += 1
        
        if min(residue_counts) > 0:
            # Find class with fewest active points
            min_idx = min(range(p), key=lambda i: residue_counts[i])
            # Remove all active points in this class
            for x in range(min_idx, N + 1, p):
                if active[x]:
                    active[x] = False
                    active_count -= 1
    
    # Build current set and removed list
    current_set = [x for x in range(N + 1) if active[x]]
    removed = [x for x in range(N + 1) if not active[x]]
    
    # Precompute prime indices for quick lookup
    # We need to check primes p <= current_size
    # Let's store counts and zero_counts for all primes up to N
    # counts[p][r]
    # zero_counts[p]
    
    # Initialize counts for current set
    # Note: We only care about primes p <= N
    counts = {}
    zero_counts = {}
    for p in primes:
        if p > N:
            break
        c = [0] * p
        for x in current_set:
            c[x % p] += 1
        counts[p] = c
        zero_counts[p] = c.count(0)
        
    # Local search: try to add back removed points
    # We iterate through removed points.
    # To be safe and potentially find better configurations, we could do multiple passes
    # or sort removed points by some heuristic.
    # Simple heuristic: try all, add if valid.
    # Since adding points fills holes, the order might matter.
    # Let's try iterating in reverse order of removal? Or just random?
    # Given time, let's stick to a single pass but with the efficient check.
    # We can also try to restart the local search a few times with different orderings if time permits.
    
    # Efficient check function
    def can_add(x, current_size):
        # Check primes p <= current_size
        for p in primes:
            if p > current_size:
                break
            r = x % p
            c = counts[p]
            zc = zero_counts[p]
            if c[r] == 0:
                if zc == 1:
                    return False
        return True

    # We will perform multiple passes to ensure we don't miss points due to ordering
    # Pass 1: Ascending
    # Pass 2: Descending
    # Pass 3: Random (if time)
    
    # Helper to add a point
    def add_point(x):
        nonlocal active_count
        active[x] = True
        active_count += 1
        current_set.append(x)
        for p in primes:
            if p > N:
                break
            r = x % p
            counts[p][r] += 1
            if counts[p][r] == 1: # Was 0, now 1
                zero_counts[p] -= 1

    # Pass 1: Ascending
    for x in removed:
        if time.time() > limit_time:
            return sorted(current_set)
        
        # Only check if x is not already added (it isn't, from 'removed')
        if can_add(x, active_count):
            add_point(x)
            
    # Pass 2: Descending
    # Re-identify removed (those not in current_set)
    # Actually, we can just iterate over the original 'removed' list again and check if active[x] is False
    for x in reversed(removed):
        if time.time() > limit_time:
            return sorted(current_set)
            
        if not active[x]:
            if can_add(x, active_count):
                add_point(x)
                
    # Pass 3: Randomized/Shuffled if time permits
    # Let's check time
    if time.time() < limit_time - 5.0:
        import random
        # Create a list of currently removed points
        still_removed = [x for x in range(N + 1) if not active[x]]
        random.shuffle(still_removed)
        for x in still_removed:
            if time.time() > limit_time:
                break
            if can_add(x, active_count):
                add_point(x)
                
    return sorted(current_set)
# EVOLVE-BLOCK-END
