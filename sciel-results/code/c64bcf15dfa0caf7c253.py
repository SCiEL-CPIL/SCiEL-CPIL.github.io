"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    target = 123457
    
    # Sieve-based precomputation of largest prime factor for all numbers up to target.
    # This enables O(1) lookup for P(n) for any n in [1, target].
    # Memory: ~123457 integers, very manageable.
    # Time: O(target * log(log(target))) which is fast for this size.
    
    if time.time() > deadline:
        return [1, target - 1]
    
    # Initialize largest_prime_factor array
    # lp[n] will store the largest prime factor of n
    lp = [0] * (target + 1)
    lp[0] = 0
    lp[1] = 1
    
    # Sieve: for each prime p, mark all multiples with p as their largest prime factor
    # We iterate through numbers and update lp for multiples
    for i in range(2, target + 1):
        if lp[i] == 0:
            # i is prime
            for j in range(i, target + 1, i):
                lp[j] = i  # Since we go in increasing order, the last prime that divides j is the largest
    
    if time.time() > deadline:
        return [1, target - 1]
    
    # Now search for the optimal split
    best_max_pf = float('inf')
    best_a = 1
    best_b = target - 1
    
    for a in range(1, target // 2 + 1):
        if time.time() > deadline:
            break
        
        b = target - a
        
        pa = lp[a]
        if pa >= best_max_pf:
            continue
            
        pb = lp[b]
        if pb >= best_max_pf:
            continue
            
        current_max = max(pa, pb)
        if current_max < best_max_pf:
            best_max_pf = current_max
            best_a = a
            best_b = b
            
            # Early exit if we reach a very low value
            # Theoretical minimum: since target is odd, one of a,b is even (P>=2) 
            # and the other is odd (P>=3). So min possible max is at least 3.
            # But 3^k + 3^m = 123457 has no solution (123457 not div by 3).
            # So min is at least 5. If we find 5, we can stop.
            if best_max_pf <= 5:
                break

    return [best_a, best_b]
# EVOLVE-BLOCK-END
