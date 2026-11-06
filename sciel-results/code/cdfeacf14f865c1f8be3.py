"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute valid residues for each prime
    # For a prime p, n is a "hit" if there exists i in 1..24 such that (n+i) % p == 0
    # This means n % p == (-i) % p
    valid_residues = {}
    for p in primes:
        res_set = set()
        for i in range(1, 25):
            res_set.add((-i) % p)
        valid_residues[p] = res_set
        
    best_n = 1000
    best_score = float('inf')
    
    # Deadline guard: 90s limit, aim for 85s
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Search range
    min_n = 1000
    max_n = 1000000
    
    # Iterate through n
    # Optimization: Unroll the prime checks for speed
    p0, p1, p2, p3, p4, p5, p6, p7, p8 = primes
    v0 = valid_residues[p0]
    v1 = valid_residues[p1]
    v2 = valid_residues[p2]
    v3 = valid_residues[p3]
    v4 = valid_residues[p4]
    v5 = valid_residues[p5]
    v6 = valid_residues[p6]
    v7 = valid_residues[p7]
    v8 = valid_residues[p8]
    
    for n in range(min_n, max_n + 1):
        if time.time() > deadline:
            break
            
        count = 0
        if n % p0 in v0: count += 1
        if n % p1 in v1: count += 1
        if n % p2 in v2: count += 1
        if n % p3 in v3: count += 1
        if n % p4 in v4: count += 1
        if n % p5 in v5: count += 1
        if n % p6 in v6: count += 1
        if n % p7 in v7: count += 1
        if n % p8 in v8: count += 1
        
        if count < best_score:
            best_score = count
            best_n = n
            # If we found 1, that's the theoretical minimum (at least one prime must hit? 
            # Actually, can we have 0? No, because 2 hits every other number, so in any 24-number interval, 2 will hit.
            # So minimum is 1? Wait.
            # If n is such that no number in n+1..n+24 is divisible by 3,5,7,...
            # 2 always hits.
            # Can we avoid 3? Interval length 24. 24 is multiple of 3. So 3 MUST hit.
            # Can we avoid 5? 24 is not multiple of 5. Yes, possible.
            # So minimum is at least 2 (primes 2 and 3).
            # If best_score == 2, we can potentially stop if we are confident it's global min, 
            # but we should finish the search to be sure, or break if we know 2 is min.
            # Given the speed, let's just finish the loop or break early if 2 is found and we assume 2 is min.
            # But to be safe, let's just run the whole thing, it's fast.
            
    return best_n
# EVOLVE-BLOCK-END
