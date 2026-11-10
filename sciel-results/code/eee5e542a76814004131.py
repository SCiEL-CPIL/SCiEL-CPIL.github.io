"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes)
    
    # Precompute for each number in [1, 1000024]:
    # - which primes < 24 divide it (bitmask)
    # - the complete small-prime factor (product of p^v for p < 24)
    MAX_VAL = 1000024
    small_part = [1] * (MAX_VAL + 1)
    prime_mask = [0] * (MAX_VAL + 1)
    
    # Sieve-like: for each prime, compute its contribution
    for pi, p in enumerate(primes):
        bit = 1 << pi
        # For each multiple of p, extract p^v
        for m in range(p, MAX_VAL + 1, p):
            # Count power of p in m
            val = m
            while val % p == 0:
                val //= p
            p_part = m // val  # p^v
            small_part[m] *= p_part
            prime_mask[m] |= bit
    
    # For each n in [1000, 1000000], compute the union of prime masks for n+1..n+24
    # Use sliding window with a count array
    best_n = 1000
    best_count = 9  # worst case: all 9 primes
    
    # Compute initial window mask for n=1000
    # We need masks for positions 1001 to 1024
    # Sliding window approach: maintain count of each prime in window
    
    # For efficiency, precompute prefix of prime masks
    # Actually, let's use a different approach:
    # For each prime p, mark which positions in [1001, 1000024] have p in their small-part
    # Then for each window, count how many primes appear
    
    # More efficient: for each prime, create a boolean array of where it appears
    # Then for a window [n+1, n+24], a prime appears if any position in window has it
    
    # Even more efficient: for each prime p, compute the "next occurrence" array
    # next_occ[i] = smallest j >= i such that p divides small_part[j]
    # Then p appears in window [n+1, n+24] iff next_occ[n+1] <= n+24
    
    # Let's use a simpler approach: for each n, compute union mask
    # With 999001 windows and 24 elements each, that's ~24M operations - should be fast
    
    # Actually, let's be smarter. For each prime p, compute the set of n values
    # where p does NOT appear in any of n+1..n+24.
    # p does NOT appear in window starting at n iff
    #   for all i in 1..24, p does not divide small_part[n+i]
    #   i.e., n is not in {m-p : m is multiple of p, m in [n+1, n+24]}
    #   i.e., there's no multiple of p in [n+1, n+24]
    
    # p does NOT appear in [n+1, n+24] iff the gap between consecutive multiples of p
    # covers the entire interval. The multiples of p are at ..., p*k, p*(k+1), ...
    # p does NOT appear in [n+1, n+24] iff there exists k such that
    #   p*k < n+1 and p*(k+1) > n+24
    #   i.e., n+1 > p*k and n+24 < p*(k+1)
    #   i.e., n > p*k - 1 and n < p*(k+1) - 24
    #   i.e., p*k - 1 < n < p*(k+1) - 24
    #   i.e., n in (p*k - 1, p*(k+1) - 24)
    # This requires p*(k+1) - 24 > p*k - 1, i.e., p > 23.
    # For p <= 23, this is impossible! So EVERY prime p < 24 appears in EVERY window of length 24.
    
    # Wait, that means the answer is always 9? That can't be right.
    # Let me re-read: "extract its complete factor composed of primes STRICTLY BELOW 24"
    # So for number m, we extract the part of m that is composed of primes < 24.
    # E.g., if m = 2^3 * 5 * 29, the extracted factor is 2^3 * 5 = 40.
    # The distinct primes in the extracted factor are {2, 5}.
    
    # So for a window, we collect all extracted factors and count distinct primes across all.
    # A prime p appears in the window if p divides the extracted factor of some n+i.
    # p divides the extracted factor of m iff p divides m (since extracted factor contains all p^v for p<24).
    
    # So yes, p appears in window [n+1, n+24] iff some n+i is divisible by p.
    # And as I showed, for any p <= 23, in any 24 consecutive integers, at least one is divisible by p.
    # So ALL 9 primes always appear, and the answer is always 9?
    
    # Hmm, but the task says "MINIMIZE" and the seed gets 1.0. Let me reconsider.
    # Maybe I'm wrong about the gap argument. Let me check p=2:
    # In any 24 consecutive integers, there are 12 even numbers. So 2 always appears.
    # p=3: in any 24 consecutive, there are 8 multiples. Always appears.
    # p=5: in any 24 consecutive, there are at least floor(24/5)=4 multiples. Always appears.
    # p=7: in any 24 consecutive, there are at least floor(24/7)=3 multiples. Always appears.
    # p=11: in any 24 consecutive, there are at least floor(24/11)=2 multiples. Always appears.
    # p=13: in any 24 consecutive, there are at least floor(24/13)=1 multiple. Always appears.
    # p=17: in any 24 consecutive, there are at least floor(24/17)=1 multiple. Always appears.
    # p=19: in any 24 consecutive, there are at least floor(24/19)=1 multiple. Always appears.
    # p=23: in any 24 consecutive, there are at least floor(24/23)=1 multiple. Always appears.
    
    # So indeed, for ANY n, all 9 primes appear in the window. The minimum is 9.
    # But then the seed returning 1000 with score 1.0 means the "raw value" is 9,
    # and seed/value = 1.0 implies seed = 9.
    
    # Wait, but the task says "MINIMIZE the number of distinct extracted factors" - maybe it's
    # the number of distinct FACTORS (values), not distinct primes?
    
    # Let me re-read: "extract its complete factor composed of primes STRICTLY BELOW 24. 
    # MINIMIZE the number of distinct extracted factors."
    
    # "distinct extracted factors" - this could mean distinct VALUES of the extracted factors,
    # not distinct primes!
    
    # So for each n+i, we get a value f(n+i) = the part of n+i composed of primes < 24.
    # We want to minimize the number of DISTINCT values among {f(n+1), ..., f(n+24)}.
    
    # That's a different problem! Let me recompute.
    
    # The extracted factor of m is the product of p^v_p for all p < 24 where v_p >= 1.
    # So f(m) = m / (largest divisor of m coprime to all primes < 24)
    #          = m / (part of m composed of primes >= 29)
    
    # We want to minimize |{f(n+1), f(n+2), ..., f(n+24)}|.
    
    # The minimum possible is 1 (all 24 values equal).
    # This would require f(n+1) = f(n+2) = ... = f(n+24).
    
    # Let's search for such n.
    
    # Recompute small_part correctly (already done above)
    # Now search for n where all 24 small_part values are equal.
    
    # First, let's just find the minimum number of distinct values.
    
    # For efficiency, let's search:
    # 1. First try to find n where all 24 are equal (distinct count = 1)
    # 2. If not found, try distinct count = 2, etc.
    
    # But 999001 windows * 24 comparisons = ~24M, which is fast.
    
    # Let's just scan all windows and track the best.
    
    best_n = 1000
    best_distinct = 25  # max possible is 24 distinct values + 1 for safety
    
    # Precompute small_part array (already done)
    # Now scan windows
    
    for n in range(1000, 1000001):
        if time.time() > deadline:
            break
        
        # Get the 24 values
        vals = small_part[n+1:n+25]
        distinct = len(set(vals))
        
        if distinct < best_distinct:
            best_distinct = distinct
            best_n = n
            if best_distinct == 1:
                break  # Can't do better than 1
    
    return best_n
# EVOLVE-BLOCK-END
