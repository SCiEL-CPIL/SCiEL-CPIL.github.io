"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

def generate_powerful_numbers(limit):
    """
    Generate all powerful numbers up to limit.
    A powerful number is of the form a^2 * b^3 where b is square-free.
    However, generating for all b (not just square-free) and taking a set is simpler and correct,
    as duplicates are handled by the set.
    """
    if limit < 1:
        return []
    
    powerfuls = set()
    max_b = int(limit ** (1/3)) + 2
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= limit:
                powerfuls.add(n)
    
    return sorted(powerfuls)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer for 150s limit
    
    # Phase 1: Broad scan with a smaller limit to find candidate Ns
    L1 = 10**7
    pow_list_1 = generate_powerful_numbers(L1)
    n1 = len(pow_list_1)
    
    # Count sums for the small set to find top candidates
    sum_counts = defaultdict(int)
    # Optimization: use local variables
    pl = pow_list_1
    for i in range(n1):
        a = pl[i]
        for j in range(i, n1):
            s = a + pl[j]
            sum_counts[s] += 1
            
    # Get top 50 candidate Ns from the small scan
    # We sort by count descending, then by N (arbitrary tie-breaker)
    sorted_items = sorted(sum_counts.items(), key=lambda x: (-x[1], x[0]))
    candidates = [item[0] for item in sorted_items[:50]]
    
    # Phase 2: Deep dive with a larger limit
    # We want to check these candidates against a larger set of powerful numbers.
    # Let's try to generate powerful numbers up to L2.
    # L2 should be as large as possible. 
    # Generation time is roughly O(L2^(2/3) * L2^(1/2)) ? No, count is ~ L2^0.5.
    # Let's aim for L2 = 10^11. Count ~ 3*10^5.
    # Checking 50 candidates against 3*10^5 numbers:
    # For each candidate N, we iterate through pow_list_2 and check if N - a is in set.
    # That's 50 * 300,000 = 15,000,000 operations. Very fast.
    
    # Let's push L2 higher if time permits.
    # Let's start with L2 = 10^10.
    L2 = 10**10
    pow_list_2 = generate_powerful_numbers(L2)
    pow_set_2 = set(pow_list_2)
    n2 = len(pow_list_2)
    
    # Check time
    if time.time() > deadline - 20:
         # If we are running out of time, just return the best from Phase 1
         # But Phase 1 candidates are N <= 2*L1 = 2*10^7.
         # We need to construct pairs for the best candidate from Phase 1 using pow_list_1.
         best_N = candidates[0] if candidates else 2
         pairs = []
         for a in pow_list_1:
             if 2 * a > best_N: break
             b = best_N - a
             if b in pow_set_2: # Using set_2 is safer if we generated it, but pow_list_1 is subset
                 # Actually, if b > L1, it might not be in pow_list_1, but it IS in pow_set_2 if b <= L2.
                 # Wait, if we only have pow_list_1, we can't check b > L1.
                 # But in Phase 1, we only found N using pairs from pow_list_1.
                 # So for the "fallback", we should use pow_set_1.
                 pass
         # Let's just rely on Phase 2 being fast enough.
         # Generation of 10^10 powerfuls is ~ 100k-200k numbers. Takes < 1s.
         # So we are likely safe.
         pass

    # If time is still tight, we might have generated L2=10^10.
    # Let's try to go even higher if we have time.
    # Count of powerfuls up to 10^12 is ~ 10^6.
    # Checking 50 candidates * 10^6 = 50M ops. Might take 5-10s in Python.
    # Let's stick to L2 = 10^10 for safety, or dynamically increase if fast.
    
    # Let's refine L2 based on remaining time.
    # Current time used:
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    # If we have plenty of time, try L2 = 10^11
    if remaining > 30:
        L2 = 10**11
        pow_list_2 = generate_powerful_numbers(L2)
        pow_set_2 = set(pow_list_2)
        n2 = len(pow_list_2)
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
    # If we still have time, try L2 = 10^12
    if remaining > 40:
        L2 = 10**12
        pow_list_2 = generate_powerful_numbers(L2)
        pow_set_2 = set(pow_list_2)
        n2 = len(pow_list_2)

    # Now, for each candidate N, count how many pairs (a,b) from pow_list_2 sum to N.
    # Note: a <= b.
    # We iterate through pow_list_2.
    # For each a, b = N - a.
    # We need b >= a and b in pow_set_2.
    
    best_N = 0
    best_count = 0
    best_pairs = []
    
    # We can optimize by only checking candidates that are <= 2 * max(pow_list_2)
    max_pow = pow_list_2[-1] if pow_list_2 else 0
    
    for N in candidates:
        if N > 2 * max_pow:
            continue
            
        count = 0
        pairs = []
        
        # Iterate through powerful numbers
        # Optimization: a must be <= N/2
        for a in pow_list_2:
            if a > N // 2:
                break
            b = N - a
            if b in pow_set_2:
                count += 1
                pairs.append([a, b])
                
        if count > best_count:
            best_count = count
            best_N = N
            best_pairs = pairs

    # It's possible that a non-candidate N has a higher count in the larger range.
    # But this is unlikely if the density is somewhat uniform or correlated.
    # To be safer, we could also do a full count for the top few candidates 
    # and maybe check a few "random" Ns? No, that's inefficient.
    # The structural change is the two-phase approach.
    
    # One edge case: what if the best N from Phase 1 is not in candidates? 
    # We took top 50. It's very likely the global best is in there.
    
    # If no pairs found (shouldn't happen), fallback
    if not best_pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
