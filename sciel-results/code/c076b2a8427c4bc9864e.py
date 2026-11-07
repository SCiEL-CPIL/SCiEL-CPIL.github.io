import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s

    N = 10_000_000

    # Compute sigma(n) for all n up to N using SPF (smallest prime factor) sieve
    # Then use the multiplicative property of sigma to compute sigma in O(N)
    
    # Step 1: Compute SPF for all numbers up to N
    spf = np.zeros(N + 1, dtype=np.int64)
    spf[1] = 1
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i::i] = i
    
    # Step 2: Compute sigma(n) using multiplicative property
    # sigma(n) = prod_{p^k || n} (p^(k+1)-1)/(p-1)
    sigma = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    
    # For each n from 2 to N:
    # Let p = spf[n], m = n // p
    # If m % p == 0, let n = p^k * m' where p doesn't divide m'
    #   sigma(n) = sigma(m') * (p^(k+1)-1)/(p-1)
    # If m % p != 0, n = p * m with gcd(p, m) = 1
    #   sigma(n) = sigma(m) * (p+1)
    
    # We need to track the "sigma contribution" of the prime power part
    # Let's use a different approach: for each n, compute using the recurrence
    
    # Better approach: for each n, find its smallest prime factor p
    # Let m = n // p
    # If p divides m, then n has p^k for k >= 2
    #   We need to track the highest power of p dividing n
    
    # Let's use arrays:
    # pp[n] = largest power of spf[n] dividing n
    # sigma[n] = sigma(n // pp[n]) * (pp[n]*(p-1) + 1) / (p-1) ... this is getting complex
    
    # Alternative: direct sieve method but vectorized more aggressively
    # sigma[n] = sum_{d|n} d
    # For each i from 1 to N, add i to sigma[i], sigma[2i], sigma[3i], ...
    
    # Let's try a hybrid: use the fact that for large i, there are few multiples
    # And for small i, we can vectorize
    
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # For i from 1 to N, sigma[i::i] += i
    # This is O(N log N) but with numpy vectorization per step
    # The total number of operations is sum_{i=1}^{N} N/i = N * H_N ~ N * ln(N)
    # For N = 10^7, that's about 10^7 * 16 = 1.6 * 10^8 additions
    # But each step is a vectorized numpy operation
    
    # To speed this up, let's only go up to N and use the fact that 
    # for i > N//2, each i contributes to exactly one multiple (itself)
    
    # Actually, let's just do the full sieve and hope it's fast enough
    # We can also precompute in chunks
    
    for i in range(1, N + 1):
        if time.time() > deadline - 5.0:
            break
        sigma[i::i] += i
    
    if time.time() > deadline - 3.0:
        return [11, 6]
    
    # Now find pairs with equal sigma values
    # sigma[1] to sigma[N]
    sig_vals = sigma[1:]  # sigma(1) to sigma(N)
    indices = np.arange(1, N + 1)
    
    # Sort by sigma value
    order = np.argsort(sig_vals, kind='mergesort')
    sorted_sig = sig_vals[order]
    sorted_idx = indices[order]
    
    # Find groups of equal sigma values
    changes = np.where(np.diff(sorted_sig) != 0)[0] + 1
    boundaries = np.concatenate(([0], changes, [len(sorted_sig)]))
    
    best_diff = float('inf')
    best_a = 11
    best_b = 6
    
    # For each group, use two-pointer to find min |2a - 3b|
    for k in range(len(boundaries) - 1):
        if time.time() > deadline:
            break
        lo = boundaries[k]
        hi = boundaries[k + 1]
        if hi - lo < 2:
            continue
        
        group_idx = sorted_idx[lo:hi]  # sorted array of n values
        
        n_group = len(group_idx)
        
        # Two-pointer: for each b, find a closest to 3b/2
        # Since both a and b come from the same sorted group_idx,
        # we can use the fact that as b increases, 3b/2 increases,
        # so the optimal a is non-decreasing
        
        # For each position pos_b, ideal_a = 3 * group_idx[pos_b] / 2
        # Find position in group_idx closest to ideal_a
        
        pos_a = 0
        for pos_b in range(n_group):
            b_val = group_idx[pos_b]
            ideal_a = 3.0 * b_val / 2.0
            
            # Advance pos_a to be near ideal_a
            while pos_a < n_group - 1 and group_idx[pos_a] < ideal_a:
                pos_a += 1
            
            # Check pos_a and pos_a-1 (if valid)
            for pa in [pos_a, pos_a - 1]:
                if pa < 0 or pa >= n_group:
                    continue
                a_val = group_idx[pa]
                diff_num = abs(2 * a_val - 3 * b_val)
                if diff_num == 0:
                    continue  # Invalid: raw value 0
                err = diff_num / (2.0 * b_val)
                if err < best_diff:
                    best_diff = err
                    best_a = a_val
                    best_b = b_val
    
    best_a = int(best_a)
    best_b = int(best_b)
    
    return [best_a, best_b]
