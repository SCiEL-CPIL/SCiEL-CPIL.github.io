import numpy as np
import time
import bisect

def construct():
    start_time = time.time()
    # Use a conservative deadline to ensure we return within 60s
    # Sieve takes ~15-20s, search takes ~5s. Total ~25s.
    deadline = 55.0
    
    N = 10_000_000
    
    # 1. Precompute sigma(n) for n in [1, N]
    # Using a sieve method: sigma(n) = sum of divisors
    # sigmas[i::i] += i adds i to all multiples of i
    sigmas = np.zeros(N + 1, dtype=np.uint32)
    
    # To speed up the Python loop overhead, we can't easily vectorize the outer loop.
    # But we can check time periodically.
    # However, breaking early leaves invalid data. 
    # We assume 10^7 is fast enough.
    for i in range(1, N + 1):
        sigmas[i::i] += i
        # Optional: check time
        # if i % 1000000 == 0 and time.time() - start_time > deadline - 5:
        #     break # This would corrupt the result, so we rely on it being fast enough.
            
    # 2. Prepare data for searching pairs with equal sigma
    # We only care about n >= 1
    indices = np.arange(1, N + 1, dtype=np.int64)
    sig_vals = sigmas[1:]
    
    # Sort indices by their sigma value
    # 'stable' sort preserves original order for equal sigmas, which might help locality?
    # Actually, we re-sort the group values later anyway.
    order = np.argsort(sig_vals, kind='stable')
    sorted_sigmas = sig_vals[order]
    sorted_indices = indices[order]
    
    best_a = 11
    best_b = 6
    best_err = abs(11.0 / 6.0 - 1.5)
    
    # 3. Iterate through groups of equal sigma
    i = 0
    n_items = N
    while i < n_items:
        j = i
        # Find the end of the current group
        current_sigma = sorted_sigmas[i]
        while j < n_items and sorted_sigmas[j] == current_sigma:
            j += 1
        
        group_size = j - i
        if group_size >= 2:
            # Extract the numbers in this group
            # These are the indices (values of n)
            group_vals = sorted_indices[i:j]
            # Sort the values to allow binary search
            group_vals.sort()
            
            # For each b in the group, find a in the group closest to 1.5 * b
            # We want to minimize |a/b - 1.5|
            # Since group_vals is sorted, we can use bisect
            for b_val in group_vals:
                target_a = 1.5 * b_val
                # Find position where target_a would be inserted
                pos = bisect.bisect_left(group_vals, target_a)
                
                # Check the candidate at pos and pos-1
                # Note: a and b must be distinct? 
                # If a == b, ratio is 1. |1 - 1.5| = 0.5.
                # The problem doesn't forbid a == b, but sigma(a)=sigma(b) is trivial.
                # However, usually "ratio" implies distinct or just any pair.
                # "Approximate a ratio...". 
                # If a=b, error is 0.5. 
                # Our initial best is 0.333. So a=b is worse.
                # But we should check anyway just in case.
                
                for p in (pos, pos - 1):
                    if 0 <= p < group_size:
                        a_val = group_vals[p]
                        # Calculate error
                        # Use float for comparison, but exact rational for final check?
                        # The task says "Minimize |a/b - 3/2|".
                        # "Factorization and rational error are exact... score is binary64 serialization"
                        # So we can use float for search, as precision is high enough.
                        err = abs(a_val / b_val - 1.5)
                        
                        if err < best_err:
                            best_err = err
                            best_a = a_val
                            best_b = b_val
                            if best_err == 0.0:
                                # Can't get better than 0
                                return [int(best_a), int(best_b)]
        
        i = j
        
    return [int(best_a), int(best_b)]
