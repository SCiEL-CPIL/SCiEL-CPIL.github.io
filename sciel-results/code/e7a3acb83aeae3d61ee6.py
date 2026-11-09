import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Try to search up to N
    # Start with a smaller N to get a quick result, then expand if time allows?
    # Or just pick a large N and hope it finishes.
    # Given 10^7 took reasonable time in previous run, let's try 10^7.
    N = 10**7
    
    try:
        sigma = np.zeros(N + 1, dtype=np.int64)
        # Sieve for sigma
        # This loop is the bottleneck
        for i in range(1, N + 1):
            if time.time() > deadline - 10.0:
                break
            sigma[i::i] += i
        
        # Compute k * sigma(k)
        k_vals = np.arange(1, N + 1, dtype=np.int64)
        prod = k_vals * sigma
        
        # Filter M <= 10^18
        # prod is int64, max value 10^18 fits in int64?
        # 10^18 < 2^63-1 (~9.2 * 10^18). Yes.
        mask = prod <= 10**18
        valid_prods = prod[mask]
        valid_ks = k_vals[mask]
        
        if len(valid_prods) == 0:
             # Fallback to seed
             return {'M': 333312, 'k': [336, 372, 434]}
             
        # Group by prod
        # Use pandas? No, stick to numpy.
        # Sort by prod
        # To count occurrences, we can use np.unique with return_counts
        # But we need the indices to retrieve k's
        
        # Since N is 10^7, sorting 10^7 elements takes ~1s.
        sorted_idx = np.argsort(valid_prods)
        sorted_prods = valid_prods[sorted_idx]
        sorted_ks = valid_ks[sorted_idx]
        
        # Find groups
        # Change in prod
        changes = np.where(sorted_prods[1:] != sorted_prods[:-1])[0] + 1
        start_indices = np.concatenate(([0], changes))
        end_indices = np.concatenate((changes, [len(sorted_prods)]))
        
        counts = end_indices - start_indices
        
        # Find max count
        max_count = np.max(counts)
        if max_count < 3:
             # Fallback to seed if we don't find better
             return {'M': 333312, 'k': [336, 372, 434]}
             
        # Find all M with max_count
        # There might be multiple M with the same max count.
        # We just need one.
        best_idx = np.argmax(counts)
        
        best_M = int(sorted_prods[start_indices[best_idx]])
        best_ks = sorted_ks[start_indices[best_idx]:end_indices[best_idx]]
        
        # Ensure distinct k (they are by construction from unique k_vals)
        # Ensure k <= sqrt(M)
        # The problem states k_i <= sqrt(M).
        # Our k_vals are 1..N.
        # We must verify k <= sqrt(M).
        sqrt_M = int(np.sqrt(best_M))
        valid_best_ks = [int(k) for k in best_ks if k <= sqrt_M]
        
        if len(valid_best_ks) < len(best_ks):
             # Some k were too large?
             # If we filtered them, the count drops.
             # We should have filtered before grouping or checked during grouping.
             # Let's re-filter the whole set.
             pass # This logic is getting messy.
             
        # Better approach:
        # Filter k <= sqrt(M) is tricky because M depends on k.
        # But k * sigma(k) = M.
        # If k > sqrt(M), then k > sqrt(k sigma(k)) => k^2 > k sigma(k) => k > sigma(k).
        # sigma(k) >= k+1 (for k>1). So k > k+1 is impossible.
        # So for k>1, k < sigma(k) is always true?
        # sigma(k) = 1 + k + ... >= 1+k.
        # So k^2 < k sigma(k) = M => k < sqrt(M).
        # So all k > 1 satisfy k < sqrt(M).
        # For k=1, M=sigma(1)=1, sqrt(1)=1, k=1 <= 1.
        # So the constraint k <= sqrt(M) is always satisfied for k * sigma(k) = M.
        
        return {'M': best_M, 'k': [int(k) for k in valid_best_ks]}
        
    except Exception as e:
        # Fallback
        return {'M': 333312, 'k': [336, 372, 434]}
