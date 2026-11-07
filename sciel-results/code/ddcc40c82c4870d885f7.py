import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    best_m = 333312
    best_ks = [336, 372, 434]
    best_count = 3

    def try_search(N):
        nonlocal best_m, best_ks, best_count
        
        if time.time() > deadline - 5:
            return

        try:
            # Sieve for sigma(k)
            sigma = np.ones(N + 1, dtype=np.int64)
            for i in range(1, N + 1):
                if time.time() > deadline - 10:
                    return
                sigma[i::i] += i
            
            ks_arr = np.arange(1, N + 1, dtype=np.int64)
            vals = ks_arr * sigma
            
            # Condition: M <= 10^18 and k <= sqrt(M) <=> k <= sigma(k)
            mask = (vals <= 10**18) & (ks_arr <= sigma)
            
            valid_ks = ks_arr[mask]
            valid_ms = vals[mask]
            
            if len(valid_ms) == 0:
                return

            # Group by M
            unique_ms, counts, inverse = np.unique(valid_ms, return_counts=True, return_inverse=True)
            
            max_count = np.max(counts)
            if max_count > best_count:
                best_count = int(max_count)
                target_idx = np.argmax(counts)
                best_m = int(unique_ms[target_idx])
                
                # Extract ks for this M
                target_mask = (inverse == target_idx)
                target_ks = valid_ks[target_mask]
                best_ks = [int(x) for x in target_ks]
                
        except MemoryError:
            pass
        except Exception:
            pass

    # Try increasing N
    # 20 million is a good starting point for ~1-2 seconds sieve time
    try_search(20_000_000)
    
    # If time remains, try larger
    if time.time() < deadline - 20:
        try_search(50_000_000)
        
    if time.time() < deadline - 40:
        try_search(100_000_000)

    # Finalize
    best_ks = list(set(int(k) for k in best_ks))
    best_ks.sort()
    
    return {'M': int(best_m), 'k': best_ks}
# EVOLVE-BLOCK-END
