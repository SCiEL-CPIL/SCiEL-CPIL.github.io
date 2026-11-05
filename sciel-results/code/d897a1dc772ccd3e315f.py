import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Build smooth part using NumPy vectorized operations
    # smooth[i] = product of p^v_p(i) for p in small_primes
    # For each prime p and each power p^k, multiply smooth[j] by p for all j divisible by p^k
    smooth = np.ones(max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        pk = p
        while pk <= max_val:
            # All indices j that are multiples of pk get multiplied by p
            # This correctly accounts for v_p(j): j divisible by p^k means it gets
            # one factor of p for each k where p^k | j
            mask = np.arange(pk, max_val + 1, pk)
            smooth[mask] *= p
            pk *= p
    
    if time.time() > deadline:
        return 1000
    
    # Sliding window to find n minimizing distinct smooth values in [n+1, n+24]
    freq = {}
    for i in range(1, 25):
        val = int(smooth[1000 + i])
        freq[val] = freq.get(val, 0) + 1
    
    best_n = 1000
    min_distinct = len(freq)
    
    if min_distinct <= 1:
        return best_n
    
    for n in range(1001, max_n + 1):
        if (n & 0x1FFF) == 0 and time.time() > deadline:
            break
        
        v_out = int(smooth[n])
        freq[v_out] -= 1
        if freq[v_out] == 0:
            del freq[v_out]
            
        v_in = int(smooth[n + 24])
        freq[v_in] = freq.get(v_in, 0) + 1
        
        cur_distinct = len(freq)
        if cur_distinct < min_distinct:
            min_distinct = cur_distinct
            best_n = n
            if min_distinct <= 1:
                break
    
    return best_n
