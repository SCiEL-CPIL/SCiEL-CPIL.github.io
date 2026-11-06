import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    lower = 1000
    upper = 1000000
    max_val = upper + 24
    
    # We need smooth parts for numbers from lower+1 to max_val
    # Range: [1001, 1000024]
    start_num = lower + 1
    end_num = max_val
    count = end_num - start_num + 1
    
    # Create array of numbers
    nums = np.arange(start_num, end_num + 1, dtype=np.int64)
    
    # Initialize smooth parts to 1
    smooth = np.ones(count, dtype=np.int64)
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Extract smooth parts
    for p in small_primes:
        temp = nums.copy()
        factor = np.ones(count, dtype=np.int64)
        # Divide out p from temp and accumulate in factor
        while True:
            mask = (temp % p) == 0
            if not np.any(mask):
                break
            temp[mask] //= p
            factor[mask] *= p
        smooth *= factor
        
    # Sliding window to find min distinct factors
    best_n = lower
    best_count = 25  # Max possible distinct is 24, so 25 is infinity
    
    # Frequency dictionary for current window
    freq = {}
    distinct = 0
    
    # Initialize window for n = lower (indices 0 to 23 in smooth array)
    for i in range(24):
        v = int(smooth[i])
        if v not in freq:
            freq[v] = 0
            distinct += 1
        freq[v] += 1
        
    if distinct < best_count:
        best_count = distinct
        best_n = lower
        if best_count == 1:
            return best_n

    # Slide window from n = lower + 1 to upper
    # For n, the window covers smooth[n - start_num] to smooth[n - start_num + 23]
    # Previous window (n-1) covered smooth[n - start_num - 1] to smooth[n - start_num + 22]
    # Outgoing: smooth[n - start_num - 1]
    # Incoming: smooth[n - start_num + 23]
    
    for n in range(lower + 1, upper + 1):
        if time.time() > deadline:
            break
            
        idx = n - start_num
        
        # Remove outgoing element: smooth[idx - 1]
        v_out = int(smooth[idx - 1])
        if v_out in freq:
            freq[v_out] -= 1
            if freq[v_out] == 0:
                del freq[v_out]
                distinct -= 1
        
        # Add incoming element: smooth[idx + 23]
        v_in = int(smooth[idx + 23])
        if v_in not in freq:
            freq[v_in] = 0
            distinct += 1
        freq[v_in] += 1
        
        if distinct < best_count:
            best_count = distinct
            best_n = n
            if best_count == 1:
                break
                
    return best_n
