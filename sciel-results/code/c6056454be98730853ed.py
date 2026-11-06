import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _build_spf(limit):
    """Build smallest prime factor sieve up to limit."""
    spf = np.arange(limit + 1, dtype=np.int64)
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:
            # Mark multiples of i
            start = i * i
            step = i
            spf[start:limit + 1:step] = np.where(spf[start:limit + 1:step] == spf[start:limit + 1:step], i, spf[start:limit + 1:step])
            # The above np.where is tricky because spf changes. 
            # Standard sieve is easier with lists for this specific "smallest" property update.
            # Let's stick to list for SPF building for correctness and simplicity, 
            # then convert to numpy for the PP computation if needed, or just use lists.
            # Actually, the SPF building in the original code was fine.
            pass
    # Re-implementing SPF with list for safety, then convert to numpy array for PP calc?
    # Or just use the original list-based SPF which is fast enough.
    spf_list = list(range(limit + 1))
    if limit >= 0: spf_list[0] = 0
    if limit >= 1: spf_list[1] = 1
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf_list[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf_list[j] == j:
                    spf_list[j] = i
    return spf_list

def _compute_powerful_parts(max_n, spf):
    """Compute powerful part for all numbers up to max_n."""
    pp = [1] * (max_n + 1)
    for i in range(2, max_n + 1):
        temp = i
        p = spf[temp]
        e = 0
        # Extract p^e
        while temp % p == 0:
            temp //= p
            e += 1
        if e >= 2:
            pp[i] = p ** e
        # Continue with remaining factors
        # Optimization: We can just loop through factors.
        # The original code did this correctly.
        # Let's replicate the original logic but in a function.
        # Actually, the original code was:
        # while temp > 1:
        #   p = spf[temp]
        #   e = 0
        #   while temp % p == 0: ...
        #   if e >= 2: pp *= p**e
        # This is correct.
        
    # The above function is a bit messy. Let's stick to the original efficient loop.
    return pp

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    max_n = 1000000
    spf = _build_spf(max_n)
    
    # Precompute powerful part for all numbers in [1, max_n]
    pp = [1] * (max_n + 1)
    for i in range(2, max_n + 1):
        if time.time() > deadline - 5:
            break
        temp = i
        p = spf[temp]
        e = 0
        val = 1
        while temp > 1:
            p = spf[temp]
            e = 0
            while temp % p == 0:
                temp //= p
                e += 1
            if e >= 2:
                val *= p ** e
        pp[i] = val
    
    # Use NumPy for the final scan to be structurally different and faster
    pp_arr = np.array(pp, dtype=np.int64)
    
    # We need n in [10000, 999999] such that n+1 <= 1000000
    # So n ranges from 10000 to 999999
    # pp[n] * pp[n+1]
    
    # Slice the array
    # n starts at 10000
    # n+1 starts at 10001
    # n ends at 999999
    # n+1 ends at 1000000
    
    start_idx = 10000
    end_idx = max_n - 1 # 999999
    
    # products[i] = pp[start_idx + i] * pp[start_idx + i + 1]
    # We can do this with a loop or vectorized multiplication of shifted arrays.
    
    # Vectorized:
    a = pp_arr[start_idx:end_idx + 1] # pp[10000] to pp[999999]
    b = pp_arr[start_idx + 1:end_idx + 2] # pp[10001] to pp[1000000]
    
    products = a * b
    
    # Find index of max
    max_idx = np.argmax(products)
    
    best_n = start_idx + max_idx
    best_val = products[max_idx]
    
    # Safety check: ensure best_n is in range
    if best_n < 10000 or best_n > 1000000:
        best_n = 10000
    
    return best_n

# EVOLVE-BLOCK-END
