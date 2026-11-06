import math
import time
import numpy as np

def construct():
    """
    Finds an integer n in [10000, 1000000] that maximizes powerful_part(n) * powerful_part(n+1).
    powerful_part(x) is the product of p^e for all prime powers p^e dividing x with e >= 2.
    """
    t0 = time.time()
    DEADLINE = 85.0

    MAX_N = 1_000_000
    MAX_VAL = MAX_N + 1  # Need to check up to 1,000,001

    # 1. Sieve of Eratosthenes to find all primes up to MAX_VAL
    is_prime = bytearray(b'\x01') * (MAX_VAL + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(MAX_VAL)) + 1):
        if is_prime[i]:
            is_prime[i*i : MAX_VAL+1 : i] = b'\x00' * ((MAX_VAL - i*i)//i + 1)
    
    primes = [i for i in range(2, MAX_VAL + 1) if is_prime[i]]

    # 2. Compute powerful part using NumPy
    # For each prime p:
    #   - Divide by p for all multiples of p (removes squarefree part)
    #   - Multiply by p for all multiples of p^2 (restores the factor since v_p >= 2)
    # Net effect: v_p(m) >= 2 keeps p^v, v_p(m) == 1 removes p, v_p(m) == 0 unchanged
    
    pow_part_arr = np.arange(MAX_VAL + 1, dtype=np.int64)
    
    for p in primes:
        if time.time() - t0 > DEADLINE - 5:
            # Fallback: return a known valid number
            return 10000
        
        # Divide by p for all multiples of p
        mask = np.arange(p, MAX_VAL + 1, p, dtype=np.int64)
        pow_part_arr[mask] //= p
        
        # Multiply by p for all multiples of p^2
        p_sq = p * p
        if p_sq <= MAX_VAL:
            mask_sq = np.arange(p_sq, MAX_VAL + 1, p_sq, dtype=np.int64)
            pow_part_arr[mask_sq] *= p

    # 3. Maximize product pow_part[n] * pow_part[n+1] for n in [10000, 1000000]
    start = 10000
    end = 1_000_000
    
    # Extract relevant range: indices start to end+1
    seg = pow_part_arr[start : end+2]
    
    # Calculate products: prod[i] = seg[i] * seg[i+1]
    products = seg[:-1] * seg[1:]
    
    # Find the index of the maximum product
    best_idx = int(np.argmax(products))
    best_n = start + best_idx
    
    # Safety check
    if best_n < 10000 or best_n > 1000000:
        best_n = 10000
        
    return best_n
