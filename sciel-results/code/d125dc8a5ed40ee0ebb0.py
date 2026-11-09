import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 1000000
    MAX_VAL = N + 1
    
    # 1. Vectorized Sieve of Eratosthenes
    is_prime = np.ones(MAX_VAL + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i : MAX_VAL+1 : i] = False
            
    primes = np.nonzero(is_prime)[0]
    
    # 2. Compute powerful part
    pow_part = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in primes:
        p_val = int(p)
        power = p_val * p_val
        while power <= MAX_VAL:
            # Multiples of 'power'
            indices = np.arange(power, MAX_VAL + 1, power)
            
            # Filter: exponent is exactly current power's exponent
            # m = k * power. We want k % p_val != 0
            # This is equivalent to m not being a multiple of power * p_val
            
            # Optimization: 
            # Instead of calculating mask on the large array of indices,
            # we can subtract the set of multiples of power*p_val?
            # Or just use the mask as before, it's efficient enough.
            
            if p_val > 1:
                mask = (indices // power) % p_val != 0
                valid_indices = indices[mask]
                if len(valid_indices) > 0:
                    pow_part[valid_indices] *= power
            
            power *= p_val
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
    
    # 3. Find max product
    products = pow_part[10000:N+1] * pow_part[10001:N+2]
    max_val = np.max(products)
    best_n = 10000 + np.argmax(products)
    
    return int(best_n)
