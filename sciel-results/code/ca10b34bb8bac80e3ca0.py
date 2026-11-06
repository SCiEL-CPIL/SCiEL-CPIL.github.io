import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 58  # Safety margin before 60s kill
    
    N = 1000000
    MAX_VAL = N + 1
    
    # Sieve primes up to sqrt(MAX_VAL)
    limit = int(math.isqrt(MAX_VAL))
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # Initialize powerful part array P with 1s
    # We need P[0]..P[MAX_VAL]
    P = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    # Compute powerful parts using vectorized operations
    for p in primes:
        if time.time() - start_time > deadline - 5:
            break
            
        p2 = p * p
        if p2 > MAX_VAL:
            break
            
        power = p2
        while power <= MAX_VAL:
            if time.time() - start_time > deadline - 5:
                break
                
            next_power = power * p
            
            # Indices are 0-indexed, but we care about values 1..MAX_VAL
            # Multiples of 'power' in [1, MAX_VAL]
            # Slice: power, 2*power, ...
            # Note: np.arange creates a new array, slicing P creates a view.
            # We need to be careful with the mask.
            
            if next_power > MAX_VAL:
                # All multiples of power in range have exactly this exponent
                # Multiples: power, 2*power, ...
                # Indices: power, power+power, ...
                # P[power::power] corresponds to P[power], P[2*power], ...
                P[power::power] *= power
            else:
                # Multiples of power but not next_power
                # First get all multiples of power
                multiples = np.arange(power, MAX_VAL + 1, power)
                
                # Mask for those not divisible by next_power
                # multiples % next_power != 0
                mask = (multiples % next_power) != 0
                
                # Apply multiplication to valid indices
                valid_indices = multiples[mask]
                P[valid_indices] *= power
                
            power = next_power
    
    # Find best n in [10000, 1000000]
    # We want to maximize P[n] * P[n+1]
    # n ranges from 10000 to 1000000
    # P[n] is at index n
    # P[n+1] is at index n+1
    
    if time.time() - start_time > deadline - 2:
        return 10000

    # Vectorized product
    # n starts at 10000, ends at 1000000
    # P[10000:1000001] and P[10001:1000002]
    products = P[10000:1000001] * P[10001:1000002]
    
    # Find index of max
    max_idx = np.argmax(products)
    best_n = 10000 + max_idx
    
    return int(best_n)
