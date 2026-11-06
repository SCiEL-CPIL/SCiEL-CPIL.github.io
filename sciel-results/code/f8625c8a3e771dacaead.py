import time
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 80  # Safety margin for 90s limit
    
    def elapsed():
        return time.time() - t0
    
    MAXN = 1_000_000
    
    # 1. Sieve of Eratosthenes to get primes
    is_prime = bytearray(b'\x01') * (MAXN + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(MAXN**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    # We only need primes p such that p^2 <= MAXN
    # p <= sqrt(MAXN)
    max_prime = int(MAXN**0.5)
    primes = [i for i in range(2, max_prime + 1) if is_prime[i]]
    
    # 2. Compute powerful part P(m) for all m using NumPy vectorization
    # P(m) = product of p^k where k = v_p(m) and k >= 2
    pp = np.ones(MAXN + 2, dtype=np.int64)
    
    for p in primes:
        if elapsed() > DEADLINE:
            break
            
        p_power = p * p
        
        while p_power <= MAXN:
            if elapsed() > DEADLINE:
                break
            
            next_power = p_power * p
            
            # Get indices that are multiples of p_power
            indices = np.arange(p_power, MAXN + 1, p_power)
            
            # Filter out those that are also multiples of next_power
            # m % next_power != 0  <=>  (m // p_power) % p != 0
            # Since m = p_power * j, m % next_power == 0 iff j % p == 0
            j_vals = indices // p_power
            mask = (j_vals % p) != 0
            
            if mask.any():
                pp[indices[mask]] *= p_power
            
            p_power = next_power
            
    # 3. Find max pp[n] * pp[n+1]
    best_val = 0
    best_n = 10000
    
    if elapsed() > DEADLINE:
        pass

    # Scan using NumPy for speed
    if elapsed() < DEADLINE:
        # Vectorized product calculation
        # n ranges from 10000 to MAXN-1
        # pp[n] * pp[n+1]
        start = 10000
        end = MAXN
        if start < end:
            prods = pp[start:end] * pp[start+1:end+1]
            max_idx = np.argmax(prods)
            best_val = prods[max_idx]
            best_n = start + max_idx
            
    return best_n
