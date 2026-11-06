import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s margin, evaluator kills at 90s
    
    N_MAX = 1000000
    N_LIMIT = N_MAX + 1  # We need n+1 up to 1000001
    
    # 1. Sieve of Eratosthenes to find primes up to N_LIMIT
    is_prime = [True] * (N_LIMIT + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(N_LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N_LIMIT + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N_LIMIT + 1) if is_prime[i]]
    
    # 2. Precompute powerful parts
    # val[x] will store P(x)
    val = [1] * (N_LIMIT + 1)
    
    for p in primes:
        p2 = p * p
        if p2 > N_LIMIT:
            break
        
        # Iterate through powers p^k for k >= 2
        power = p2
        next_power = power * p
        
        while power <= N_LIMIT:
            # For each multiple m of 'power'
            # Check if exponent of p in m is exactly log_p(power)
            # i.e., m is divisible by power but not by next_power
            
            if next_power > N_LIMIT:
                for m in range(power, N_LIMIT + 1, power):
                    val[m] *= power
            else:
                for m in range(power, N_LIMIT + 1, power):
                    if m % next_power != 0:
                        val[m] *= power
            
            power = next_power
            next_power *= p
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break

    # 3. Find best n
    best_score = -1
    best_n = 10000
    
    # Range [10000, 1000000]
    for n in range(10000, N_MAX + 1):
        score = val[n] * val[n + 1]
        if score > best_score:
            best_score = score
            best_n = n
            
        if time.time() > deadline:
            break

    return best_n
