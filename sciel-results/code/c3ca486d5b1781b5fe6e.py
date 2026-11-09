import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # powerful_part[i] = product of all p^e (e>=2) dividing i
    # Use sieve: for each prime p, for each e>=2, p^e divides multiples of p^e
    powerful = [1] * (N_MAX + 2)
    
    # Sieve for primes and their powers
    # For each prime p, compute p^2, p^3, ... up to N_MAX
    # For each such power q = p^e, multiply q into all multiples of q
    
    # First, mark composites and find primes
    is_prime = [True] * (N_MAX + 2)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, N_MAX + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, N_MAX + 1, i):
                is_prime[j] = False
    
    # For each prime, generate all powers p^e (e>=2) and update multiples
    for p in primes:
        power = p * p  # p^2
        while power <= N_MAX:
            # Multiply power into all multiples of power
            for m in range(power, N_MAX + 1, power):
                powerful[m] *= power
            # Next power
            if power > N_MAX // p:
                break
            power *= p
    
    # Now find the best n in [N_MIN, N_MAX-1] maximizing powerful[n] * powerful[n+1]
    best_n = N_MIN
    best_product = powerful[N_MIN] * powerful[N_MIN + 1]
    
    # Scan through the range
    for n in range(N_MIN, N_MAX):
        prod = powerful[n] * powerful[n + 1]
        if prod > best_product:
            best_product = prod
            best_n = n
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    return best_n
# EVOLVE-BLOCK-END
