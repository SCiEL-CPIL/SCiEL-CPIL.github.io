import sys
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 1000000
    MAX_VAL = N + 1  # We need powerful_part for n and n+1, so up to 1000001
    
    # 1. Sieve of Eratosthenes to find primes
    # We only need primes up to MAX_VAL
    is_prime = [True] * (MAX_VAL + 1)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    
    for i in range(2, MAX_VAL + 1):
        if is_prime[i]:
            primes.append(i)
            if i * i <= MAX_VAL:
                for j in range(i * i, MAX_VAL + 1, i):
                    is_prime[j] = False
    
    # 2. Compute powerful part for each number 1..MAX_VAL using sieve accumulation
    powerful_part = [1] * (MAX_VAL + 1)
    
    for p in primes:
        power = p * p  # p^2
        while power <= MAX_VAL:
            next_power = power * p
            
            # Iterate through multiples of 'power'
            # If m is a multiple of power but NOT next_power, 
            # then the exponent of p in m is exactly the current exponent k.
            # We multiply powerful_part[m] by power (which is p^k).
            
            if next_power > MAX_VAL:
                # All multiples of power have exponent exactly k (since k+1 exceeds range)
                for m in range(power, MAX_VAL + 1, power):
                    powerful_part[m] *= power
            else:
                for m in range(power, MAX_VAL + 1, power):
                    if m % next_power != 0:
                        powerful_part[m] *= power
            
            power = next_power
            
            # Time check
            if time.time() > deadline - 5.0:
                # If we run out of time, we still have a partially filled array?
                # No, this is bad. We need to finish the sieve to get correct values.
                # The sieve is fast, this should not trigger.
                # But if it does, we might return garbage. 
                # Given N=1e6, the sieve is very fast (<1s).
                pass
    
    # 3. Find n in [10000, 1000000] maximizing pow_part[n] * pow_part[n+1]
    max_val = 0
    best_n = 10000
    
    for n in range(10000, N + 1):
        val = powerful_part[n] * powerful_part[n + 1]
        if val > max_val:
            max_val = val
            best_n = n
        
        # Time check for the final loop
        if time.time() > deadline:
            break
            
    return best_n
