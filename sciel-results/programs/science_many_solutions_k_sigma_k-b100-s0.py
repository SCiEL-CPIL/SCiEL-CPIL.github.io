import time
import math
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit

    # More primes to increase collision chances
    # Generate primes up to ~200
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]
    
    primes = sieve(200)
    
    max_m = 10**18
    sqrt_max_m = math.isqrt(max_m)
    
    solutions = defaultdict(list)
    time_check_counter = [0]
    
    def check_time():
        time_check_counter[0] += 1
        if time_check_counter[0] % 10000 == 0:
            if time.time() > deadline:
                return True
        return False
    
    def gen_smooth(idx, current_k, current_sigma):
        if check_time():
            return

        if idx == len(primes):
            if current_k > 1:
                m_val = current_k * current_sigma
                if m_val <= max_m:
                    solutions[m_val].append(current_k)
            return

        p = primes[idx]
        
        # Option 1: Do not use prime p (exponent 0)
        gen_smooth(idx + 1, current_k, current_sigma)
        
        # Option 2: Use prime p with exponent e >= 1
        k_curr = current_k
        sigma_curr = current_sigma
        sum_powers = 1
        p_power = 1
        
        while True:
            if k_curr > sqrt_max_m // p:
                break
            
            k_curr *= p
            p_power *= p
            sum_powers += p_power
            
            if sum_powers > 0 and sigma_curr > max_m // sum_powers:
                break
                
            sigma_curr_new = sigma_curr * sum_powers
            k_curr_new = k_curr
            
            if k_curr_new * sigma_curr_new > max_m:
                break
                
            gen_smooth(idx + 1, k_curr_new, sigma_curr_new)

    gen_smooth(0, 1, 1)
    
    best_m = 0
    best_k_list = []
    
    for m, k_list in solutions.items():
        if len(k_list) > len(best_k_list):
            best_m = m
            best_k_list = k_list
            
    if not best_k_list:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    distinct_k = list(set(best_k_list))
    sqrt_m = math.isqrt(best_m)
    valid_k = [k for k in distinct_k if k <= sqrt_m]
    valid_k.sort()
    
    M_out = int(best_m)
    k_out = [int(k) for k in valid_k]
    
    return {'M': M_out, 'k': k_out}

if __name__ == "__main__":
    result = construct()
    print(result)
