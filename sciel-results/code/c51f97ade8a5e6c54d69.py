import time
import math
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit

    # Generate primes up to 200
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
    sqrt_max_m = math.isqrt(max_m)  # 10^9

    solutions = defaultdict(list)
    check_counter = [0]

    def check_time():
        check_counter[0] += 1
        if check_counter[0] % 5000 == 0:
            if time.time() > deadline:
                return True
        return False

    # Iterative generation of smooth numbers using state expansion
    # Start with (1, 1)
    states = [(1, 1)]
    
    for idx, p in enumerate(primes):
        new_states = []
        for (k, sigma) in states:
            # Exponent 0: don't use this prime
            new_states.append((k, sigma))
            
            # Exponent e >= 1: multiply by p^e
            k_curr = k
            sigma_curr = sigma
            sum_powers = 1
            p_power = 1
            
            for _ in range(20):
                if k_curr > sqrt_max_m // p:
                    break
                k_curr *= p
                p_power *= p
                sum_powers += p_power
                
                sigma_new = sigma_curr * sum_powers
                
                # Pruning: k * sigma must be <= max_m
                if k_curr > max_m // sigma_new:
                    break
                # Pruning: k must be <= sqrt(M), i.e., k <= sigma
                if k_curr > sigma_new:
                    break
                    
                new_states.append((k_curr, sigma_new))
        
        states = new_states
        
        if check_time():
            break
    
    # Process all final states
    for (k, sigma) in states:
        if k > 1:
            m_val = k * sigma
            if m_val <= max_m and k <= sigma:
                solutions[m_val].append(k)
    
    # Find best M
    best_m = 0
    best_k_list = []
    
    for m, k_list in solutions.items():
        if len(k_list) > len(best_k_list):
            best_m = m
            best_k_list = k_list
    
    # Post-processing: try to find more solutions for promising M values
    # For M with >= 2 solutions, try to find more by checking divisors
    if time.time() < deadline - 10:
        # Get top candidates
        candidates = sorted(solutions.items(), key=lambda x: -len(x[1]))[:50]
        
        for m, k_list in candidates:
            if len(k_list) < 2:
                continue
            if time.time() > deadline - 5:
                break
            
            # Try to find more k values that satisfy k*sigma(k) = m
            # k must be a divisor of m and k <= sqrt(m)
            sqrt_m = math.isqrt(m)
            
            # Get all divisors of m up to sqrt(m)
            # This is expensive, so only do it for very promising M
            if len(k_list) >= 3:
                # Find all divisors
                divisors = []
                for i in range(1, sqrt_m + 1):
                    if m % i == 0:
                        divisors.append(i)
                        other = m // i
                        if other != i and other <= sqrt_m:
                            divisors.append(other)
                
                # Check each divisor
                for d in divisors:
                    if d in k_list:
                        continue
                    # Check if d * sigma(d) == m
                    # Compute sigma(d)
                    n = d
                    sigma_d = 1
                    temp = d
                    for p in primes:
                        if p * p > temp:
                            break
                        if temp % p == 0:
                            power = 0
                            while temp % p == 0:
                                temp //= p
                                power += 1
                            sigma_d *= (p**(power+1) - 1) // (p - 1)
                    if temp > 1:
                        sigma_d *= (temp * 2 - 1)
                    
                    if d * sigma_d == m:
                        k_list.append(d)
    
    # Final selection
    best_m = 0
    best_k_list = []
    
    for m, k_list in solutions.items():
        if len(k_list) > len(best_k_list):
            best_m = m
            best_k_list = k_list
    
    if not best_k_list or len(best_k_list) < 3:
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
