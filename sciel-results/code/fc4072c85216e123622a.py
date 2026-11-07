import time
import math
import sys
from collections import defaultdict, Counter

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    def check_time():
        return time.time() > deadline

    def sieve(n):
        is_prime = [True] * (n + 1)
        if n >= 0:
            is_prime[0] = False
        if n >= 1:
            is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    # Primes for generating candidate M
    primes_small = sieve(100)
    primes_large = sieve(500)
    
    max_m = 10**18
    
    # Dictionary to count how many k's map to each M
    m_counts = defaultdict(int)
    m_examples = defaultdict(list)
    
    # Strategy 1: Generate M by iterating over multiplicative structures
    # For a given M = prod p_i^a_i, find all k = prod p_i^e_i such that k*sigma(k) = M
    # This means: prod p_i^(e_i) * prod (p_i^(e_i+1)-1)/(p_i-1) = prod p_i^a_i
    # For each prime p_i in M, we need e_i such that the contribution matches
    
    # Alternative approach: enumerate k values and count M = k*sigma(k)
    # Use a smarter enumeration with early termination
    
    def gen_smooth_k(idx, current_k, current_sigma, primes_list):
        """Recursively generate smooth k values and count their M = k*sigma(k)"""
        if check_time():
            return
        
        if idx == len(primes_list):
            if current_k > 1:
                m_val = current_k * current_sigma
                if m_val <= max_m:
                    m_counts[m_val] += 1
                    if len(m_examples[m_val]) < 10:
                        m_examples[m_val].append(current_k)
            return

        p = primes_list[idx]
        
        # Option: exponent 0 for this prime
        gen_smooth_k(idx + 1, current_k, current_sigma, primes_list)
        
        # Option: exponent e >= 1
        k_curr = current_k
        sigma_curr = current_sigma
        sum_powers = 1
        p_power = 1
        
        while True:
            p_power *= p
            k_curr *= p
            sum_powers += p_power
            
            sigma_curr_new = current_sigma * sum_powers
            m_val = k_curr * sigma_curr_new
            
            if m_val > max_m:
                break
            
            gen_smooth_k(idx + 1, k_curr, sigma_curr_new, primes_list)
    
    # Use different prime sets to explore different M structures
    # First pass: small primes (high collision potential)
    gen_smooth_k(0, 1, 1, primes_small)
    
    # Second pass: slightly larger primes for different M structures
    if not check_time():
        gen_smooth_k(0, 1, 1, primes_large[:30])
    
    if not m_counts:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Sort candidates by count
    sorted_ms = sorted(m_counts.items(), key=lambda item: item[1], reverse=True)
    
    # Helper functions for factorization
    def is_prime(n):
        if n < 2: return False
        if n in [2, 3]: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def pollard_rho(n):
        if n % 2 == 0: return 2
        if n % 3 == 0: return 3
        while True:
            x = 2
            y = 2
            d = 1
            c = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d
            c += 1

    def factorize(n):
        factors = []
        # Trial division for small factors
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            while n % p == 0:
                factors.append(p)
                n //= p
        if n == 1:
            return factors
        if is_prime(n):
            factors.append(n)
        else:
            stack = [n]
            while stack:
                x = stack.pop()
                if x == 1: continue
                if is_prime(x):
                    factors.append(x)
                else:
                    d = pollard_rho(x)
                    stack.append(d)
                    stack.append(x // d)
        return sorted(factors)

    def generate_divisors_from_factors(factors):
        c = Counter(factors)
        primes_list = list(c.keys())
        exponents_list = [c[p] for p in primes_list]
        
        divs = [1]
        for i in range(len(primes_list)):
            p = primes_list[i]
            e = exponents_list[i]
            current_divs = []
            power = 1
            for k in range(e + 1):
                for d in divs:
                    current_divs.append(d * power)
                power *= p
            divs = current_divs
        return divs

    def sigma_from_prime_exponents(prime_exp_dict):
        s = 1
        for p, e in prime_exp_dict.items():
            if e > 0:
                term = (p**(e+1) - 1) // (p - 1)
                s *= term
        return s

    # For each top candidate, find ALL valid k values
    best_m = 0
    best_k_list = []
    
    # Check top candidates (limit to avoid timeout)
    num_candidates_to_check = min(30, len(sorted_ms))
    
    for idx in range(num_candidates_to_check):
        if check_time():
            break
            
        M, count = sorted_ms[idx]
        
        try:
            factors = factorize(M)
            if not factors:
                continue
                
            divs = generate_divisors_from_factors(factors)
            
            valid_ks = []
            m_primes_set = set(factors)
            
            for d in divs:
                if d > math.isqrt(M):
                    continue
                    
                # Calculate sigma(d) using prime factorization derived from M
                temp_d = d
                sigma_d = 1
                for p in m_primes_set:
                    if temp_d % p == 0:
                        e = 0
                        while temp_d % p == 0:
                            temp_d //= p
                            e += 1
                        sigma_d *= (p**(e+1) - 1) // (p - 1)
                
                if temp_d != 1:
                    # d has a prime factor not in M - shouldn't happen for divisors
                    continue
                
                if d * sigma_d == M:
                    valid_ks.append(d)
            
            if len(valid_ks) > len(best_k_list):
                best_k_list = sorted(set(valid_ks))
                best_m = M
                
        except Exception as e:
            continue
    
    # If we found solutions, use them
    if best_k_list:
        distinct_k = list(set(best_k_list))
        sqrt_m = math.isqrt(best_m)
        valid_k = [k for k in distinct_k if k <= sqrt_m]
        valid_k.sort()
        
        if valid_k:
            M_out = int(best_m)
            k_out = [int(k) for k in valid_k]
            return {'M': M_out, 'k': k_out}
    
    # Fallback: use the smooth k examples we collected
    if sorted_ms:
        best_m, best_count = sorted_ms[0]
        examples = m_examples.get(best_m, [])
        
        if examples:
            distinct_k = list(set(examples))
            sqrt_m = math.isqrt(best_m)
            valid_k = [k for k in distinct_k if k <= sqrt_m and k > 0]
            valid_k.sort()
            
            if valid_k:
                M_out = int(best_m)
                k_out = [int(k) for k in valid_k]
                return {'M': M_out, 'k': k_out}
    
    # Final fallback
    return {'M': 333312, 'k': [336, 372, 434]}

if __name__ == "__main__":
    result = construct()
    print(result)
