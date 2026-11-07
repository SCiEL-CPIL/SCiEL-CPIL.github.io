import time
import math
import sys
from collections import defaultdict, Counter

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit
    
    max_m = 10**18
    
    # Fast primality test for small numbers
    def is_prime_small(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    # Pollard Rho for factoring
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
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            while n % p == 0:
                factors.append(p)
                n //= p
        if n == 1:
            return factors
        if is_prime_small(n):
            factors.append(n)
        else:
            stack = [n]
            while stack:
                x = stack.pop()
                if x == 1: continue
                if is_prime_small(x):
                    factors.append(x)
                else:
                    d = pollard_rho(x)
                    stack.append(d)
                    stack.append(x // d)
        return sorted(factors)
    
    def sigma_from_factorization(factors):
        """Compute sigma(n) from list of prime factors (with multiplicity)."""
        c = Counter(factors)
        s = 1
        for p, e in c.items():
            term = (p**(e+1) - 1) // (p - 1)
            s *= term
        return s
    
    def generate_divisors_from_factors(factors):
        """Generate all divisors from a list of prime factors (with multiplicity)."""
        c = Counter(factors)
        primes_list = sorted(c.keys())
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
    
    def find_all_k_for_M(M):
        """Find all k such that k * sigma(k) = M, with k <= sqrt(M)."""
        if M < 2:
            return []
        
        factors = factorize(M)
        divs = generate_divisors_from_factors(factors)
        
        # Pre-compute the prime factors of M for efficient sigma computation
        m_primes = sorted(set(factors))
        
        valid_ks = []
        sqrt_m = math.isqrt(M)
        
        for d in divs:
            if d > sqrt_m:
                continue
            if d < 1:
                continue
            
            # Compute sigma(d) using the prime factors of M
            temp_d = d
            sigma_d = 1
            for p in m_primes:
                if temp_d % p == 0:
                    e = 0
                    while temp_d % p == 0:
                        temp_d //= p
                        e += 1
                    sigma_d *= (p**(e+1) - 1) // (p - 1)
            if temp_d > 1:
                # This shouldn't happen since d divides M
                # But just in case, factorize temp_d
                temp_factors = factorize(temp_d)
                temp_c = Counter(temp_factors)
                for p, e in temp_c.items():
                    sigma_d *= (p**(e+1) - 1) // (p - 1)
            
            if d * sigma_d == M:
                valid_ks.append(d)
        
        return valid_ks
    
    # Strategy: Generate candidate M values by enumerating smooth k values
    # and counting how many map to each M. Then find all k for the best M.
    
    # Use primes up to a reasonable limit
    primes = []
    p = 2
    while len(primes) < 20:
        if is_prime_small(p):
            primes.append(p)
        p += 1
    
    # Generate smooth numbers and their sigma values
    # k = prod p_i^e_i, sigma(k) = prod (p_i^(e_i+1)-1)/(p_i-1)
    m_counts = defaultdict(int)
    m_examples = defaultdict(list)
    
    def gen_smooth(idx, current_k, current_sigma):
        if time.time() > deadline - 15:
            return
        
        if idx == len(primes):
            if current_k > 1:
                m_val = current_k * current_sigma
                if m_val <= max_m:
                    m_counts[m_val] += 1
                    if len(m_examples[m_val]) < 3:
                        m_examples[m_val].append(current_k)
            return
        
        p = primes[idx]
        
        # Option: exponent 0
        gen_smooth(idx + 1, current_k, current_sigma)
        
        # Option: exponent e >= 1
        k_curr = current_k
        sum_powers = 1
        p_power = 1
        
        while True:
            p_power *= p
            k_curr *= p
            sum_powers += p_power
            
            sigma_curr_new = current_sigma * sum_powers
            if k_curr * sigma_curr_new > max_m:
                break
            
            gen_smooth(idx + 1, k_curr, sigma_curr_new)
    
    gen_smooth(0, 1, 1)
    
    if not m_counts:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Sort by count, take top candidates
    sorted_ms = sorted(m_counts.items(), key=lambda item: item[1], reverse=True)
    
    # Try top candidates to find all k's
    best_m = 0
    best_k_list = []
    
    # Try top 10 candidates
    num_candidates = min(10, len(sorted_ms))
    
    for i in range(num_candidates):
        if time.time() > deadline - 10:
            break
        
        M, count = sorted_ms[i]
        
        try:
            valid_ks = find_all_k_for_M(M)
            if len(valid_ks) > len(best_k_list):
                best_k_list = valid_ks
                best_m = M
        except Exception:
            continue
    
    # If we didn't find anything better, use the smooth examples
    if not best_k_list:
        best_m, count = sorted_ms[0]
        best_k_list = m_examples[best_m]
    
    # Final validation
    distinct_k = list(set(best_k_list))
    sqrt_m = math.isqrt(best_m)
    valid_k = [k for k in distinct_k if 1 <= k <= sqrt_m]
    valid_k.sort()
    
    # Ensure types are correct
    M_out = int(best_m)
    k_out = [int(k) for k in valid_k]
    
    return {'M': M_out, 'k': k_out}

if __name__ == "__main__":
    result = construct()
    print(result)
