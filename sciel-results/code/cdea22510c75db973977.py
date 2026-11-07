import time
import math
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    max_m = 10**18
    
    # Linear sieve for sigma(n) up to L
    L = 10**7
    
    sigma = [0] * (L + 1)
    lp = [0] * (L + 1)  # least prime factor
    spow = [1] * (L + 1)  # highest power of lp[i] dividing i
    primes = []
    
    sigma[1] = 1
    lp[1] = 1
    spow[1] = 1
    
    for i in range(2, L + 1):
        if lp[i] == 0:
            lp[i] = i
            primes.append(i)
            sigma[i] = 1 + i
            spow[i] = i
        for p in primes:
            if p > lp[i] or i * p > L:
                break
            lp[i * p] = p
            if i % p == 0:
                spow[i * p] = spow[i] * p
                m_part = i // spow[i]
                sigma[i * p] = sigma[i] + spow[i] * p * sigma[m_part]
            else:
                spow[i * p] = p
                sigma[i * p] = sigma[i] * (1 + p)
    
    # Map M = k * sigma(k) to list of k
    solutions = defaultdict(list)
    
    for k in range(1, L + 1):
        m_val = k * sigma[k]
        if m_val <= max_m:
            solutions[m_val].append(k)
    
    # Sort candidates by number of solutions
    candidates = sorted(solutions.items(), key=lambda x: -len(x[1]))
    
    best_m = 0
    best_k_list = []
    
    # Helper: compute sigma(n) by trial division for n > L
    def _sigma_trial(n):
        if n <= 0:
            return 0
        result = 1
        d = 2
        while d * d <= n:
            if n % d == 0:
                power_sum = 1
                power = 1
                while n % d == 0:
                    n //= d
                    power *= d
                    power_sum += power
                result *= power_sum
            d += 1 if d == 2 else 2
        if n > 1:
            result *= (1 + n)
        return result
    
    # Helper: get all divisors of n
    def _divisors(n):
        divs = [1]
        d = 2
        while d * d <= n:
            if n % d == 0:
                exp = 0
                while n % d == 0:
                    n //= d
                    exp += 1
                new_divs = []
                pow_d = 1
                for _ in range(exp + 1):
                    for dv in divs:
                        new_divs.append(dv * pow_d)
                    pow_d *= d
                divs = new_divs
            d += 1 if d == 2 else 2
        if n > 1:
            new_divs = []
            for dv in divs:
                new_divs.append(dv)
                new_divs.append(dv * n)
            divs = new_divs
        return divs
    
    # For each top candidate, try to find additional k values by checking all divisors
    for m_val, k_list in candidates[:200]:
        if time.time() > deadline - 5:
            break
        
        if len(k_list) <= len(best_k_list) and len(k_list) < 4:
            break
        
        sqrt_m = math.isqrt(m_val)
        
        divs = _divisors(m_val)
        
        existing = set(k_list)
        found_new = False
        
        for k in divs:
            if k in existing or k > sqrt_m or k < 1:
                continue
            if k <= L:
                sk = sigma[k]
            else:
                sk = _sigma_trial(k)
            if k * sk == m_val:
                existing.add(k)
                k_list.append(k)
                found_new = True
        
        if len(k_list) > len(best_k_list):
            best_m = m_val
            best_k_list = list(k_list)
    
    # NEW: Multiplicative structure search - directly construct M with high collision density
    # For k = prod p_i^e_i, sigma(k) = prod (p_i^(e_i+1)-1)/(p_i-1)
    # M = k * sigma(k) = prod p_i^e_i * prod (p_i^(e_i+1)-1)/(p_i-1)
    # We want many different exponent vectors giving the same M
    
    # Strategy: Use small primes and search for M values that have many representations
    # Focus on primes where (p^(e+1)-1)/(p-1) has nice factorization
    
    # Precompute useful values: for prime p and exponent e, compute p^e and sigma_p(e) = 1+p+...+p^e
    # Then M = prod p_i^e_i * prod sigma_p_i(e_i)
    
    # Use a smaller set of primes for the multiplicative search to keep it tractable
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    # For each prime, precompute p^e and sigma_p(e) for e=0,1,2,...
    # sigma_p(e) = (p^(e+1)-1)/(p-1)
    
    # We'll use a meet-in-the-middle approach:
    # Split primes into two groups, enumerate partial products for each, then combine
    
    # Group 1: first half of primes
    # Group 2: second half of primes
    
    # For each group, enumerate all combinations of exponents (with bounded exponents)
    # and compute the partial M value and partial k value
    
    # To keep this tractable, we'll limit the number of primes per group and the max exponent
    
    # Let's try a different approach: enumerate k values of the form prod p_i^e_i
    # where p_i are from a small set, and check if M = k*sigma(k) has many preimages
    
    # Actually, let's be more systematic. We'll enumerate M values directly by their
    # prime factorization structure.
    
    # Key insight: If M = prod q_j^f_j, then for k to satisfy k*sigma(k) = M,
    # k must divide M, and sigma(k) = M/k must also be an integer.
    
    # So we can enumerate divisors k of M and check if k*sigma(k) = M.
    
    # The question is: which M have many such k?
    
    # Let's try constructing M from scratch using the multiplicative structure.
    
    # For a given set of primes P = {p_1, ..., p_n} and exponents e_i,
    # k = prod p_i^e_i
    # sigma(k) = prod (1 + p_i + ... + p_i^e_i)
    # M = k * sigma(k)
    
    # We want to find M that can be written this way in many different ways.
    
    # Let's use a search over exponent vectors for a small set of primes.
    
    # Use primes: 2, 3, 5, 7, 11, 13
    search_primes = [2, 3, 5, 7, 11, 13]
    n_sp = len(search_primes)
    
    # For each prime p, precompute p^e and sigma_p(e) for e = 0, 1, 2, ..., max_e
    # max_e is chosen so that p^max_e * sigma_p(max_e) <= max_m
    
    prime_data = {}
    for p in search_primes:
        powers = []  # list of (p^e, sigma_p(e))
        pe = 1
        sigma_pe = 1
        e = 0
        while True:
            # Check if adding this prime factor would exceed max_m
            # M contribution from this prime: p^e * sigma_p(e)
            contrib = pe * sigma_pe
            if contrib > max_m:
                break
            powers.append((pe, sigma_pe))
            e += 1
            pe *= p
            sigma_pe = sigma_pe + pe  # sigma_p(e+1) = sigma_p(e) + p^(e+1)
            if e > 60:  # safety
                break
        prime_data[p] = powers
    
    # Now enumerate all combinations of exponents
    # For 6 primes, if each has ~5-10 choices, that's 5^6 to 10^6 = 15625 to 1M combinations
    # This is feasible
    
    # We'll use recursion to enumerate
    m_to_ks = defaultdict(list)
    
    def _enumerate(idx, current_k, current_m):
        if time.time() > deadline - 10:
            return
        if idx == n_sp:
            if current_m <= max_m and current_m > 0:
                m_to_ks[current_m].append(current_k)
            return
        
        p = search_primes[idx]
        for pe, sigma_pe in prime_data[p]:
            new_k = current_k * pe
            new_m = current_m * pe * sigma_pe
            if new_m > max_m:
                break
            _enumerate(idx + 1, new_k, new_m)
    
    _enumerate(0, 1, 1)
    
    # Find the M with the most k values
    if m_to_ks:
        best_m_mult = max(m_to_ks.items(), key=lambda x: len(x[1]))
        if len(best_m_mult[1]) > len(best_k_list):
            best_m = best_m_mult[0]
            best_k_list = list(best_m_mult[1])
    
    # Fallback if no solutions found
    if not best_k_list:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Ensure k_i <= sqrt(M) and distinct
    sqrt_m = math.isqrt(best_m)
    valid_k = sorted(set(k for k in best_k_list if k <= sqrt_m and k >= 1))
    
    # Verify all solutions are correct
    verified_k = []
    for k in valid_k:
        if k <= L:
            sk = sigma[k]
        else:
            sk = _sigma_trial(k)
        if k * sk == best_m:
            verified_k.append(k)
    
    M_out = int(best_m)
    k_out = [int(k) for k in verified_k]
    
    return {'M': M_out, 'k': k_out}

if __name__ == "__main__":
    result = construct()
    print(result)
