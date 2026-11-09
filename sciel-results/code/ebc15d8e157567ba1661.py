"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from collections import defaultdict

# --- Fast Factorization Utilities ---

def _is_prime(n):
    """Deterministic Miller-Rabin for n < 3.2 * 10^18"""
    if n < 2:
        return False
    # Small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n % p == 0:
            return n == p
    
    # Write n-1 as d * 2^r
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
        
    # Bases sufficient for n < 3.2 * 10^18
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def _pollard_rho(n):
    """Pollard's Rho factorization"""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    
    while True:
        x = random.randint(2, n - 1)
        y = x
        c = random.randint(1, n - 1)
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d

def _factorize(n):
    """Factorize n into prime factors. Returns dict {prime: exponent}."""
    if n <= 1:
        return {}
    factors = defaultdict(int)
    def _factor(n, factors):
        if n == 1:
            return
        if _is_prime(n):
            factors[n] += 1
            return
        d = _pollard_rho(n)
        _factor(d, factors)
        _factor(n // d, factors)
        
    _factor(n, factors)
    return dict(factors)

def _sigma_from_factors(factors):
    """Compute sigma(n) from its factorization dict."""
    result = 1
    for p, exp in factors.items():
        # sum_{i=0}^{exp} p^i
        # Using geometric series formula
        result *= (p**(exp + 1) - 1) // (p - 1)
    return result

def _sigma(n):
    """Compute sigma(n) - sum of divisors of n."""
    if n == 1:
        return 1
    f = _factorize(n)
    return _sigma_from_factors(f)

def _find_k_for_M(M, deadline):
    """Find all k such that k * sigma(k) = M and k <= sqrt(M)."""
    if M < 1:
        return []
    
    max_k = math.isqrt(M)
    if max_k < 1:
        return []
    
    M_factors = _factorize(M)
    if not M_factors:
        return []
    
    primes = list(M_factors.keys())
    max_exps = [M_factors[p] for p in primes]
    
    k_values = []
    
    # We generate divisors k of M.
    # k = prod p_i^e_i where 0 <= e_i <= max_exps[i]
    # We need k <= max_k.
    # We also need to compute sigma(k) efficiently.
    # sigma(k) = prod (p_i^(e_i+1) - 1) / (p_i - 1)
    
    # Precompute sigma terms for each prime power
    # sigma_terms[i][e] = (p_i^(e+1) - 1) / (p_i - 1)
    sigma_terms = []
    for i, p in enumerate(primes):
        terms = [1]
        curr = 1
        for e in range(1, max_exps[i] + 1):
            curr *= p
            terms.append((curr * p - 1) // (p - 1)) # Wait, formula is (p^(e+1)-1)/(p-1)
            # Let's just compute directly to be safe and simple, exps are small
            # Actually (p^(e+1)-1)/(p-1) is sum_{j=0}^e p^j.
            # Let's recompute properly.
        sigma_terms.append(terms)
        
    # Recalculate sigma_terms correctly
    sigma_terms = []
    for i, p in enumerate(primes):
        terms = [1] # e=0 -> sigma(1)=1? No, term for p^0 in product is 1.
        # The factor in sigma(k) for prime p with exponent e is 1 + p + ... + p^e.
        s = 1
        terms.append(s) # e=0
        for e in range(1, max_exps[i] + 1):
            s += p**e
            terms.append(s)
        sigma_terms.append(terms)
        
    def _generate(idx, current_k, current_sigma_k):
        if time.time() > deadline:
            return
        if idx == len(primes):
            if current_k <= max_k and current_k * current_sigma_k == M:
                k_values.append(current_k)
            return
        
        p = primes[idx]
        max_e = max_exps[idx]
        
        for e in range(max_e + 1):
            new_k = current_k * (p ** e)
            if new_k > max_k:
                # Since p >= 2, further increasing e will only increase new_k
                # We can break if we iterate e in increasing order
                # But we also have other primes. If new_k > max_k, we can't fix it by multiplying by 1 for others.
                # So we can break the loop for this e and higher e?
                # Yes, p^e is increasing.
                break
            
            new_sigma_k = current_sigma_k * sigma_terms[idx][e]
            
            # Optimization: if new_sigma_k > M // new_k, then new_k * new_sigma_k > M.
            # Since we need equality, if product > M, it's invalid.
            # But we don't know the final product yet?
            # Actually, sigma(k) is multiplicative. The final sigma is current_sigma_k * (product of remaining terms).
            # The remaining terms are at least 1.
            # So min possible final sigma is current_sigma_k.
            # If new_k * current_sigma_k > M, then even with min sigma, product > M.
            # Wait, sigma(k) >= k. So k*sigma(k) >= k^2.
            # We already check k <= sqrt(M).
            
            _generate(idx + 1, new_k, new_sigma_k)
            
    _generate(0, 1, 1)
    return k_values

def _generate_candidate_Ms(max_primes=10, max_exp=8, deadline=None):
    """Generate candidate M values from products of small primes with varying exponents."""
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    candidates = set()
    
    def _gen(idx, current_M, current_k):
        if deadline is not None and time.time() > deadline:
            return
        if idx >= len(small_primes):
            if 1 < current_M <= 10**18:
                candidates.add(current_M)
            return
        
        p = small_primes[idx]
        # Try exponent 0 (skip this prime) and exponents 1..max_exp
        for exp in range(0, max_exp + 1):
            if exp == 0:
                _gen(idx + 1, current_M, current_k)
            else:
                sigma_p = (p ** (exp + 1) - 1) // (p - 1)
                new_M = current_M * sigma_p
                if new_M > 10**18:
                    break
                new_k = current_k * (p ** exp)
                _gen(idx + 1, new_M, new_k)
    
    _gen(0, 1, 1)
    
    return list(candidates)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin (10s buffer for 120s limit)
    
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # Generate candidate M values
    # We want M with many divisors to increase chance of multiple k solutions.
    # The generator produces M = prod sigma(p^a).
    # Let's limit the search space to avoid timeout.
    candidates = _generate_candidate_Ms(max_primes=8, max_exp=6, deadline=deadline)
    
    # Sort candidates: prefer those with more prime factors or larger value?
    # Heuristic: M with many small prime factors in its factorization might have more divisors k.
    # But factorizing all candidates is expensive.
    # Let's just iterate. The number of candidates is limited by max_exp and max_primes.
    # 8 primes, exp 0-6. 7^8 = 5.7M candidates? No, the recursion breaks early if M > 10^18.
    # Let's estimate.
    # If we just take the first few valid ones, we might miss the best.
    # Let's filter candidates: only keep those M that are divisible by many small numbers?
    # Or just check the top N candidates by some heuristic.
    
    # Heuristic: Number of prime factors of M (with multiplicity) is a good proxy for number of divisors.
    # But calculating this requires factorization.
    # Let's factorize a subset.
    
    # To be safe, let's just process candidates in a random or sorted order until deadline.
    # Sorting by M is cheap.
    candidates.sort()
    
    # We can also try specific known good M values if we have them.
    # But the generator should find them.
    
    processed = 0
    for M in candidates:
        if time.time() > deadline:
            break
        if M > 10**18:
            continue
            
        # Quick check: if M is prime, only k=1 works (sigma(1)=1, 1*1=1 != M unless M=1).
        # If M is prime, no k > 1 works because k must divide M? No, k doesn't have to divide M.
        # Wait, k * sigma(k) = M. If M is prime, k must be 1? 1*1=1. So M=1.
        # If M > 1 is prime, no solution.
        # So we can skip primes.
        if _is_prime(M):
            continue
            
        ks = _find_k_for_M(M, deadline)
        count = len(ks)
        
        if count > best_count:
            best_M = M
            best_ks = ks
            best_count = count
            
        # Early stop if we find a very high count?
        # Unlikely to get > 20 solutions quickly.
        
    # Ensure types are correct
    best_M = int(best_M)
    best_ks = [int(k) for k in best_ks]
    
    return {'M': best_M, 'k': best_ks}
# EVOLVE-BLOCK-END
