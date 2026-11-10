"""Rewritten program: find M <= 10^18 with many distinct k <= sqrt(M) satisfying k*sigma(k) = M."""
import time
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def _isqrt(x):
    return math.isqrt(x)

def _sigma_by_factorization(n):
    """Compute sigma(n) by trial division factorization."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    result = 1
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            power = 0
            while temp % d == 0:
                temp //= d
                power += 1
            factor = (d ** (power + 1) - 1) // (d - 1)
            result *= factor
        d += 1
    if temp > 1:
        result *= (1 + temp)
    return result

def _divisors(n):
    """Return sorted list of all divisors of n."""
    if n <= 0:
        return []
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    divs.sort()
    return divs

def _linear_sigma_sieve(N):
    """Compute sigma(k) for k=1..N using a linear sieve. Returns list indexed 0..N."""
    sigma = [0] * (N + 1)
    sigma[1] = 1
    primes = []
    
    spf = [0] * (N + 1)
    p_pow = [1] * (N + 1)
    sigma_p = [1] * (1)
    rest_sig = [1] * (1)
    
    sigma[1] = 1
    p_pow[1] = 1
    sigma_p = [1] * (N + 1)
    rest_sig = [1] * (N + 1)
    sigma_p[1] = 1
    rest_sig[1] = 1
    
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
            p_pow[i] = i
            sigma_p[i] = 1 + i
            rest_sig[i] = 1
            sigma[i] = 1 + i
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            spf[i * p] = p
            if p == spf[i]:
                p_pow[i * p] = p_pow[i] * p
                sigma_p[i * p] = sigma_p[i] + p_pow[i * p]
                rest_sig[i * p] = rest_sig[i]
            else:
                p_pow[i * p] = p
                sigma_p[i * p] = 1 + p
                rest_sig[i * p] = sigma[i]
            sigma[i * p] = sigma_p[i * p] * rest_sig[i * p]
    
    return sigma

def _numpy_sigma_sieve(N):
    """Compute sigma(k) for k=1..N using NumPy vectorized approach. Returns numpy array indexed 0..N."""
    import numpy as np
    
    sigma = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    
    for d in range(1, N + 1):
        sigma[d::d] += d
    
    return sigma

def _generate_smooth_numbers(primes, limit):
    """Generate smooth numbers up to limit using DFS. Returns list of (k, sigma_k)."""
    results = []
    n = len(primes)
    
    def dfs(idx, current_k, current_sigma):
        if current_k > limit:
            return
        results.append((current_k, current_sigma))
        
        if idx >= n:
            return
            
        p = primes[idx]
        pk = current_k
        sum_p = 1
        
        while pk <= limit // p:
            pk *= p
            sum_p = sum_p * p + 1
            new_sigma = current_sigma * sum_p
            dfs(idx + 1, pk, new_sigma)
    
    dfs(0, 1, 1)
    return results

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # Phase 1: Use NumPy sieve for fast sigma computation
    import numpy as np
    
    sizes_to_try = [2000000, 5000000, 10000000]
    
    for N in sizes_to_try:
        if time.time() > deadline - 30:
            break
        
        # Compute sigma using NumPy
        sigma = _numpy_sigma_sieve(N)
        
        # Compute M_val = k * sigma[k] for all k
        k_vals = np.arange(1, N + 1, dtype=np.int64)
        M_vals = k_vals * sigma[1:N + 1]
        
        # Filter M_vals <= 10^18
        mask = M_vals <= 10**18
        valid_M_vals = M_vals[mask]
        valid_k_vals = k_vals[mask]
        
        # Find collisions: group by M_val
        m_to_k = defaultdict(list)
        for i in range(len(valid_M_vals)):
            m_to_k[int(valid_M_vals[i])].append(int(valid_k_vals[i]))
        
        # Find top candidates
        candidates = sorted(m_to_k.items(), key=lambda x: len(x[1]), reverse=True)
        top_candidates = candidates[:200]
        
        # For each top candidate, try to extend
        for M_val, ks in top_candidates:
            if time.time() > deadline - 15:
                break
            
            sqrt_M = _isqrt(M_val)
            valid_ks = [k for k in ks if k <= sqrt_M]
            current_count = len(valid_ks)
            
            # Skip if can't beat best even with generous extension
            if current_count + 10 <= best_count:
                continue
            
            # Try to extend: find divisors k of M_val with k > N, k <= sqrt_M
            divs = _divisors(M_val)
            existing = set(valid_ks)
            extended = list(valid_ks)
            
            for k in divs:
                if k <= N or k > sqrt_M or k in existing:
                    continue
                expected_sigma = M_val // k
                if k > 1 and expected_sigma < k + 1:
                    continue
                if _sigma_by_factorization(k) == expected_sigma:
                    extended.append(k)
                    existing.add(k)
            
            if len(extended) > best_count:
                best_M = M_val
                best_ks = sorted(extended)
                best_count = len(extended)
    
    # Phase 2: Smooth number DFS for high-collision M values
    if time.time() < deadline - 20:
        # Use small primes for smooth numbers
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        # Generate smooth numbers up to a reasonable limit
        # The limit should be such that k * sigma(k) <= 10^18
        # For smooth numbers, sigma(k)/k can be large, so k can be up to ~10^18 / (sigma(k)/k)
        # For highly composite smooth numbers, sigma(k)/k can be ~10-20
        # So k can be up to ~10^17
        
        # But we need to be careful about the number of smooth numbers
        # Let's start with a smaller limit and increase
        smooth_limits = [10**6, 10**7, 10**8]
        
        for limit in smooth_limits:
            if time.time() > deadline - 10:
                break
            
            # Generate smooth numbers
            smooth_nums = _generate_smooth_numbers(small_primes, limit)
            
            # Group by M = k * sigma(k)
            m_to_k_smooth = defaultdict(list)
            for k, sig_k in smooth_nums:
                M_val = k * sig_k
                if M_val <= 10**18:
                    m_to_k_smooth[M_val].append(k)
            
            # Find top candidates
            candidates = sorted(m_to_k_smooth.items(), key=lambda x: len(x[1]), reverse=True)
            top_candidates = candidates[:100]
            
            # For each top candidate, verify and extend
            for M_val, ks in top_candidates:
                if time.time() > deadline - 5:
                    break
                
                sqrt_M = _isqrt(M_val)
                valid_ks = [k for k in ks if k <= sqrt_M]
                current_count = len(valid_ks)
                
                # Skip if can't beat best
                if current_count + 5 <= best_count:
                    continue
                
                # Try to extend with other divisors
                divs = _divisors(M_val)
                existing = set(valid_ks)
                extended = list(valid_ks)
                
                for k in divs:
                    if k > sqrt_M or k in existing:
                        continue
                    expected_sigma = M_val // k
                    if k > 1 and expected_sigma < k + 1:
                        continue
                    if _sigma_by_factorization(k) == expected_sigma:
                        extended.append(k)
                        existing.add(k)
                
                if len(extended) > best_count:
                    best_M = M_val
                    best_ks = sorted(extended)
                    best_count = len(extended)
    
    # Ensure types are correct
    best_M = int(best_M)
    best_ks = [int(k) for k in best_ks]
    
    # Final validation: verify all k satisfy k*sigma(k) == M and k <= sqrt(M)
    sqrt_M = _isqrt(best_M)
    final_ks = []
    for k in best_ks:
        if k <= sqrt_M and _sigma_by_factorization(k) * k == best_M:
            final_ks.append(k)
    
    return {'M': best_M, 'k': final_ks}

# EVOLVE-BLOCK-END

