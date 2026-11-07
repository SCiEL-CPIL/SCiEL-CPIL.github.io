import math
import time
from array import array

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s margin
    
    # Phase 1: Linear sieve for small k
    N = 8000000
    
    # Use array module for memory efficiency
    sigma = array('q', [0]) * (N + 1)
    p_pow = array('q', [0]) * (N + 1)
    sigma_rest = array('q', [0]) * (N + 1)
    spf = array('i', [0]) * (N + 1)
    
    sigma[1] = 1
    p_pow[1] = 1
    sigma_rest[1] = 1
    
    primes = []
    primes.append(2)
    spf[2] = 2
    sigma[2] = 3
    p_pow[2] = 2
    sigma_rest[2] = 1
    
    for i in range(3, N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
            sigma[i] = 1 + i
            p_pow[i] = i
            sigma_rest[i] = 1
            
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            spf[i * p] = p
            if p == spf[i]:
                p_pow[i * p] = p_pow[i] * p
                sigma_rest[i * p] = sigma_rest[i]
                sigma[i * p] = sigma[i] + p_pow[i] * p * sigma_rest[i]
                break
            else:
                p_pow[i * p] = p
                sigma_rest[i * p] = sigma[i]
                sigma[i * p] = sigma[i] * (1 + p)
                
        if time.time() > deadline - 40.0:
            break

    # Find the actual max k processed
    max_k_processed = N
    while max_k_processed > 1 and sigma[max_k_processed] == 0:
        max_k_processed -= 1
    
    # Group by k*sigma(k)
    groups = {}
    for k in range(1, max_k_processed + 1):
        c = k * sigma[k]
        if c not in groups:
            groups[c] = []
        groups[c].append(k)
        if time.time() > deadline - 30.0:
            break

    # Find best group
    best_group = []
    best_c = 0
    for c, xs in groups.items():
        if len(xs) > len(best_group):
            best_group = xs
            best_c = c
            
    best_count = len(best_group)
    best_M = int(best_c)
    
    # Phase 2: Targeted search using multiplicative structure
    # Key insight: k*sigma(k) = prod_i p_i^{e_i} * (p_i^{e_i+1}-1)/(p_i-1)
    # We can enumerate "shapes" and find M with many representations
    
    def sigma_individual(k):
        """Compute sigma(k) by factorization"""
        if k <= max_k_processed and sigma[k] > 0:
            return sigma[k]
        n = k
        result = 1
        d = 2
        while d * d <= n:
            if n % d == 0:
                power = 0
                while n % d == 0:
                    n //= d
                    power += 1
                result *= (d**(power+1) - 1) // (d - 1)
            d += 1 if d == 2 else 2
        if n > 1:
            result *= (n*n + n)
        return result
    
    # Search for k values in larger range that are products of small primes
    if time.time() < deadline - 20.0:
        # Generate candidate k values: products of small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        
        # Generate numbers that are 7-smooth or similar
        def gen_smooth(lo, hi, primes_list, max_depth=7):
            if max_depth == 0:
                if lo <= 1 <= hi:
                    yield 1
                return
            p = primes_list[0]
            rest = primes_list[1:]
            power = 1
            while power <= hi:
                for sub in gen_smooth(max(lo // power, 1), hi // power, rest, max_depth - 1):
                    val = power * sub
                    if lo <= val <= hi:
                        yield val
                power *= p
                if power > hi:
                    break
        
        # Generate 17-smooth numbers in range
        lo = max_k_processed + 1
        hi = min(20000000, int(10**9))  # cap at 20M for now
        
        count_checked = 0
        for k in gen_smooth(lo, hi, small_primes, 7):
            if time.time() > deadline - 15.0:
                break
            s = sigma_individual(k)
            c = k * s
            if c not in groups:
                groups[c] = []
            groups[c].append(k)
            count_checked += 1
            
            if count_checked % 100000 == 0 and time.time() > deadline - 15.0:
                break
        
        # Re-find best group
        for c, xs in groups.items():
            if len(xs) > len(best_group):
                best_group = xs
                best_c = c
                best_count = len(xs)
                best_M = int(c)
    
    # Phase 3: Explore multiplicative structure for higher collision density
    # For M = prod_i T_i where T_i = p_i^{e_i} * (p_i^{e_i+1}-1)/(p_i-1)
    # Multiple different factorizations of M into such terms give different k
    
    if time.time() < deadline - 10.0:
        # Precompute terms T(p,e) for small primes and exponents
        def term(p, e):
            """Return p^e * (p^(e+1)-1)/(p-1)"""
            if e == 0:
                return 1
            pe = p ** e
            # (p^(e+1)-1)/(p-1) = 1 + p + p^2 + ... + p^e
            sigma_pe = (p ** (e+1) - 1) // (p - 1)
            return pe * sigma_pe
        
        # Build a map from M to list of (k, representation)
        # Focus on M that can be written as product of terms in multiple ways
        
        # Strategy: For each prime p, compute T(p,e) for e=1,2,...
        # Then find M values that have many factorizations into such terms
        
        # This is complex, so let's do a targeted search:
        # For M in groups with high count, try to find additional k by 
        # exploring the multiplicative structure
        
        # Instead, let's focus on finding k values where k*sigma(k) equals
        # one of the existing high-count M values
        
        # Get the top M values by count
        top_M_values = []
        for c, xs in groups.items():
            if len(xs) >= 3:
                top_M_values.append((len(xs), c))
        top_M_values.sort(reverse=True)
        
        # For each top M, try to find additional k values
        # by exploring k = prod p_i^{e_i} where the product of terms equals M
        
        for _, M_val in top_M_values[:50]:  # Check top 50 M values
            if time.time() > deadline - 8.0:
                break
            
            # Try to find k such that k*sigma(k) = M_val
            # k must be a divisor of M_val, and k*sigma(k) = M_val
            
            # Enumerate divisors of M_val that could be k
            # k must satisfy: k | M_val and sigma(k) = M_val / k
            
            # This is expensive, so only do it for very promising M values
            if len(groups[M_val]) < 5:
                continue
            
            # Find divisors of M_val
            sqrt_Mv = math.isqrt(M_val)
            divisors = []
            for d in range(1, sqrt_Mv + 1):
                if M_val % d == 0:
                    divisors.append(d)
                    if d != M_val // d:
                        divisors.append(M_val // d)
            
            # For each divisor d of M_val, check if d*sigma(d) = M_val
            # and d <= sqrt(M_val)
            new_ks = []
            for d in divisors:
                if d > sqrt_Mv:
                    continue
                if d in groups.get(M_val, []):
                    continue  # Already have this k
                s = sigma_individual(d)
                if d * s == M_val:
                    new_ks.append(d)
            
            if new_ks:
                groups[M_val].extend(new_ks)
                if len(groups[M_val]) > len(best_group):
                    best_group = groups[M_val]
                    best_c = M_val
                    best_count = len(best_group)
                    best_M = int(M_val)
    
    # Verify and return
    if best_count >= 3:
        sqrt_M = math.isqrt(best_M)
        valid_ks = [int(x) for x in best_group if x <= sqrt_M]
        if len(valid_ks) >= 3:
            return {'M': best_M, 'k': valid_ks}
    
    # Fallback
    return {'M': 333312, 'k': [336, 372, 434]}
