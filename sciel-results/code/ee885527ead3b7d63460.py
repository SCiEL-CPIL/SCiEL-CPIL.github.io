import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s margin before 150s limit
    
    def time_check():
        return time.time() > deadline
    
    # Generate primes up to a limit
    # For k*sigma(k) <= 10^18, if k is a prime power p^a:
    # p^a * (1 + p + ... + p^a) <= 10^18
    # For a=1: p*(p+1) <= 10^18 => p <= ~10^9 (but we need to be practical)
    # Let's generate primes up to 2*10^6 for single prime powers
    # For composite k with multiple prime factors, we'll use combinations
    
    def sieve(n):
        if n < 2:
            return []
        is_prime = bytearray(b'\x01') * (n + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, n+1) if is_prime[i]]
    
    # Generate primes up to 1,000,000 for good coverage
    primes = sieve(1000000)
    
    # Generate prime powers (p^a) with their sigma values
    # k = p^a, sigma(k) = 1 + p + p^2 + ... + p^a
    prime_powers = []  # (k_val, sigma_val, p)
    
    for p in primes:
        pk = p
        sigma = 1 + p
        while pk * sigma <= 10**18:
            prime_powers.append((pk, sigma, p))
            pk *= p
            sigma += pk
            if pk > 10**18:
                break
    
    # Sort by k value
    prime_powers.sort(key=lambda x: x[0])
    
    # Map M -> set of k values
    m_to_ks = defaultdict(set)
    
    # Add single prime powers
    for k_val, sigma_val, p in prime_powers:
        M = k_val * sigma_val
        if M <= 10**18 and k_val <= math.isqrt(M):
            m_to_ks[M].add(k_val)
    
    # Combine pairs of prime powers with different primes
    # k = p1^a1 * p2^a2, sigma(k) = sigma(p1^a1) * sigma(p2^a2)
    n_pp = len(prime_powers)
    
    # Use a reasonable subset for pairing to stay within time
    max_pp_for_pairs = min(n_pp, 50000)
    pp_for_pairs = prime_powers[:max_pp_for_pairs]
    
    # Group by prime to avoid same-prime combinations
    from collections import defaultdict as dd
    pp_by_prime = dd(list)
    for pp in pp_for_pairs:
        pp_by_prime[pp[2]].append(pp)
    
    prime_keys = sorted(pp_by_prime.keys())
    
    for i in range(len(prime_keys)):
        if time_check():
            break
        p1 = prime_keys[i]
        list1 = pp_by_prime[p1]
        for j in range(i+1, len(prime_keys)):
            p2 = prime_keys[j]
            list2 = pp_by_prime[p2]
            for k1, s1, _ in list1:
                for k2, s2, _ in list2:
                    k = k1 * k2
                    if k > 10**18:
                        continue
                    M = k * s1 * s2
                    if M > 10**18:
                        continue
                    if k <= math.isqrt(M):
                        m_to_ks[M].add(k)
            if time_check():
                break
    
    # Find M with most k's
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    for M, ks in m_to_ks.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_M = M
            best_ks = list(ks)
    
    # Verify and return
    final_ks = []
    for k in best_ks:
        if k <= math.isqrt(best_M):
            final_ks.append(k)
    
    if not final_ks:
        final_ks = [336, 372, 434]
        best_M = 333312
    
    return {'M': int(best_M), 'k': [int(k) for k in final_ks]}
