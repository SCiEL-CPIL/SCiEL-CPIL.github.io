# EVOLVE-BLOCK-START
import time
from math import isqrt
import numpy as np

def _sieve_primes(n):
    """Sieve of Eratosthenes for primes up to n."""
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using m^2 * b where b is squarefree divisor of m."""
    # Precompute smallest prime factor for efficient factorization
    spf = list(range(isqrt(limit) + 2))
    for i in range(2, isqrt(isqrt(limit)) + 1):
        if spf[i] == i:
            for j in range(i*i, len(spf), i):
                if spf[j] == j:
                    spf[j] = i
    
    powerful_set = set([1])
    sqrt_limit = isqrt(limit)
    
    for m in range(1, sqrt_limit + 1):
        # Factor m to get distinct prime factors
        temp = m
        primes = []
        while temp > 1:
            if temp <= len(spf):
                p = spf[temp]
            else:
                # temp is prime (greater than sqrt(limit))
                p = temp
            primes.append(p)
            while temp % p == 0:
                temp //= p
        
        # Enumerate all subsets of distinct primes
        n_primes = len(primes)
        for mask in range(1 << n_primes):
            b = 1
            for i in range(n_primes):
                if mask & (1 << i):
                    b *= primes[i]
            if b <= limit // (m * m):
                powerful_set.add(m * m * b)
    
    return sorted(powerful_set)

def _divisors_count(n):
    """Count divisors of n."""
    count = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            count += 1
            if i * i != n:
                count += 1
        i += 1
    return count

def _construct_N_from_powerful(pw):
    """Construct N as sum of two powerful numbers with high multiplicity.
    
    Strategy: pick powerful numbers with many divisors (like 2^a * 3^b * 5^c * 7^d),
    and try N = a + b where both a and b are such "rich" powerful numbers.
    The idea is that if a and b have many divisors, then N = a + b might also
    have many representations as sums of powerful numbers.
    """
    # Find powerful numbers with high divisor count
    scored = []
    for p in pw:
        if p > 1:
            d = _divisors_count(p)
            scored.append((d, p))
    
    scored.sort(reverse=True)
    
    # Try combinations of top powerful numbers
    top = [p for _, p in scored[:200]]
    
    candidates = []
    for i in range(len(top)):
        for j in range(i, len(top)):
            N = top[i] + top[j]
            if N <= 10**18:
                candidates.append(N)
    
    # Also try N = 2*p for top p (a=b case)
    for p in top[:100]:
        if 2 * p <= 10**18:
            candidates.append(2 * p)
    
    # Deduplicate and sort by N
    candidates = sorted(set(candidates))
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Phase 1: Generate powerful numbers
    N_LIMIT = 1_000_000_000  # 10^9
    powerful = _generate_powerful(N_LIMIT)
    powerful_set = set(powerful)
    P = len(powerful)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 60:
        # Fallback: return a small valid result
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Phase 2: Generate candidate N values
    candidates = _construct_N_from_powerful(powerful)
    
    # Phase 3: Verify candidates using numpy
    pw_arr = np.array(powerful, dtype=np.int64)
    
    best_N = 2
    best_pairs = [[1, 1]]
    best_count = 1
    
    # Sort candidates by potential (we'll verify all, but prioritize)
    # For each candidate N, count how many pairs (a, b) with a <= b, a + b = N, a, b powerful
    # Use: for each a in powerful, check if N - a is in powerful_set and N - a >= a
    
    # Process candidates, checking time periodically
    checked = 0
    for N in candidates:
        if time.time() > deadline - 15:
            break
        
        # Count representations
        # a must satisfy: a <= N/2, a in powerful, N-a in powerful
        # Use numpy for vectorized check
        # a values: powerful numbers <= N/2
        mask_a = pw_arr <= N // 2
        a_candidates = pw_arr[mask_a]
        
        if len(a_candidates) == 0:
            continue
        
        b_values = N - a_candidates
        # Check which b_values are in powerful_set
        # Convert to list and check membership
        b_list = b_values.tolist()
        valid = [a for a, b in zip(a_candidates.tolist(), b_list) if b in powerful_set]
        
        count = len(valid)
        if count > best_count:
            best_count = count
            best_N = N
            best_pairs = [[a, N - a] for a in valid]
        
        checked += 1
        if checked % 100 == 0 and time.time() > deadline - 30:
            break
    
    # Phase 4: If we have time, do a broader search around the best N
    if time.time() < deadline - 30 and best_N > 2:
        # Try nearby N values
        for delta in range(1, 200):
            for N_try in [best_N + delta, best_N - delta]:
                if N_try < 2 or N_try > 10**18:
                    continue
                if time.time() > deadline - 15:
                    break
                
                mask_a = pw_arr <= N_try // 2
                a_candidates = pw_arr[mask_a]
                
                if len(a_candidates) == 0:
                    continue
                
                b_values = N_try - a_candidates
                b_list = b_values.tolist()
                valid = [a for a, b in zip(a_candidates.tolist(), b_list) if b in powerful_set]
                
                count = len(valid)
                if count > best_count:
                    best_count = count
                    best_N = N_try
                    best_pairs = [[a, N_try - a] for a in valid]
    
    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
