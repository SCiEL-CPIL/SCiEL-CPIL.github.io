"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def time_left():
        return deadline - time.time()
    
    # Generate powerful numbers up to LIMIT
    # Powerful: every prime exponent >= 2 in factorization
    # Strategy: generate via primes and their powers
    LIMIT = 10**10
    
    # Generate primes up to sqrt(LIMIT) ~ 1e5
    def get_primes(n):
        sieve = bytearray(b'\x01') * (n + 1)
        sieve[0:2] = b'\x00\x00'
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i in range(2, n+1) if sieve[i]]
    
    primes = get_primes(isqrt(LIMIT) + 1)
    
    # Generate powerful numbers recursively
    powerful = set()
    powerful.add(1)
    
    def gen_powerful(idx, current, max_val):
        if time.time() > deadline - 5:
            return
        if current > max_val:
            return
        if idx >= len(primes):
            return
        p = primes[idx]
        # p^1 is not powerful (exponent 1), so start from p^2
        val = current * p * p
        while val <= max_val:
            powerful.add(val)
            # Multiply by next primes
            gen_powerful(idx + 1, val, max_val // (p * p))
            # Also try higher powers of p
            if val <= max_val // p:
                val *= p
            else:
                break
        # Also recurse without using this prime
        gen_powerful(idx + 1, current, max_val)
    
    # Use iterative approach to avoid deep recursion
    powerful_list = [1]
    # Generate all powerful numbers by considering each prime independently
    # A powerful number is a product of prime powers where each exponent >= 2
    # We'll use a different approach: generate by iterating possible structures
    
    # Better approach: generate powerful numbers by considering
    # n = a^2 * b^3 where b is squarefree
    # This is the standard characterization
    powerful_set = set()
    powerful_set.add(1)
    
    # b^3 part: b squarefree, b^3 <= LIMIT
    max_b_cubert = int(LIMIT ** (1/3)) + 1
    # a^2 part: a^2 <= LIMIT / b^3
    
    # Generate squarefree numbers up to max_b_cubert
    # A squarefree number has no prime factor appearing more than once
    def is_squarefree(n):
        if n == 1:
            return True
        for p in primes:
            if p * p > n:
                break
            if n % (p * p) == 0:
                return False
        return True
    
    # Actually, let's just generate powerful numbers directly
    # Using the a^2 * b^3 characterization
    # b must be squarefree
    
    # Precompute squarefree numbers up to max_b_cubert
    squarefree = []
    for b in range(1, max_b_cubert + 1):
        if is_squarefree(b):
            squarefree.append(b)
    
    for b in squarefree:
        b3 = b * b * b
        max_a = isqrt(LIMIT // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= LIMIT:
                powerful_set.add(val)
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
    
    powerful = sorted(powerful_set)
    powerful_set = set(powerful)
    n_powerful = len(powerful)
    
    if time.time() > deadline - 10:
        # Fallback to small search
        small_limit = 200000
        small_powerful = [p for p in powerful if p <= small_limit]
        small_set = set(small_powerful)
        best_n = 2
        best_count = 0
        for n in range(2, small_limit + 1):
            count = sum(1 for a in small_powerful if 2*a <= n and (n-a) in small_set)
            if count > best_count:
                best_count = count
                best_n = n
        pairs = [[a, best_n - a] for a in small_powerful if 2*a <= best_n and (best_n-a) in small_set]
        return {'N': best_n, 'pairs': pairs}
    
    # Now find N with maximum representations
    # Strategy: use frequency counting on sums
    # For each pair (i,j) with i<=j, compute sum and count
    # But n_powerful could be ~10k, so n^2/2 ~ 50M which is feasible
    
    sum_count = defaultdict(int)
    sum_pairs = defaultdict(list)
    
    n_p = len(powerful)
    for i in range(n_p):
        a = powerful[i]
        if 2 * a > LIMIT:
            break
        for j in range(i, n_p):
            b = powerful[j]
            s = a + b
            if s > LIMIT:
                break
            sum_count[s] += 1
            if len(sum_pairs[s]) < 200:  # Cap stored pairs
                sum_pairs[s].append((a, b))
        if time.time() > deadline - 15:
            break
    
    if time.time() > deadline - 15:
        # Find best from partial count
        if sum_count:
            best_n = max(sum_count, key=sum_count.get)
            pairs = sum_pairs.get(best_n, [])
            # Verify and complete pairs
            verified = []
            for a in powerful:
                if 2*a > best_n:
                    break
                b = best_n - a
                if b in powerful_set and b >= a:
                    verified.append([a, b])
            return {'N': best_n, 'pairs': verified}
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Find the N with maximum count
    best_n = max(sum_count, key=sum_count.get)
    best_count = sum_count[best_n]
    
    # Verify and get all pairs for best_n
    pairs = []
    for a in powerful:
        if 2 * a > best_n:
            break
        b = best_n - a
        if b in powerful_set and b >= a:
            pairs.append([a, b])
    
    # Try to improve with local search around best_n
    if time.time() < deadline - 20 and len(pairs) > 0:
        # Check neighbors
        candidates = [best_n]
        for delta in [1, -1, 2, -2, 3, -3, 5, -5, 10, -10, 100, -100]:
            cand = best_n + delta
            if 2 <= cand <= LIMIT:
                candidates.append(cand)
        
        best_n_final = best_n
        best_pairs_final = pairs
        best_count_final = len(pairs)
        
        for cand in candidates:
            if time.time() > deadline - 10:
                break
            count = 0
            cand_pairs = []
            for a in powerful:
                if 2 * a > cand:
                    break
                b = cand - a
                if b in powerful_set and b >= a:
                    count += 1
                    cand_pairs.append([a, b])
            if count > best_count_final:
                best_count_final = count
                best_n_final = cand
                best_pairs_final = cand_pairs
    
    return {'N': best_n_final, 'pairs': best_pairs_final}
# EVOLVE-BLOCK-END
