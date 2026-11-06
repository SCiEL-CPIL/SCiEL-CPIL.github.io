import time
import math

_START = time.time()
_HARD_DEADLINE = _START + 85.0  # Safety margin for the 90s limit

def _prime_factors(num):
    if num <= 1:
        return frozenset()
    out = []
    x = num
    if x % 2 == 0:
        out.append(2)
        while x % 2 == 0:
            x //= 2
    f = 3
    while f * f <= x:
        if x % f == 0:
            out.append(f)
            while x % f == 0:
                x //= f
        f += 2
    if x > 1:
        out.append(x)
    return frozenset(out)

def construct():
    deadline = time.time() + 85.0
    n = 12
    
    # Precompute prime factors for all possible sums (max sum 20000)
    pf_cache = {s: _prime_factors(s) for s in range(2, 20001)}
    
    def count_distinct_primes(arr):
        union = set()
        for i in range(n - 1):
            ai = arr[i]
            for j in range(i + 1, n):
                union |= pf_cache[ai + arr[j]]
        return len(union)

    # Build a pool of candidate numbers.
    # We want numbers that are "smooth" (have small prime factors) to encourage
    # sums to also have small prime factors.
    # Let's generate 7-smooth numbers up to 10000.
    pool = set()
    # Generate numbers of form 2^a * 3^b * 5^c * 7^d
    for a in range(0, 14): # 2^13 > 10000
        v2 = 2 ** a
        if v2 > 10000: break
        for b in range(0, 9): # 3^8 > 10000
            v3 = v2 * (3 ** b)
            if v3 > 10000: break
            for c in range(0, 7): # 5^6 > 10000
                v5 = v3 * (5 ** c)
                if v5 > 10000: break
                for d in range(0, 6): # 7^5 > 10000
                    v7 = v5 * (7 ** d)
                    if v7 > 10000: break
                    if v7 >= 1:
                        pool.add(v7)
                    
    # Also add some small integers just in case
    for i in range(1, 101):
        pool.add(i)
        
    # Sort pool by value to prioritize smaller numbers which often have smoother sums
    pool = sorted(list(pool))
    
    # Initial seeds
    seeds = []
    # Seed 1: Smallest smooth numbers
    seeds.append(pool[:12])
    
    # Seed 2: Multiples of 30 (2*3*5)
    seeds.append([30 * i for i in range(1, 13)])
    
    # Seed 3: Numbers congruent to 1 mod 30 (odd, not div by 3, 5)
    # Sums will be 2 mod 3, 2 mod 5, even. Primes: 2, and others.
    seeds.append([30 * i + 1 for i in range(1, 13)])
    
    # Seed 4: Range 1-12
    seeds.append(list(range(1, 13)))
    
    # Seed 5: Powers of 2 and 3 mixed
    seq = []
    for a in range(5):
        for b in range(5):
            val = (2**a) * (3**b)
            if 1 <= val <= 10000:
                seq.append(val)
    seq = sorted(set(seq))
    if len(seq) >= 12:
        seeds.append(seq[:12])

    best = None
    best_score = 10**9
    
    for s in seeds:
        if len(set(s)) != 12: continue
        if any(x > 10000 for x in s): continue
        if any(x <= 0 for x in s): continue
        sc = count_distinct_primes(s)
        if sc < best_score:
            best_score = sc
            best = list(s)
            
    if best is None:
        return list(range(1, 13))

    current = list(best)
    improved = True
    iters = 0
    max_iters = 1000
    
    # Local Search
    while improved and iters < max_iters and time.time() < deadline - 2.0:
        improved = False
        iters += 1
        cur_score = count_distinct_primes(current)
        
        # Try replacing each element
        for idx in range(n):
            if time.time() > deadline - 2.0:
                return list(best)
            
            old_val = current[idx]
            existing = set(current)
            
            # Try candidates from the pool
            # To speed up, we can limit the number of candidates tried per step
            # or just iterate through the pool. The pool is ~200-300 items.
            # 12 * 300 * 144 ops = ~500k ops per full pass. Very fast.
            
            best_cand = None
            best_cand_score = cur_score
            
            for cand in pool:
                if cand == old_val or cand in existing:
                    continue
                if cand > 10000:
                    break
                    
                current[idx] = cand
                sc = count_distinct_primes(current)
                
                if sc < best_cand_score:
                    best_cand_score = sc
                    best_cand = cand
                    
            if best_cand is not None:
                current[idx] = best_cand
                cur_score = best_cand_score
                improved = True
                if cur_score < best_score:
                    best_score = cur_score
                    best = list(current)
                if best_score <= 6: # Theoretical lower bound is likely higher, but 6 is very good
                    return list(best)
                # Restart scan from index 0 for better convergence
                break
                
    return list(best)
