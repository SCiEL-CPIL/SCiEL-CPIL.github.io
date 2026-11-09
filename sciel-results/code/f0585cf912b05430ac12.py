import random
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 115.0
    
    # Best result so far
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = len(best_k)
    
    # Primes for factoring and divisor generation
    # We need primes up to 10^6 to factor numbers up to 10^18 efficiently
    # (If remaining cofactor > 1 after dividing by primes <= 10^6, it's prime or product of two primes > 10^6)
    # For highly composite numbers, factors are small, so this is sufficient.
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(100000) # 100,000 primes is plenty for small factor bases
    
    def factorize(n, primes):
        factors = {}
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                cnt = 0
                while temp % p == 0:
                    temp //= p
                    cnt += 1
                factors[p] = cnt
        if temp > 1:
            factors[temp] = 1
        return factors

    def get_divisors(factors):
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            p_pow = 1
            for _ in range(e + 1):
                for d in divs:
                    new_divs.append(d * p_pow)
                p_pow *= p
            divs = new_divs
        return divs

    def sigma_from_factors(factors):
        s = 1
        for p, e in factors.items():
            s *= (p**(e+1) - 1) // (p - 1)
        return s

    # Generate candidate M's
    # We want M to be highly composite.
    # Let's try M of the form 2^a 3^b 5^c 7^d ...
    # We'll do a DFS to generate numbers with many divisors.
    
    candidates = []
    
    def gen_candidates(prime_idx, current_val, max_val, exponents, max_depth=10):
        if time.time() - start_time > deadline - 10:
            return
        if prime_idx >= len(primes) or max_depth < 0:
            if current_val > 1:
                candidates.append(current_val)
            return
        
        p = primes[prime_idx]
        # Try exponents from high to low? Or just a range.
        # For highly composite, exponents are non-increasing.
        # But we don't strictly need that, just many divisors.
        # Let's just try a few combinations of small primes.
        
        # Heuristic: Try exponents 1 to 20 for first few primes.
        # This is a heuristic search.
        
        # To ensure we find good candidates, let's manually construct some known highly composite structures.
        # Or just iterate a grid of exponents for the first 6 primes.
        
        # This recursive approach might be too slow or generate too many.
        # Let's stick to a simpler heuristic:
        # Generate numbers of form 2^a 3^b 5^c 7^d 11^e 13^f
        # with a >= b >= c >= d >= e >= f >= 0
        # and value <= 10^18.
        
        pass

    # Simpler approach: Just pick a few specific highly composite numbers known to be large.
    # Or, generate them by multiplying primes.
    
    # Let's try to find M by checking random highly composite-like numbers.
    # Actually, let's just check the top 100 highly composite numbers below 10^18.
    # We can generate them by taking products of small primes with exponents.
    
    # List of primes to use in M
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # We will generate candidates by varying exponents.
    # To avoid explosion, we limit the number of candidates.
    
    # A better strategy:
    # The number of solutions is bounded by the number of divisors.
    # So we want M with many divisors.
    # Let's generate M by taking 2^a * 3^b * 5^c * 7^d * 11^e * 13^f ...
    # such that M <= 10^18.
    
    # We'll use a priority queue or just a nested loop for the first few primes.
    
    # Let's try a specific set of M's that are "dense".
    # M = 2^40 * 3^25 * 5^15 ...
    
    # Let's just iterate through a range of "smooth" numbers.
    # Actually, the most efficient way is to just pick a few M's with high divisor counts.
    
    # We can use the fact that if M is a product of first k primes, it has many divisors.
    
    # Let's try to find M by searching.
    # We'll generate M values that are highly composite.
    
    # Generate highly composite numbers up to 10^18
    # This is a standard algorithm.
    
    hcn = []
    
    def dfs_hcn(idx, curr, max_exp, limit):
        if time.time() - start_time > deadline - 5:
            return
        if idx >= len(small_primes):
            hcn.append(curr)
            return
        p = small_primes[idx]
        # Max exponent is constrained by limit and previous exponent
        # For HCN, exponents are non-increasing.
        for e in range(max_exp, -1, -1):
            if curr * (p ** e) > limit:
                break
            dfs_hcn(idx + 1, curr * (p ** e), e, limit)
    
    # This might generate too many or not enough.
    # Let's just pick a few specific ones.
    
    # Known highly composite numbers:
    # 1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260, 1680, 2520, 5040, 7560, 10080, 15120, 20160, 25200, 27720, 45360, 50400, 55440, 83160, 110880, 166320, 221760, 277200, 332640, 498960, 554400, 665280, 720720, 1081080, 1441440, 2162160, 2882880, 3603600, 4324320, 6486480, 7207200, 8648640, 10810800, 14414400, 17297280, 21621600, 32432400, 36756720, 43243200, 61261200, 73513440, 110270160, 122522400, 147026880, 183783600, 245044800, 294053760, 367567200, 551350800, 698377680, 735134400, 1102701600, 1396755360, 2095133040, 2205403200, 2327925600, 2793510720, 4190266080, 4655851200, 6983776800, 8380532160, 9311702400, 11639628000, 13967553600, 18623404800, 20951330400, 27935107200, 32387788800, 41902660800, 55870214400, 64775577600, 73513440000, 110270160000, 129541152000, 147026880000, 183783600000, 219340320000, 259092480000, 338770224000, 367567200000, 439184928000, 490134528000, 558702144000, 612612612000, 698377680000, 777930768000, 918918918000, 997468128000, 1096744272000, 1197744432000, 1396755360000, 1593851328000, 1793851320000, 1995488864000, 2193403200000, 2393510736000, 2590924800000, 2793510720000, 3193403200000, 3587826240000, 3987702240000, 4391849280000, 4791741760000, 5191850400000, 5591968960000, 5992087520000, 6392206080000, 6792324640000, 7192443200000, 7592561760000, 7992680320000, 8392798880000, 8792917440000, 9193036000000, 9593154560000, 9993273120000
    
    # This list is too long and hard to manage.
    
    # Let's just use the DFS to generate a few.
    
    # We'll limit the depth and exponents.
    
    # Actually, let's just try to find M by checking divisors of a few large highly composite numbers.
    
    # Let's pick M = 2^40 * 3^24 * 5^14 * 7^9 * 11^6 * 13^4 * 17^3 * 19^2 * 23
    # This is roughly 10^18.
    
    # Let's calculate a few such M's.
    
    test_Ms = []
    
    # Helper to create M from exponents
    def make_M(exps):
        m = 1
        for i, e in enumerate(exps):
            m *= small_primes[i] ** e
        return m
    
    # Try a few patterns
    # Pattern 1: 2^40, 3^24, 5^14, 7^9, 11^6, 13^4, 17^3, 19^2, 23^1
    # Pattern 2: 2^36, 3^22, 5^13, 7^8, 11^5, 13^3, 17^2, 19^1
    # Pattern 3: 2^32, 3^20, 5^12, 7^7, 11^4, 13^2, 17^1
    
    patterns = [
        [40, 24, 14, 9, 6, 4, 3, 2, 1, 0, 0, 0],
        [36, 22, 13, 8, 5, 3, 2, 1, 0, 0, 0, 0],
        [32, 20, 12, 7, 4, 2, 1, 0, 0, 0, 0, 0],
        [28, 18, 11, 6, 3, 1, 0, 0, 0, 0, 0, 0],
        [24, 16, 10, 5, 2, 1, 0, 0, 0, 0, 0, 0],
        [20, 14, 9, 4, 1, 0, 0, 0, 0, 0, 0, 0],
        [16, 12, 8, 3, 1, 0, 0, 0, 0, 0, 0, 0],
        [12, 10, 7, 2, 0, 0, 0, 0, 0, 0, 0, 0],
        [8, 8, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    
    for p in patterns:
        m = make_M(p)
        if m <= 10**18:
            test_Ms.append(m)
            
    # Also add some random highly composite-like numbers
    for _ in range(100):
        exps = [0]*12
        for i in range(12):
            if small_primes[i] ** (i+1) > 10**18:
                break
            exps[i] = random.randint(0, 30)
        m = make_M(exps)
        if m <= 10**18 and m > 1:
            test_Ms.append(m)
            
    # Remove duplicates
    test_Ms = list(set(test_Ms))
    
    # Sort by number of divisors (descending)
    def num_divisors(n):
        f = factorize(n, primes)
        d = 1
        for e in f.values():
            d *= (e + 1)
        return d
    
    # We can't sort 100+ numbers by num_divisors if it's slow, but it's fast.
    # But we only have ~100 numbers.
    
    # Actually, let's just check them all.
    
    for M in test_Ms:
        if time.time() - start_time > deadline - 2:
            break
            
        factors = factorize(M, primes)
        divisors = get_divisors(factors)
        
        valid_ks = []
        for k in divisors:
            if k > M ** 0.5:
                continue
            # Check if k is a solution
            # M / k = sigma(k)
            if M % k != 0:
                continue
            target_sigma = M // k
            # Compute sigma(k)
            k_factors = factorize(k, primes)
            s = sigma_from_factors(k_factors)
            if s == target_sigma:
                valid_ks.append(k)
                
        if len(valid_ks) > best_count:
            best_count = len(valid_ks)
            best_M = M
            best_k = valid_ks
            
    return {'M': int(best_M), 'k': [int(k) for k in best_k]}
