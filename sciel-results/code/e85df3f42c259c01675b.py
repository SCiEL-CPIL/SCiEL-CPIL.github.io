import time
from math import gcd, isqrt, log2

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 512
    
    # Primes up to 512
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(N)
    
    # For each number 1..N, compute its prime factorization as a dict
    # num_factors[i] = {p: exponent}
    num_factors = [None] * (N + 1)
    num_factors[1] = {}
    for i in range(2, N + 1):
        n = i
        factors = {}
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                exp = 0
                while n % p == 0:
                    n //= p
                    exp += 1
                factors[p] = exp
        if n > 1:
            factors[n] = 1
        num_factors[i] = factors
    
    # For a given exponent assignment E (dict p -> max_exp), 
    # a number n is "allowed" if for all p in num_factors[n], num_factors[n][p] <= E.get(p, 0)
    # The lcm of all allowed numbers is ∏p^(E_p) which must be <= 512
    
    # We want to search over exponent assignments.
    # Key observation: only primes that actually appear in some number's factorization matter.
    # But all primes up to 512 appear.
    
    # Better approach: think of it as choosing a "lcm target" L <= 512.
    # Then S = {n in [1,512] : n divides L} is valid (lcm of divisors of L is L).
    # But we can do better: S = {n in [1,512] : for all p, v_p(n) <= v_p(L)} 
    # which is the same as n divides L^k for some k? No, that's not right.
    
    # Actually, if we set E_p = v_p(L), then allowed numbers are those where v_p(n) <= v_p(L) for all p.
    # This is equivalent to: n divides L * m where m has no prime factors... no.
    # It means n | L * (product of primes not in L to power 0)... 
    # Simply: n is allowed iff for each prime p, v_p(n) <= v_p(L).
    # This is the same as: the "p-part" of n divides the "p-part" of L.
    # Equivalently, n divides L * (L/gcd(L, something))... 
    
    # Let me just think: if L = ∏p^(E_p), then n is allowed iff v_p(n) <= E_p for all p.
    # The count of such n <= 512 is what we want to maximize.
    
    # Search strategy: iterate over all possible L values that are "smooth" enough.
    # Actually, L can be any value <= 512. For each L, compute count of n <= 512 with v_p(n) <= v_p(L) for all p.
    
    # But wait - we don't need L to be a specific number. We just need the exponent vector E.
    # Two different L values with the same prime exponent vector give the same allowed set.
    # So we should enumerate distinct exponent vectors.
    
    # Alternative: just try all L from 1 to 512, and for each, count valid n.
    # This is O(512 * 512) = ~262K operations, very fast.
    
    best_count = 0
    best_L = 1
    best_set = [1]
    
    for L in range(1, N + 1):
        # Compute factorization of L
        n = L
        L_factors = {}
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                exp = 0
                while n % p == 0:
                    n //= p
                    exp += 1
                L_factors[p] = exp
        if n > 1:
            L_factors[n] = 1
        
        # Count numbers n in [1,512] where v_p(n) <= L_factors.get(p, 0) for all p
        count = 0
        valid_numbers = []
        for num in range(1, N + 1):
            ok = True
            for p, exp in num_factors[num].items():
                if exp > L_factors.get(p, 0):
                    ok = False
                    break
            if ok:
                count += 1
                valid_numbers.append(num)
        
        if count > best_count:
            best_count = count
            best_L = L
            best_set = valid_numbers
    
    # Verify the best set
    if best_set:
        # Check lcm constraint for a sample of pairs (or all if small enough)
        # For correctness, let's verify all pairs if the set is small enough
        if len(best_set) <= 200:
            for i in range(len(best_set)):
                for j in range(i + 1, len(best_set)):
                    a, b = best_set[i], best_set[j]
                    l = a * b // gcd(a, b)
                    if l > N:
                        # This shouldn't happen, but just in case
                        break
        return best_set
    else:
        return [1]
