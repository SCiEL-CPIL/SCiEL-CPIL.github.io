import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Deterministic Miller-Rabin for 64-bit
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def is_prime(n):
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in MR_BASES:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
            y = random.randrange(1, n)
            c = random.randrange(1, n)
            m = 128
            g, q, r = 1, 1, 1
            while g == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % n
                k = 0
                while k < r and g == 1:
                    ys = y
                    for _ in range(min(m, r - k)):
                        y = (y * y + c) % n
                        q = q * abs(x - y) % n
                    g = math.gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g

    def all_factors_leq(n, limit):
        """Check if all prime factors of n are <= limit."""
        if n <= 1:
            return True
        # Trial division by small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for p in small_primes:
            if n % p == 0:
                if p > limit:
                    return False
                while n % p == 0:
                    n //= p
                if n == 1:
                    return True
                if is_prime(n):
                    return n <= limit
                break

        if is_prime(n):
            return n <= limit

        # Pollard Brent
        d = pollard_brent(n)
        return all_factors_leq(d, limit) and all_factors_leq(n // d, limit)

    def check_run_full(p, max_k=200):
        """Full factorization check for p^2+1 .. p^2+k."""
        k = 0
        base = p * p
        while k < max_k:
            n = base + k + 1
            if not all_factors_leq(n, p):
                break
            k += 1
        return k

    # Sieve of Eratosthenes
    def sieve_primes(limit):
        if limit < 2:
            return []
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i, is_p in enumerate(sieve) if is_p]

    # Tonelli-Shanks for modular square root
    def tonelli_shanks(n, p):
        if p == 2:
            return n % 2
        if n % p == 0:
            return 0
        if pow(n, (p - 1) // 2, p) != 1:
            return None
        if p % 4 == 3:
            return pow(n, (p + 1) // 4, p)
        # General case
        q_val = p - 1
        s = 0
        while q_val % 2 == 0:
            q_val //= 2
            s += 1
        z = 2
        while pow(z, (p - 1) // 2, p) != p - 1:
            z += 1
        m = s
        c = pow(z, q_val, p)
        t = pow(n, q_val, p)
        r = pow(n, (q_val + 1) // 2, p)
        while t != 1:
            i = 0
            t2 = t
            while t2 != 1:
                t2 = (t2 * t2) % p
                i += 1
            b = pow(c, 1 << (m - i - 1), p)
            m = i
            c = (b * b) % p
            t = (t * c) % p
            r = (r * b) % p
        return r

    # Precompute: for each prime q up to Q_LIMIT, for each i in [1, 200],
    # the residues r mod q where r^2 ≡ -i (mod q)
    Q_LIMIT = 50000
    small_primes_list = sieve_primes(Q_LIMIT)

    # For efficiency, precompute roots for each q
    # q_roots[q] = list of length 201, where q_roots[q][i] = set of roots
    q_roots = {}
    for q in small_primes_list:
        roots_for_q = [set() for _ in range(201)]
        for i in range(1, 201):
            target = (-i) % q
            if target == 0:
                roots_for_q[i].add(0)
            elif q == 2:
                if target == 1:
                    roots_for_q[i].add(1)
            else:
                # Check if target is a quadratic residue mod q
                if pow(target, (q - 1) // 2, q) == 1:
                    root1 = tonelli_shanks(target, q)
                    if root1 is not None:
                        roots_for_q[i].add(root1)
                        roots_for_q[i].add((q - root1) % q)
        q_roots[q] = roots_for_q

    # Phase 1: Sieve-based search for primes up to 10^7
    # For each prime p, check if it's "forbidden" by any q > p in small_primes_list
    # A prime p is forbidden for i if there exists q > p (q in small_primes_list) 
    # such that p mod q is in q_roots[q][i]
    
    SIEVE_LIMIT = 10**7
    all_primes = sieve_primes(SIEVE_LIMIT)
    
    best_k = 0
    best_p = 2

    # For each prime p, do a quick sieve check
    # Optimization: for each p, only check q in (p, Q_LIMIT]
    # This is O(200 * pi(Q_LIMIT) - pi(p)) per p, which is expensive for small p
    
    # Better approach: iterate through q and mark forbidden p values
    # For each q in small_primes_list, for each i in [1, 200],
    # for each root r in q_roots[q][i], mark p ≡ r (mod q) as forbidden for i
    # But we need p to be prime and p < q (since we only care about q > p)
    
    # Actually, let's just do the direct check but with early termination
    # For a given p, check i=1,2,... until we find a forbidden i
    
    def quick_check(p):
        """Return the maximum k such that for all i in [1, k], p^2+i is not divisible by any prime q in (p, Q_LIMIT]."""
        k = 0
        for i in range(1, 201):
            bad = False
            for q in small_primes_list:
                if q <= p:
                    continue
                if q > Q_LIMIT:
                    break
                p_mod_q = p % q
                if p_mod_q in q_roots[q][i]:
                    bad = True
                    break
            if bad:
                break
            k += 1
        return k

    # Iterate through primes
    for p in all_primes:
        if time.time() > deadline:
            break
        
        # Quick check
        k_quick = quick_check(p)
        
        # If k_quick is 0, skip
        if k_quick == 0:
            continue
        
        # If k_quick is high, do full check
        if k_quick > best_k or k_quick >= best_k - 3:
            k_full = check_run_full(p)
            if k_full > best_k:
                best_k = k_full
                best_p = p

    # Phase 2: Try larger primes using Miller-Rabin
    if time.time() < deadline - 15:
        # Try random large primes in [10^7, 10^9]
        for _ in range(200):
            if time.time() > deadline:
                break
            p = random.randrange(10**7, 10**9)
            p |= 1  # Make odd
            if not is_prime(p):
                continue
            k_quick = quick_check(p)
            if k_quick > 0:
                k_full = check_run_full(p)
                if k_full > best_k:
                    best_k = k_full
                    best_p = p

    return {'p': best_p, 'k': best_k}
