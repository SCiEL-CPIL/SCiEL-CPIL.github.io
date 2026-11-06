import math
import time
import random
import sys
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 110  # 10s margin before 120s limit
    
    # Deterministic Miller-Rabin bases for n < 2^64
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0:
            return False
        if n % 3 == 0:
            return n == 3
        
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
            
        for a in MR_BASES:
            if a % n == 0:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
            
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

    def factorize(n):
        factors = []
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                while n % p == 0:
                    factors.append(p)
                    n //= p
        if n == 1:
            return factors
        if is_prime(n):
            factors.append(n)
            return factors
        
        stack = [n]
        while stack:
            x = stack.pop()
            if is_prime(x):
                factors.append(x)
            else:
                d = pollard_rho(x)
                stack.append(d)
                stack.append(x // d)
        return factors

    def check_run(p):
        k = 0
        for offset in range(1, 201):
            n = p * p + offset
            if is_prime(n):
                break
            
            factors = factorize(n)
            if max(factors) > p:
                break
            k += 1
        return k

    # --- New Strategy: CRT Sieve ---
    # We want to find primes p such that for all i in [1, 200], p^2 + i is not divisible by any q <= Q.
    # This is equivalent to: p is NOT a root of p^2 = -i mod q for any q <= Q, i in [1, 200].
    # We sieve a range of potential p values.
    
    # 1. Generate small primes q up to Q
    Q = 50000
    sieve_q = bytearray(b'\x01') * (Q + 1)
    sieve_q[0] = sieve_q[1] = 0
    for i in range(2, int(Q ** 0.5) + 1):
        if sieve_q[i]:
            sieve_q[i*i::i] = bytearray(len(sieve_q[i*i::i]))
    small_primes = [i for i, is_p in enumerate(sieve_q) if is_p]
    
    # 2. Precompute CRT constraints
    # For each q, we want to exclude p such that p^2 + i == 0 mod q for some i in [1, 200].
    # This means p^2 == -i mod q.
    # We compute the set of residues R_q = { x in [0, q) | x^2 == -i mod q for some i in [1, 200] }
    # Then we exclude p if p mod q in R_q.
    
    # We will sieve a range of numbers [L, R] for p.
    # Let's pick a range. p must be prime.
    # Since p^2 + i <= p^2 + 200, and we want factors <= p, p must be at least sqrt(p^2+200) which is always true.
    # But if p is very small, p^2+1 might be prime > p.
    # Let's scan p in [10^5, 10^6] first with the sieve.
    
    L = 100000
    R = 200000
    size = R - L + 1
    
    # is_valid array
    is_valid = bytearray(b'\x01') * size
    
    # Apply sieve for each small prime q
    for q in small_primes:
        if q > 200:
            # If q > 200, then for i in [1, 200], -i mod q is just q-i.
            # We need to check if x^2 == q-i has a solution.
            # But actually, if q > 200, then p^2 + i < q for small p? No, p is ~10^5, p^2 is ~10^10.
            # The condition is p^2 + i = k*q.
            # If q > 200, then i < q.
            # So we need x^2 = -i mod q.
            # This is the same logic.
            
            # However, note that if q > 200, then for a fixed p, p^2+i can only be divisible by q if p^2+i >= q.
            # Since p >= 10^5, p^2 is huge, so this is always true.
            
            pass
        
        # Compute forbidden residues for this q
        # We need to solve x^2 = -i mod q for i=1..200.
        # This is equivalent to x^2 = q-i mod q.
        # Let S = { (q-i) % q for i in 1..200 }
        # We need x such that x^2 mod q in S.
        
        # Efficient way:
        # Iterate x from 0 to q-1, check if x^2 % q is in S.
        # But q can be up to 50000. 50000 * 50000 = 2.5e9 operations. Too slow for 5000 primes.
        # We need a faster way to find square roots mod q.
        # Or, we can just iterate i in 1..200, and find roots of x^2 = -i mod q.
        # Number of i is 200.
        # For each i, finding roots of x^2 = a mod q is O(sqrt(q)) or O(log q) with Tonelli-Shanks?
        # Actually, we can just precompute squares.
        
        # Let's optimize:
        # For a given q, create a boolean array is_forbidden of size q.
        # For i in 1..200:
        #   target = (-i) % q
        #   Find all x such that x^2 % q == target.
        #   Mark is_forbidden[x] = True.
        
        # How to find x?
        # If q is prime, we can use Tonelli-Shanks.
        # But q is prime (from small_primes).
        # Implementing Tonelli-Shanks for 200 targets * 5000 primes might be slow in Python.
        
        # Alternative:
        # Just iterate x from 0 to q-1?
        # Sum of q for all primes q <= 50000 is approx 50000^2 / (2 * ln 50000) ~ 2.5e9 / 10 ~ 2.5e8.
        # This is too slow in Python (10-100 seconds).
        
        # We need to limit Q or optimize the root finding.
        # Let's reduce Q to 5000.
        # Sum of primes <= 5000 is approx 5000^2 / (2 * ln 5000) ~ 25e6 / 9 ~ 2.7e6.
        # This is feasible.
        
        if q > 5000:
            continue
            
        is_forb = bytearray(q)
        for i in range(1, 201):
            target = (-i) % q
            # Find roots of x^2 = target mod q
            # Since q is prime, we can try to find one root and then the other is q-x.
            # But finding a root is the hard part.
            # For small q, we can just scan?
            # No, scanning x=0..q-1 for each i is 200*q.
            # Total work sum(200*q) for q<=5000.
            # Sum of primes <= 5000 is ~150,000.
            # 200 * 150,000 = 30,000,000 operations.
            # This is fast in Python!
            
            # Let's do it.
            # Optimization: Precompute squares for this q?
            # squares = [x*x % q for x in range(q)]
            # Then check if target in squares.
            # But we need the index x.
            # We can build a dict: square_val -> list of x.
            
            # Let's build the dict once per q.
            # Actually, we can just do it inside the loop? No, build once.
            
            # Re-structure:
            # Build square_map for q.
            square_map = {}
            for x in range(q):
                sq = (x * x) % q
                if sq not in square_map:
                    square_map[sq] = []
                square_map[sq].append(x)
            
            # Now for each i, get roots.
            roots = square_map.get(target, [])
            for x in roots:
                is_forb[x] = 1
        
        # Apply sieve
        # p = L + j
        # We want to exclude if (L + j) % q in is_forb
        # i.e., j % q in (is_forb - L) mod q ?
        # Let r = L % q.
        # We want (L + j) % q = (r + j) % q.
        # So we want to exclude j such that (r + j) % q is in is_forb.
        # Let y = (r + j) % q.
        # j = (y - r) % q.
        # So for each y in is_forb (where is_forb[y]==1),
        # j = (y - r) % q.
        # Then j, j+q, j+2q, ... are forbidden.
        
        r = L % q
        for y in range(q):
            if is_forb[y]:
                j0 = (y - r) % q
                # Mark j0, j0+q, ... in is_valid
                for j in range(j0, size, q):
                    is_valid[j] = 0

    # 3. Find primes in the valid range
    best_k = 0
    best_p = 7 # Fallback
    
    # We need to check primality of the remaining candidates
    # And then check_run
    
    # Iterate through valid indices
    for j in range(size):
        if time.time() - start_time > DEADLINE - 5:
            break
        if is_valid[j]:
            p = L + j
            if is_prime(p):
                k = check_run(p)
                if k > best_k:
                    best_k = k
                    best_p = p
    
    # If we didn't find a good one, or to explore more, try random large primes?
    # The sieve method is very fast for small p.
    # For large p, the sieve range must be large, which is memory intensive.
    # But p^2+i smoothness is rare.
    # Let's stick to the sieve result.
    
    if best_k < 1:
        # Fallback to original simple search if sieve failed
        for p in [7, 11, 13, 17, 19, 23]:
            k = check_run(p)
            if k > best_k:
                best_k = k
                best_p = p

    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
