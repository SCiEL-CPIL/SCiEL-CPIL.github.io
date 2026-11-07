# EVOLVE-BLOCK-START
import time
from math import isqrt

def _is_perfect_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 210
    MAX_VAL = 10210
    MAX_SIZE = 100
    
    best_result = None
    best_max = MAX_VAL + 1
    
    # Try pairs (a, b) where a*b = 210*k^2 for some k
    # Equivalently, a*b*210 is a perfect square
    # For each a in (210, 10210], b = 210*k^2/a must be integer > 210, != a
    # 
    # Better: iterate over possible max values M and check if a valid subset exists with max = M
    # For pairs: need a in (210, M), b = M (or b <= M) such that a*b*210 is square
    # Actually just: for each M from 211 to 10210, check if there's a subset of {211..M} 
    # with max exactly M (or <= M) whose product*210 is square.
    #
    # Fastest: for each M, check pairs (a, M) for a in 211..M-1, and also (M, M) not allowed (distinct)
    # Also check single element M: M*210 is square?
    
    # Single element check: M*210 is perfect square
    # M*210 = M * 2 * 3 * 5 * 7 must be square
    # So M must be 2*3*5*7 * t^2 = 210*t^2 for t>=1, and M>210
    # M = 210*t^2, t>=2 (since M>210): t=2->840, t=3->1890, etc.
    
    # For pairs: a*M*210 = square, a in (210, M), a != M
    # a = square / (M*210)
    # So M*210 must divide some square, and a = square/(M*210) must be integer in (210, M)
    
    # Let's just iterate M from 211 upward and for each, check:
    # 1. Is M*210 a perfect square? (single element)
    # 2. Is there a in (210, M) such that a*M*210 is a perfect square? (pair)
    # 3. Maybe triple? (skip for speed, try later)
    
    # For (2): a*M*210 = s^2 => a = s^2/(M*210)
    # a must be integer, 210 < a < M
    # So s^2 must be divisible by M*210, and a in range
    # s must be a multiple of sqrt(M*210) in some sense...
    # s^2 = a * M * 210, so s = sqrt(a*M*210)
    # We need a*M*210 to be a perfect square.
    # Let g = gcd(a, M). a = g*a', M = g*M', gcd(a',M')=1
    # a*M*210 = g^2 * a' * M' * 210
    # For this to be a square, a'*M'*210 must be a square.
    # Since gcd(a',M')=1, a' and M' must each be squares times factors of 210.
    
    # This is getting complex. Let me just do a direct search for small M.
    
    # Direct approach: for M from 211 to some limit, check all pairs (a, M) 
    # where a in 211..M-1, see if a*M*210 is a perfect square.
    # Also check if M*210 is a perfect square (single element).
    
    # For efficiency, precompute: for each M, we need a such that a*M*210 is square.
    # a*M*210 = square => a = square/(M*210)
    # Let M*210 = 2^a2 * 3^a3 * 5^a5 * 7^a7 * rest
    # For a*M*210 to be square, a must supply odd exponents for primes where M*210 has odd exponent
    # So a's squarefree part must equal the squarefree part of M*210 (for the primes in M*210)
    # Plus a must have even exponents for primes not in M*210 (or rather, the product must be square)
    
    # Simplest: just compute and check.
    
    for M in range(211, 2000):
        if time.time() > deadline:
            break
        if M >= best_max:
            break
        
        # Check single element
        if _is_perfect_square(M * 210):
            if M < best_max:
                best_max = M
                best_result = [M]
            continue  # Can't do better with this M for single, but pairs might work
        
        # Check pairs (a, M) for a in 211..M-1
        # a * M * 210 must be perfect square
        # a = s^2 / (M * 210) for some s
        # s^2 must be divisible by M*210
        # Let's find the squarefree part of M*210
        temp = M * 210
        sf = 1
        p = 2
        while p * p <= temp:
            cnt = 0
            while temp % p == 0:
                temp //= p
                cnt ^= 1
            if cnt:
                sf *= p
            p += 1
        if temp > 1:
            sf *= temp
        
        # a must have squarefree part = sf (mod squares)
        # So a = sf * t^2 for some t, and 210 < a < M
        # Also a * M * 210 = sf*t^2 * M * 210
        # = t^2 * sf * M * 210
        # sf * M * 210 must be a perfect square for the whole thing to be a square
        # Wait, that's circular. Let me just check directly.
        
        # a must be of the form sf * t^2 (times a square)
        # a * M * 210 = sf * t^2 * M * 210
        # For this to be a square, sf * M * 210 must be a square (since t^2 is already square)
        # But sf is the squarefree part of M*210, so sf * (M*210) / sf^2 ... 
        # Actually: M*210 = sf * q^2 for some q (since sf is squarefree part)
        # So a * M * 210 = a * sf * q^2
        # This is a square iff a * sf is a square
        # So a must be sf * u^2 for some u (since sf is squarefree)
        
        # So a = sf * u^2, 210 < a < M, a != M
        u_max = isqrt((M - 1) // sf) if sf > 0 else 0
        for u in range(1, u_max + 1):
            a = sf * u * u
            if a > M or a <= 210:
                continue
            if a == M:
                continue
            # Verify
            if _is_perfect_square(a * M * 210):
                if M < best_max:
                    best_max = M
                    best_result = [a, M]
                break
    
    if best_result is not None:
        return [int(x) for x in best_result]
    
    # Try triples for small M
    # For M from 211 to 300, try all triples
    # This is O(M^3) which is too slow for M=300
    # Instead, for each M, try pairs (a,b) with a<b<M, check if a*b*M*210 is square
    # Still O(M^2) per M
    
    # Let's try a different approach: search for small solutions
    # known: 840 works (single), max-210=630
    # Can we do better?
    
    # Try M from 211 to 840 with the pair method above (already done up to 2000)
    # If nothing found, try triples for M up to ~300
    
    for M in range(213, 300):
        if time.time() > deadline:
            break
        if M >= best_max:
            break
        # Try triples (a, b, M) with 211 <= a < b < M
        # a*b*M*210 must be square
        # a*b must have squarefree part = squarefree part of M*210 (complement)
        temp = M * 210
        sf = 1
        p = 2
        while p * p <= temp:
            cnt = 0
            while temp % p == 0:
                temp //= p
                cnt ^= 1
            if cnt:
                sf *= p
            p += 1
        if temp > 1:
            sf *= temp
        
        # a*b must have squarefree part = sf
        # So a*b = sf * v^2 for some v
        # Try a from 211 to M-1, b from a+1 to M-1
        # Check if a*b has squarefree part = sf
        # Equivalent: a*b*sf is a perfect square (since sf is squarefree)
        # Wait: sf(a*b) = sf means a*b*sf is a square? 
        # sf(a*b) = sf => a*b = sf * square => a*b*sf = sf^2 * square = square. Yes!
        
        for a in range(211, M - 1):
            if time.time() > deadline:
                break
            for b in range(a + 1, M):
                if _is_perfect_square(a * b * sf):
                    # Then a*b*M*210 = a*b*sf*q^2 where M*210=sf*q^2
                    # = (a*b*sf) * q^2 * ... wait
                    # M*210 = sf * q^2
                    # a*b*M*210 = a*b*sf*q^2
                    # If a*b*sf is a square, say r^2, then a*b*M*210 = r^2*q^2 = (r*q)^2. Square!
                    if M < best_max:
                        best_max = M
                        best_result = [a, b, M]
                    break
            if best_result and best_max == M:
                break
    
    if best_result is not None:
        return [int(x) for x in best_result]
    
    # Fallback
    return [840]

# EVOLVE-BLOCK-END
