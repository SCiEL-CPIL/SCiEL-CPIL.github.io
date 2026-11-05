"""Direct approach for distinct three-term unit-fraction representations of 4/n using factorization."""
import time
from math import gcd

def _factorize(n):
    """Simple trial division factorization."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def _gen_divisors(prime_factors):
    """Generate all divisors from prime factor list."""
    # Count prime factors
    from collections import Counter
    cnt = Counter(prime_factors)
    # Generate divisors
    divisors = [1]
    for p, exp in cnt.items():
        new_divs = []
        p_pow = 1
        for e in range(exp + 1):
            for d in divisors:
                new_divs.append(d * p_pow)
            p_pow *= p
        divisors = new_divs
    return divisors

def construct():
    deadline = time.time() + 85.0
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x must satisfy 1/x < 4/n, so x > n/4
        x_min = n // 4 + 1
        
        for x in range(x_min, x_min + 20000):
            if time.time() > deadline:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # We need 1/y + 1/z = a/b
            # (ay - b)(az - b) = b^2
            # Let d = ay - b, then d must divide b^2 and d ≡ -b (mod a)
            # y = (d + b) / a must be positive integer
            # z = (b^2/d + b) / a must be positive integer
            
            # d must be > 0 (since y > 0 means d + b > 0, and we want y > x so d > a*x - b = 4x - n - nx... let's just check)
            # Also y > x means (d+b)/a > x => d + b > a*x => d > a*x - b = (4x-n)*x - nx = 4x^2 - nx - nx = 4x^2 - 2nx
            # And z > y
            
            # To avoid huge b^2, let's think differently.
            # b^2 can be very large. Let's use the fact that d must divide b^2.
            # Factor b, then factor b^2 by doubling exponents.
            
            # But b can be up to ~256 * 2500 = 640000, so b^2 ~ 4e11, factorization is feasible
            # but generating all divisors might be too many.
            
            # Alternative: iterate d over divisors of b^2 that are ≡ -b mod a
            # Number of divisors of b^2 could be large but manageable for b < 10^6
            
            # Let's try a different approach: iterate over possible y in a smart range
            # y must satisfy: y > x, and 1/y + 1/z = a/b with z > y
            # So 2/y > a/b => y < 2b/a
            # And 1/y < a/b => y > b/a
            # So y is in (b/a, 2b/a)
            
            y_lo = max(x + 1, b // a + 1)
            y_hi = 2 * b // a
            
            if y_hi <= y_lo:
                continue
            
            # The range is about b/a in size. b/a = nx/(4x-n)
            # For small x near n/4, a is small, so b/a is large.
            # For larger x, a grows, b/a shrinks.
            
            # Let's check: for x = n//4 + 1, a = 4*(n//4+1) - n = 4 - (n mod 4) if n%4==0, else...
            # Actually a = 4x - n, x = n//4 + 1, so a = 4*(n//4 + 1) - n = 4*floor(n/4) + 4 - n
            # If n = 4q: a = 4q + 4 - 4q = 4
            # If n = 4q+1: a = 4q + 4 - 4q - 1 = 3
            # If n = 4q+2: a = 4q + 4 - 4q - 2 = 2
            # If n = 4q+3: a = 4q + 4 - 4q - 3 = 1
            
            # So a is small (1-4) for x near n/4, meaning b/a is large.
            # This makes the y range huge.
            
            # Better to use factorization for these cases.
            # Let's use factorization when the range is too large.
            
            range_size = y_hi - y_lo + 1
            if range_size > 50000:
                # Use factorization method
                # Factor b
                b_fac = _factorize(b)
                # b^2 has prime factors with doubled exponents
                from collections import Counter
                cnt_b = Counter(b_fac)
                divisors_b2 = [1]
                for p, exp in cnt_b.items():
                    new_divs = []
                    p_pow = 1
                    for e in range(2 * exp + 1):
                        for d in divisors_b2:
                            new_divs.append(d * p_pow)
                        p_pow *= p
                    divisors_b2 = new_divs
                
                # For each divisor d of b^2, check if (d + b) % a == 0 and (b^2//d + b) % a == 0
                b_sq = b * b
                for d in divisors_b2:
                    if (d + b) % a != 0:
                        continue
                    y_val = (d + b) // a
                    if y_val <= x or y_val > 10**12:
                        continue
                    rem = b_sq // d
                    if (rem + b) % a != 0:
                        continue
                    z_val = (rem + b) // a
                    if z_val <= y_val or z_val > 10**12:
                        continue
                    if x != y_val and y_val != z_val and x != z_val:
                        found = [x, y_val, z_val]
                        break
            else:
                # Direct iteration over y
                for y in range(y_lo, y_hi + 1):
                    if time.time() > deadline:
                        break
                    num = a * y - b
                    if num <= 0:
                        continue
                    den = b * y
                    if den % num != 0:
                        continue
                    z = den // num
                    if z <= y or z > 10**12:
                        continue
                    if x != y and y != z and x != z:
                        found = [x, y, z]
                        break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
