"""Rewritten program using divisor enumeration for unit-fraction search."""
import time
from math import gcd

def _factorize(m):
    """Return dict of prime->exponent for m."""
    if m <= 1:
        return {}
    factors = {}
    d = 2
    while d * d <= m:
        while m % d == 0:
            factors[d] = factors.get(d, 0) + 1
            m //= d
        d += 1
    if m > 1:
        factors[m] = factors.get(m, 0) + 1
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from a prime factorization dict."""
    divs = [1]
    for p, e in factors.items():
        new_divs = []
        for d in divs:
            pe = 1
            for _ in range(e):
                pe *= p
                new_divs.append(d * pe)
        divs.extend(new_divs)
    return divs

def _all_divisors(m):
    """Return sorted list of all positive divisors of m."""
    if m <= 0:
        return []
    f = _factorize(m)
    divs = _divisors_from_factors(f)
    divs.sort()
    return divs

def construct():
    deadline = time.time() + 85.0
    answers = []
    BOUND = 10**12
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        x_min = n // 4 + 1
        
        # We need 1/x + 1/y + 1/z = 4/n with x < y < z distinct.
        # For a given x: 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
        # Let a = 4x - n, b = nx.
        # 1/y + 1/z = a/b  =>  (ay - b)(az - b) = b^2
        # Let A = ay - b, B = az - b. Then A*B = b^2, A < B (since y < z).
        # y = (A + b)/a, z = (B + b)/a.
        # We need: A divides b^2, A < b (so A < B), 
        #   (A + b) % a == 0, (B + b) % a == 0,
        #   y > x, z > y, all <= BOUND, all distinct.
        
        # Strategy: iterate x from x_min upward. For each x, factor b^2 and
        # enumerate divisors A of b^2 with A < b. Check conditions.
        # But b^2 can be huge. Better: iterate x, and for each x, iterate
        # divisors A of b^2 that are < b.
        
        # Alternative faster approach: for each x, the valid y range is
        # y_min = max(x+1, b//a + 1) to y_max = (2b + a - 1)//a - 1.
        # This range can be large for small x. Use divisor enumeration instead.
        
        for x in range(x_min, 20000):
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # y must satisfy: b/a < y < 2b/a
            # y_min = max(x+1, b//a + 1)
            # y_max = (2b + a - 1) // a - 1
            
            y_min = b // a + 1
            if y_min <= x:
                y_min = x + 1
            y_max = (2 * b + a - 1) // a - 1
            
            if y_min > y_max:
                continue
            
            # Use divisor enumeration of b^2
            # A * B = b^2, A < B => A < b
            # A = ay - b, so A ≡ -b (mod a), i.e., A + b ≡ 0 (mod a)
            # A >= a*y_min - b (from y >= y_min)
            # A <= a*y_max - b (from y <= y_max)
            # Also A < b
            
            # Factor b^2
            # b = n * x, so b^2 = n^2 * x^2
            # Factor n^2 and x^2 separately, combine
            f_n = _factorize(n)
            f_x = _factorize(x)
            f_b2 = {}
            for p, e in f_n.items():
                f_b2[p] = f_b2.get(p, 0) + 2 * e
            for p, e in f_x.items():
                f_b2[p] = f_b2.get(p, 0) + 2 * e
            
            divs = _divisors_from_factors(f_b2)
            divs.sort()
            
            # Find divisors A of b^2 such that:
            #   A < b
            #   A + b ≡ 0 (mod a)
            #   y = (A + b) // a is in [y_min, y_max]
            #   z = (b^2 // A + b) // a is integer, z > y, z <= BOUND
            
            b2 = b * b
            found_for_x = False
            
            for A in divs:
                if A >= b:
                    break
                # Check A + b ≡ 0 (mod a)
                if (A + b) % a != 0:
                    continue
                y = (A + b) // a
                if y < y_min or y > y_max:
                    continue
                B = b2 // A
                if (B + b) % a != 0:
                    continue
                z = (B + b) // a
                if z <= y:
                    continue
                if z > BOUND:
                    continue
                if x != y and y != z and x != z:
                    # Verify: 1/x + 1/y + 1/z = 4/n
                    # Cross multiply: n*(y*z + x*z + x*y) = 4*x*y*z
                    lhs = n * (y * z + x * z + x * y)
                    rhs = 4 * x * y * z
                    if lhs == rhs:
                        found = [x, y, z]
                        found_for_x = True
                        break
            
            if found:
                break
            # If x gets too large, the range for y shrinks; eventually no solutions
            # Also, if b//a grows too large, factorization becomes expensive
            if b > 10**8:
                break
        
        answers.append(found)
    
    return answers
