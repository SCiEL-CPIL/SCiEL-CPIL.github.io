"""Solve 1/x + 1/y + 1/z = 4/n for n=3..256 with distinct positive denominators."""
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        
        # We need 1/x + 1/y + 1/z = 4/n with x < y < z (WLOG)
        # From 1/x < 4/n, we get x > n/4
        # From 1/x >= 4/n - 2/(x+1) - 2/(x+2) ... but simpler bounds:
        # Since y > x and z > y > x, we have 4/n < 3/x, so x < 3n/4
        # Also 4/n > 1/x + 1/(x+1) + 1/(x+2) is not quite right for upper bound on x
        # Actually, the maximum x is when y and z are as small as possible (x+1, x+2)
        # 4/n >= 1/x + 1/(x+1) + 1/(x+2) gives an upper bound on x, but it's complex.
        # A simpler upper bound: since y >= x+1, z >= x+2,
        # 4/n <= 1/x + 1/(x+1) + 1/(x+2) is NOT the right direction.
        # We have 4/n = 1/x + 1/y + 1/z. Since y > x, z > y, we have 1/y < 1/x, 1/z < 1/x.
        # So 4/n < 3/x => x < 3n/4.
        # For lower bound: 1/x < 4/n => x > n/4.
        
        x_min = n // 4 + 1
        # x must satisfy x < 3n/4, so x_max = (3*n - 1) // 4
        x_max = (3 * n - 1) // 4
        
        if x_min > x_max:
            answers.append(None)
            continue
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            if found:
                break
            
            # 4/n - 1/x = (4x - n) / (nx)
            # Let a = 4x - n, b = nx
            # We need 1/y + 1/z = a/b, with y > x, z > y
            a = 4 * x - n
            b = n * x
            
            if a <= 0:
                continue
            
            # 1/y + 1/z = a/b => b(y+z) = a*y*z => a*y*z - b*y - b*z = 0
            # Multiply by a: a^2*y*z - a*b*y - a*b*z = 0
            # (a*y - b)(a*z - b) = b^2
            # Let u = a*y - b, v = a*z - b, then u*v = b^2
            # y = (u + b) / a, z = (v + b) / a
            # We need y > x, so (u + b) / a > x => u + b > a*x = (4x-n)*x = 4x^2 - nx
            # u > 4x^2 - nx - b = 4x^2 - nx - nx = 4x^2 - 2nx
            # Also z > y means v > u (since v = a*z - b, u = a*y - b, and a > 0)
            # So we need u < v, u*v = b^2, u > 4x^2 - 2nx, and (u+b) % a == 0, (v+b) % a == 0
            
            # We need to find divisors u of b^2 such that:
            # 1. u > 4x^2 - 2nx
            # 2. (u + b) % a == 0
            # 3. (b^2 // u + b) % a == 0
            # 4. b^2 // u > u (i.e., v > u)
            # 5. y = (u+b)//a > x (should be guaranteed by condition 1 if a>0)
            # 6. z = (v+b)//a > y (guaranteed by v > u)
            # 7. y, z <= 10^12
            
            # b^2 can be very large. Let's factorize b and enumerate divisors.
            # b = n*x. For n up to 256 and x up to ~192, b is at most 256*192 ~ 49152.
            # b^2 is at most ~2.4 * 10^9, which is manageable for factorization.
            
            # Let's factorize b first, then b^2 has the same prime factors with doubled exponents.
            # Actually, let's just find all divisors of b^2.
            # b^2 can be up to (256 * 192)^2 ≈ 2.4e9. Number of divisors is manageable.
            
            # Factorize b
            def factorize(num):
                factors = {}
                d = 2
                while d * d <= num:
                    while num % d == 0:
                        factors[d] = factors.get(d, 0) + 1
                        num //= d
                    d += 1
                if num > 1:
                    factors[num] = factors.get(num, 0) + 1
                return factors
            
            # For b^2, we double the exponents
            # But we need divisors of b^2, not just b.
            # Let's get prime factorization of b, then b^2 has 2*exponents.
            
            # Actually, a more efficient approach: enumerate divisors of b^2 directly
            # by iterating up to sqrt(b^2) = b. But b can be ~50000, so b^2 ~ 2.5e9, sqrt is ~50000.
            # That's fine.
            
            # But wait, we need to be careful. Let's use the factorization approach.
            # Get factors of b, then generate all divisors of b^2.
            
            b_val = b
            # Factorize b_val
            temp = b_val
            prime_factors = {}
            d = 2
            while d * d <= temp:
                while temp % d == 0:
                    prime_factors[d] = prime_factors.get(d, 0) + 1
                    temp //= d
                d += 1
            if temp > 1:
                prime_factors[temp] = prime_factors.get(temp, 0) + 1
            
            # Now generate all divisors of b^2 = b_val^2
            # For each prime p with exponent e in b, it has exponent 2e in b^2
            # Generate all combinations
            
            def get_divisors_from_factorization(factors):
                # factors is a dict {prime: exponent}
                # For b^2, we double the exponents
                primes = list(factors.keys())
                exponents = [factors[p] * 2 for p in primes]
                
                divisors = [1]
                for i, p in enumerate(primes):
                    current_divs = list(divisors)
                    power = 1
                    for _ in range(exponents[i]):
                        power *= p
                        for d in current_divs:
                            divisors.append(d * power)
                return divisors
            
            # This could generate many divisors. Let's be smarter.
            # We only need u such that u > 4x^2 - 2nx and u < b^2/u (i.e., u < b)
            # Wait, u < v means u < b^2/u, so u^2 < b^2, so u < b.
            # And u > 4x^2 - 2nx.
            # So we need divisors u of b^2 with 4x^2 - 2nx < u < b.
            
            # Let's enumerate divisors up to b.
            # Actually, let's just iterate u from max(1, 4*x*x - 2*n*x + 1) to b-1,
            # but that's too slow if the range is large.
            
            # Better: generate divisors of b^2 that are less than b.
            # Since u*v = b^2 and u < v, we have u < b.
            # So we only need divisors u of b^2 with u < b.
            
            # Let's get all divisors of b^2 that are < b.
            # We can do this by iterating d from 1 to sqrt(b^2) = b, checking if d divides b^2.
            # But b can be ~50000, so iterating up to 50000 is fine.
            
            # Actually, let's just iterate d from 1 to int(b**0.5) and for each divisor,
            # we get two divisors d and b^2//d. We want those < b.
            
            # Hmm, but b^2//d < b iff d > b. So for d <= b, b^2//d >= b.
            # So the divisors < b are exactly the divisors d of b^2 with d < b.
            
            # Let me just iterate d from 1 to b-1 and check if b^2 % d == 0.
            # b is at most ~50000, so this is at most 50000 iterations per x.
            # x ranges from ~75 to ~192, so about 117 values of x.
            # 117 * 50000 = 5.85 million iterations per n. For 254 values of n, that's ~1.5 billion.
            # Too slow.
            
            # Let's use the factorization to generate divisors more efficiently.
            
            # Generate all divisors of b^2
            divisors_b2 = get_divisors_from_factorization(prime_factors)
            
            # Filter: u < b, u > 4x^2 - 2nx
            lower_bound = 4 * x * x - 2 * n * x
            for u in divisors_b2:
                if u >= b:
                    continue
                if u <= lower_bound:
                    continue
                v = b_val * b_val // u
                if v <= u:
                    continue
                # Check (u + b) % a == 0
                if (u + b_val) % a != 0:
                    continue
                if (v + b_val) % a != 0:
                    continue
                y = (u + b_val) // a
                z = (v + b_val) // a
                # Check distinctness and bounds
                if x == y or y == z or x == z:
                    continue
                if y <= x or z <= y:
                    continue
                if y > 10**12 or z > 10**12:
                    continue
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
