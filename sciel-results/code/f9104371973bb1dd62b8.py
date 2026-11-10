import time
from math import gcd

def construct():
    deadline = time.time() + 85  # 5s safety margin before 90s kill
    answers = [None] * 254  # indices 0..253 correspond to n=3..256

    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        # x must satisfy: 1/x < 4/n => x > n/4
        # and 3/x > 4/n => x < 3n/4 (since x is smallest, 1/x is largest term)
        # So x in [floor(n/4)+1, floor(3n/4)]
        x_start = n // 4 + 1
        x_end = (3 * n) // 4
        
        for x in range(x_start, x_end + 1):
            if time.time() > deadline:
                break
            a = 4 * x - n  # numerator of (4/n - 1/x)
            b = n * x      # denominator of (4/n - 1/x)
            if a <= 0:
                continue
            # 1/y + 1/z = a/b
            # (ay - b)(az - b) = b^2
            # Let d = ay - b, then d must divide b^2
            # y = (d + b) / a, z = (b^2/d + b) / a
            # Need y, z positive integers, y != z, y != x, z != x
            # y > x (since x is smallest), so d > 0 => d > b - b = 0, and y > x => (d+b)/a > x => d + b > a*x => d > a*x - b = (4x-n)*x - nx = 4x^2 - nx - nx = 4x^2 - 2nx
            # Actually simpler: y > x means (d+b)/a > x => d > a*x - b
            # a*x - b = (4x-n)*x - n*x = 4x^2 - nx - nx = 4x^2 - 2nx
            # Also need z > y (or at least z != y), and z <= 10^12
            
            # Enumerate divisors of b^2
            # b can be up to 256 * (3*256//4) = 256 * 192 = 49152, so b^2 ~ 2.4e9
            # Factor b, then enumerate divisors of b^2
            b_val = b
            # Factorize b_val
            factors = {}
            temp = b_val
            f = 2
            while f * f <= temp:
                while temp % f == 0:
                    factors[f] = factors.get(f, 0) + 1
                    temp //= f
                f += 1
            if temp > 1:
                factors[temp] = factors.get(temp, 0) + 1
            
            # Generate all divisors of b^2 from factors of b (each exponent doubled)
            # b^2 has prime factorization with exponents 2*e_p
            divisors = [1]
            for p, e in factors.items():
                exp_in_b2 = 2 * e
                current_divs = list(divisors)
                for i in range(1, exp_in_b2 + 1):
                    for d in current_divs:
                        divisors.append(d * (p ** i))
            
            # For each divisor d of b^2, check if it gives valid y, z
            # d = ay - b => y = (d + b) / a
            # z = (b^2 / d + b) / a
            # We want y > x and z > y (distinct, ordered)
            # y > x => d > a*x - b
            # Let's just iterate and check
            for d in divisors:
                if time.time() > deadline:
                    break
                # y = (d + b) / a must be integer
                num_y = d + b
                if num_y % a != 0:
                    continue
                y = num_y // a
                if y <= x:
                    continue
                # z = (b^2 // d + b) / a
                # Check b^2 % d == 0 (it should be since d divides b^2)
                b2 = b * b
                if b2 % d != 0:
                    continue
                num_z = b2 // d + b
                if num_z % a != 0:
                    continue
                z = num_z // a
                if z <= y:
                    continue
                if z > 10**12:
                    continue
                # Check distinctness
                if x == y or y == z or x == z:
                    continue
                # Verify: 1/x + 1/y + 1/z == 4/n
                # Cross multiply: n*(y*z + x*z + x*y) == 4*x*y*z
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            if found:
                break
        
        answers[idx] = found
    
    return answers
