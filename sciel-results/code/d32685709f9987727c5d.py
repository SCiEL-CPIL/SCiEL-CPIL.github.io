"""Rewritten program with stronger search for unit-fraction representations."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    answers = []
    deadline = time.time() + 75  # 75s safety margin before 90s kill
    BOUND = 10**12
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        target = 4.0 / n
        
        # Try multiple x values: x = n//4 + k for small k
        # x must satisfy 1/x < 4/n, so x > n/4
        base_x = n // 4 + 1
        if n % 4 == 0:
            base_x = n // 4 + 1  # x > n/4 strictly
        
        for k in range(0, 200):
            if time.time() > deadline:
                break
            x = base_x + k
            if x > BOUND:
                break
            
            # We need 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
            a = 4 * x - n
            b = n * x
            
            if a <= 0:
                continue
            
            # Reduce a/b
            g = math.gcd(a, b)
            a_red = a // g
            b_red = b // g
            
            # For 1/y + 1/z = a_red / b_red:
            # y + z = a_red * y * z / b_red
            # b_red * (y + z) = a_red * y * z
            # Let y = b_red * t / a_red ... use divisor method
            # Standard: y = (b_red + d) / a_red, z = b_red * (b_red + d) / (a_red * d)
            # where d divides b_red^2
            # Actually: from b(y+z) = az, rearrange:
            # a*yz - b*y - b*z = 0
            # (a*y - b)(a*z - b) = b^2
            # So let u = a*y - b, v = a*z - b, then u*v = b^2
            # y = (u + b)/a, z = (v + b)/a where u*v = b^2
            
            b_sq = b_red * b_red
            
            # Enumerate divisors of b_sq
            # For efficiency, iterate d from 1 to sqrt(b_sq)
            limit = int(math.isqrt(b_sq))
            
            for d in range(1, limit + 1):
                if time.time() > deadline:
                    break
                if b_sq % d != 0:
                    continue
                d2 = b_sq // d
                # u = d, v = d2 (or swap)
                # y = (d + b_red) / a_red, z = (d2 + b_red) / a_red
                num1 = d + b_red
                num2 = d2 + b_red
                if num1 % a_red != 0 or num2 % a_red != 0:
                    continue
                y = num1 // a_red
                z = num2 // a_red
                if y < 1 or z < 1:
                    continue
                if y > BOUND or z > BOUND:
                    continue
                # Ensure distinctness: x, y, z all distinct
                if x == y or x == z or y == z:
                    continue
                # Ensure positive
                if y <= 0 or z <= 0:
                    continue
                # Verify the equation
                # 1/x + 1/y + 1/z == 4/n
                # Cross multiply: n*(y*z + x*z + x*y) == 4*x*y*z
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
