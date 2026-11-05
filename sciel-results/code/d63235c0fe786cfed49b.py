import time
import math

def construct():
    deadline = time.time() + 55.0  # 55s to leave margin for 90s limit
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        
        # x range: x > n/4 (so 4x-n > 0), and x <= 3n/4 (from 4/n <= 3/x)
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # (ay - b)(az - b) = b^2
            # u * v = b^2, u = ay - b, v = az - b
            # y = (u + b) / a, z = (v + b) / a
            # Need: a | (u+b), a | (v+b), y > 0, z > 0, y != z, y != x, z != x
            
            # Factor b to get divisors of b^2
            b_temp = b
            factors = {}
            d = 2
            while d * d <= b_temp:
                while b_temp % d == 0:
                    factors[d] = factors.get(d, 0) + 1
                    b_temp //= d
                d += 1
            if b_temp > 1:
                factors[b_temp] = factors.get(b_temp, 0) + 1
            
            # Divisors of b^2: for each prime p^e in b, p^(2e) in b^2
            # Generate all divisors
            divisors = [1]
            for p, e in factors.items():
                new_divs = []
                p_power = 1
                for k in range(2 * e + 1):
                    for d0 in divisors:
                        new_divs.append(d0 * p_power)
                    p_power *= p
                divisors = new_divs
            
            # Sort divisors to allow early termination
            divisors.sort()
            
            b2 = b * b
            sqrt_b2 = b  # sqrt(b^2) = b
            
            # For each divisor u of b^2, check if it gives valid y, z
            # We want u <= v (i.e., u <= b) to avoid duplicate pairs
            for u in divisors:
                if u > sqrt_b2:
                    break
                v = b2 // u
                # y = (u + b) / a, z = (v + b) / a
                # Need (u + b) % a == 0 and (v + b) % a == 0
                if (u + b) % a != 0:
                    continue
                if (v + b) % a != 0:
                    continue
                y = (u + b) // a
                z = (v + b) // a
                # Check positivity
                if y <= 0 or z <= 0:
                    continue
                # Check distinctness: x, y, z all distinct
                if x == y or x == z or y == z:
                    continue
                # Check bounds
                if y > 10**12 or z > 10**12:
                    continue
                # Found a valid solution
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
        
        if time.time() > deadline:
            # Fill remaining with None
            while len(answers) < 254:
                answers.append(None)
            return answers
    
    return answers
