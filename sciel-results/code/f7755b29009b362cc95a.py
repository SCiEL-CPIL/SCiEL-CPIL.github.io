import time
import math

def get_divisors(n):
    """Get all divisors of n. Assumes n is positive integer."""
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    return divs

def construct():
    deadline = time.time() + 75.0
    answers = [None] * 254
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        found = None
        
        # x must satisfy 1/x < 4/n => x > n/4
        # Also 1/x > 4/n - 2/x is not right.
        # 1/x + 1/y + 1/z = 4/n
        # If x < y < z, then 3/x > 4/n => x < 3n/4
        # And 1/x < 4/n => x > n/4
        x_min = n // 4 + 1
        x_max = int(3 * n / 4) + 1
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            b2 = b * b
            
            # We need (a*y - b)(a*z - b) = b^2
            # Let d be a divisor of b^2.
            # y = (d + b) / a
            # z = (b^2/d + b) / a
            # We assume y < z, so d < b^2/d => d < b.
            # So we only need to check divisors d of b^2 such that d < b.
            # Actually, if d > b, then y > z, which is symmetric.
            # So iterating d < b is sufficient if we swap y,z.
            
            # Get divisors of b^2 that are less than b
            # Instead of getting all divisors, we can iterate d from 1 to b-1?
            # No, b can be ~50000, b^2 ~ 2.5e9.
            # Number of divisors is small.
            
            # Efficiently find divisors of b^2 less than b
            # Factor b
            temp_b = b
            factors = {}
            d = 2
            while d * d <= temp_b:
                while temp_b % d == 0:
                    factors[d] = factors.get(d, 0) + 1
                    temp_b //= d
                d += 1
            if temp_b > 1:
                factors[temp_b] = factors.get(temp_b, 0) + 1
            
            # Generate divisors of b^2
            # b = p1^e1 * p2^e2 ...
            # b^2 = p1^(2e1) * p2^(2e2) ...
            # We need divisors D of b^2 such that D < b.
            # And (D + b) % a == 0 and (b^2 // D + b) % a == 0.
            
            # List of prime factors with exponents for b^2
            prime_powers = []
            for p, e in factors.items():
                prime_powers.append([p, 2 * e])
            
            # Backtracking to generate divisors
            divisors = [1]
            for p, e in prime_powers:
                new_divs = []
                current_power = 1
                for i in range(e + 1):
                    for d in divisors:
                        new_divs.append(d * current_power)
                    current_power *= p
                divisors = new_divs
            
            # Filter divisors
            # We need d such that d < b (to ensure y < z if we assign y=(d+b)/a)
            # Wait, if d < b, then d < b^2/d, so (d+b)/a < (b^2/d+b)/a => y < z.
            # So we just check d < b.
            
            for d in divisors:
                if d >= b:
                    continue
                
                num_y = d + b
                if num_y % a != 0:
                    continue
                y = num_y // a
                
                num_z = b2 // d + b
                if num_z % a != 0:
                    continue
                z = num_z // a
                
                if y == x or z == x or y == z:
                    continue
                if y <= 0 or z <= 0:
                    continue
                if y > 1000000000000 or z > 1000000000000:
                    continue
                
                # Verify
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
