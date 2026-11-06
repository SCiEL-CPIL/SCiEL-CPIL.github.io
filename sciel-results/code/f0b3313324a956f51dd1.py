import time
import math

def construct():
    deadline = time.time() + 75.0
    answers = [None] * 254
    BOUND = 10**12
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        found = None
        
        # x must satisfy x > n/4, so x_min = n//4 + 1
        # Also x < y < z, and 1/x + 1/y + 1/z = 4/n
        # Since y > x and z > y, we have 4/n < 3/x, so x < 3n/4
        # More tightly: 4/n - 1/x = (4x-n)/(nx) must be positive, so x > n/4
        # And 1/y + 1/z < 2/x (since y,x > x... wait y > x so 1/y < 1/x)
        # Actually 1/y + 1/z < 2/x, so (4x-n)/(nx) < 2/x => 4x - n < 2n => 4x < 3n => x < 3n/4
        
        x_min = n // 4 + 1
        x_max = min(3 * n // 4, BOUND)  # x < 3n/4 from above
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            a = 4 * x - n  # positive since x > n/4
            b = n * x
            
            # Need (ay - b)(az - b) = b^2
            # Let d = ay - b, so d divides b^2, d < b (for y < z)
            # y = (d + b) / a must be integer, y > x
            # z = (b^2/d + b) / a must be integer, z > y
            
            b2 = b * b
            
            # d must divide b^2, d < b, and (d + b) % a == 0
            # Also y = (d+b)/a > x => d + b > a*x => d > a*x - b = (4x-n)*x - nx = 4x^2 - nx - nx = 4x^2 - 2nx
            d_min = max(1, 4 * x * x - 2 * n * x + 1)  # d > 4x^2 - 2nx, so d >= 4x^2 - 2nx + 1
            if d_min < 1:
                d_min = 1
            
            # Find divisors of b^2 in range [d_min, b-1]
            # b^2 can be very large, so we need to be smart
            # b = n*x, for n up to 256 and x up to ~3n/4, b is at most ~256*192 ~ 49152
            # b^2 is at most ~2.4 * 10^9, which is manageable for factoring
            
            # Factor b to get divisors of b^2
            # b = n*x
            # Let's factor b
            temp = b
            factors = {}
            f = 2
            while f * f <= temp:
                while temp % f == 0:
                    factors[f] = factors.get(f, 0) + 1
                    temp //= f
                f += 1
            if temp > 1:
                factors[temp] = factors.get(temp, 0) + 1
            
            # Generate divisors of b^2: each prime p^e in b becomes p^(2e) in b^2
            # Number of divisors of b^2 = product(2*e_i + 1)
            # This can be large, so let's be careful
            
            # Actually, for small x values this should work fine
            # Let's generate divisors of b^2 that are in [d_min, b-1]
            
            # Get prime factorization of b^2
            prime_powers = []
            for p, e in factors.items():
                prime_powers.append((p, 2 * e))
            
            # Generate all divisors of b^2
            divisors = [1]
            for p, exp in prime_powers:
                new_divs = []
                for d in divisors:
                    val = d
                    for i in range(exp + 1):
                        new_divs.append(val)
                        val *= p
                divisors = new_divs
            
            # Filter divisors in [d_min, b-1]
            for d in divisors:
                if d < d_min or d >= b:
                    continue
                # Check (d + b) % a == 0
                if (d + b) % a != 0:
                    continue
                y = (d + b) // a
                if y <= x:
                    continue
                # Check (b^2//d + b) % a == 0
                bz = b2 // d  # This is integer since d divides b^2
                if (bz + b) % a != 0:
                    continue
                z = (bz + b) // a
                if z <= y:
                    continue
                if z > BOUND:
                    continue
                # All distinct: x < y < z ensures distinctness
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
