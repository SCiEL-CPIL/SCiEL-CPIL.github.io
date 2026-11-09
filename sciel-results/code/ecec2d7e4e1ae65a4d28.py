import time
import math

def construct():
    answers = []
    deadline = time.time() + 55  # Safety margin before 60s limit
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        x_start = n // 4 + 1
        x_end = (3 * n) // 4
        
        for x in range(x_start, x_end + 1):
            if time.time() > deadline:
                break
            
            a = 4 * x - n
            b = n * x
            
            if a <= 0:
                continue
            
            # (a*y - b)(a*z - b) = b^2
            # Let d = a*y - b, then d | b^2, d > 0, and d < b^2/d (i.e., d < b) for y < z
            # y = (d + b) / a must be integer
            # z = (b^2/d + b) / a must be integer
            # Need y > x, z > y, all distinct, z <= 10^12
            
            # Find divisors d of b^2 where d < b
            # b^2 can be up to ~2.4*10^9, so factorization is feasible
            
            # Factor b^2 to get all divisors
            # b = n * x, with n <= 256, x <= 192, so b <= ~49152
            # b^2 <= ~2.4 * 10^9
            
            # Get prime factorization of b
            b_val = b
            factors = {}
            temp = b_val
            p = 2
            while p * p <= temp:
                while temp % p == 0:
                    factors[p] = factors.get(p, 0) + 1
                    temp //= p
                p += 1 if p == 2 else 2
            if temp > 1:
                factors[temp] = factors.get(temp, 0) + 1
            
            # b^2 has factors with doubled exponents
            # Generate all divisors of b^2 that are < b
            # Number of divisors of b^2 is product of (2*e_i + 1)
            # For b up to 49152, this is manageable
            
            # Collect all divisors of b^2 less than b
            # Use recursive generation
            prime_powers = []
            for p, e in factors.items():
                powers = [p**(2*j) for j in range(2*e + 1)]
                prime_powers.append(powers)
            
            # Generate divisors < b
            divisors_lt_b = []
            def gen_divs(idx, current):
                if current >= b:
                    return
                if idx == len(prime_powers):
                    divisors_lt_b.append(current)
                    return
                for pp in prime_powers[idx]:
                    if current * pp >= b:
                        break
                    gen_divs(idx + 1, current * pp)
            
            gen_divs(0, 1)
            
            for d in divisors_lt_b:
                if time.time() > deadline:
                    break
                # y = (d + b) / a
                if (d + b) % a != 0:
                    continue
                y = (d + b) // a
                if y <= x:
                    continue
                # z = (b^2/d + b) / a
                b2_over_d = (b * b) // d
                if (b2_over_d + b) % a != 0:
                    continue
                z = (b2_over_d + b) // a
                if z <= y:
                    continue
                if z > 10**12:
                    continue
                # Check distinctness
                if x != y and y != z and x != z:
                    # Verify by cross multiplication
                    lhs = n * (y * z + x * z + x * y)
                    rhs = 4 * x * y * z
                    if lhs == rhs:
                        found = [x, y, z]
                        break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
