import time
import math

def construct():
    deadline = time.time() + 50.0  # Safety margin
    answers = [None] * 254
    
    # Precompute Smallest Prime Factor (SPF)
    # Max B = n * x. n <= 256. x <= 3n/4 <= 192.
    # Max B = 256 * 192 = 49152.
    # Let's use 100,000 to be safe and fast.
    LIMIT = 100_000
    spf = list(range(LIMIT + 1))
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if spf[i] == i:
            for j in range(i*i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(n):
        factors = {}
        while n > 1:
            p = spf[n]
            cnt = 0
            while n % p == 0:
                n //= p
                cnt += 1
            factors[p] = cnt
        return factors
        
    def generate_divisors(factors):
        divs = [1]
        for p, exp in factors.items():
            new_divs = []
            for d in divs:
                val = 1
                for _ in range(exp + 1):
                    new_divs.append(d * val)
                    val *= p
            divs = new_divs
        return divs

    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
            
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Iterate x in natural order
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
                
            B = n * x
            B_sq = B * B
            
            # D_min condition for y > x
            # y = (d + B) / A > x  => d > Ax - B
            # Ax - B = x(4x - n) - nx = 4x^2 - 2nx
            D_min = 4 * x * x - 2 * n * x + 1
            
            if D_min >= B:
                continue
                
            # Factorize B
            if B > LIMIT:
                # Fallback for large B (unlikely given limits, but safe)
                # Just skip or use trial division? 
                # With LIMIT 100k and max B 49k, this shouldn't happen.
                continue
                
            factors_B = factorize(B)
            factors_B_sq = {p: 2*e for p, e in factors_B.items()}
            divs = generate_divisors(factors_B_sq)
            
            # Check divisors
            for d in divs:
                if d < D_min or d >= B:
                    continue
                    
                num_y = d + B
                if num_y % A != 0:
                    continue
                y = num_y // A
                
                d_prime = B_sq // d
                num_z = d_prime + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                
                if y > x and z > y:
                    if y <= 10**12 and z <= 10**12:
                        found = [x, y, z]
                        break
                        
            if found:
                break
                
        answers[idx] = found

    return answers
