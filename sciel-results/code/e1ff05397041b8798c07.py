import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    LIMIT = 10**12
    
    # Precompute smallest prime factors up to 50000
    MAX_VAL = 50000
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(num):
        factors = {}
        while num > 1:
            p = spf[num]
            cnt = 0
            while num % p == 0:
                num //= p
                cnt += 1
            factors[p] = cnt
        return factors

    def get_divisors_pow2(factors):
        # factors is dict {prime: exponent}
        # We need divisors of b^2, so exponents are 2 * original_exponent
        # Generate all combinations
        divs = [1]
        for p, e in factors.items():
            max_exp = 2 * e
            new_divs = []
            for d in divs:
                for k in range(max_exp + 1):
                    new_divs.append(d * (p ** k))
            divs = new_divs
        return divs

    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # u = ay - b > 2x(2x - n)
            lower_bound = 2 * x * (2 * x - n)
            
            # Factorize b
            if b > MAX_VAL:
                # Fallback to simple loop if b is too large (shouldn't happen for n<=256, x<=192)
                # But just in case
                pass
            
            factors = factorize(b)
            divs = get_divisors_pow2(factors)
            
            # Sort divisors to check smaller u first? Or just iterate.
            # We want y, z distinct.
            # y = (u + b) / a
            # z = (b^2 / u + b) / a
            
            for u in divs:
                if u <= lower_bound:
                    continue
                
                if (u + b) % a != 0:
                    continue
                
                y = (u + b) // a
                
                if y <= x:
                    continue
                
                v = (b * b) // u
                if (v + b) % a != 0:
                    continue
                
                z = (v + b) // a
                
                if z <= y:
                    continue
                
                if y == x or z == x or y == z:
                    continue
                
                if y > LIMIT or z > LIMIT:
                    continue
                
                # Verify
                # 1/x + 1/y + 1/z == 4/n
                # n(yz + xz + xy) == 4xyz
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
