"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    answers = []
    LIMIT = 10**12
    
    # Precompute smallest prime factors up to max possible B
    # Max n = 256, max x < 3*256/4 = 192. Max B = 256 * 192 = 49152.
    MAX_B = 50000
    spf = list(range(MAX_B + 1))
    for i in range(2, int(MAX_B**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_B + 1, i):
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

    def get_divisors(factors):
        # factors is dict {p: e} for B^2
        # We pass factors of B, then double exponents
        divisors = [1]
        for p, e in factors.items():
            e2 = 2 * e
            new_divs = []
            for d in divisors:
                for i in range(e2 + 1):
                    new_divs.append(d * (p ** i))
            divisors = new_divs
        return divisors

    for n in range(3, 257):
        found = None
        start_x = n // 4 + 1
        end_x = (3 * n) // 4
        
        for x in range(start_x, end_x + 1):
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
                
            # Factorize B
            b_factors = factorize(B)
            # Generate divisors of B^2
            divs = get_divisors(b_factors)
            
            for u in divs:
                v = (B * B) // u
                if u > v:
                    continue
                    
                # Check congruences
                # u = A*y - B => u + B = A*y => u + B must be divisible by A
                # v = A*z - B => v + B = A*z => v + B must be divisible by A
                if (u + B) % A != 0:
                    continue
                if (v + B) % A != 0:
                    continue
                
                y = (u + B) // A
                z = (v + B) // A
                
                # Check distinctness and limits
                if y == x or z == x or y == z:
                    continue
                if y > LIMIT or z > LIMIT:
                    continue
                
                # We found a solution
                found = [x, y, z]
                break
            if found:
                break
        answers.append(found)
    return answers
# EVOLVE-BLOCK-END
