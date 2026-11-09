import time
import math

def construct():
    answers = [None] * 254
    t0 = time.time()
    DEADLINE = 85.0

    # Precompute smallest prime factor for numbers up to max possible b
    # Max n=256, x <= 3*256/4 = 192. b = n*x <= 256*192 = 49152.
    MAX_B = 50000
    spf = list(range(MAX_B + 1))
    for i in range(2, int(MAX_B**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_B + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_divisors_sq(b):
        # Factorize b
        factors = {}
        n = b
        while n > 1:
            p = spf[n]
            cnt = 0
            while n % p == 0:
                n //= p
                cnt += 1
            factors[p] = cnt
        
        # Generate divisors of b^2
        # b^2 has prime factors p with exponent 2*cnt
        divs = [1]
        for p, cnt in factors.items():
            max_exp = 2 * cnt
            new_divs = []
            for d in divs:
                for e in range(max_exp + 1):
                    new_divs.append(d * (p ** e))
            divs = new_divs
        return divs

    for idx, n in enumerate(range(3, 257)):
        if time.time() - t0 > DEADLINE:
            break
        found = None
        x_min = n // 4 + 1
        x_max = 3 * n // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() - t0 > DEADLINE:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            b_sq = b * b
            
            # y = (u+b)/a, z = (v+b)/a with u*v = b^2
            # We need y > x and z > y.
            # y > x => (u+b)/a > x => u > ax - b
            # z > y => v > u (since a>0) => u < b (since v=b^2/u, v>u => u^2 < b^2)
            # So u is in (ax - b, b)
            
            lo = ax = a * x
            lo_bound = ax - b
            hi_bound = b
            
            # Get divisors of b^2
            # Optimization: Only generate divisors in range?
            # Generating all might be okay for b < 50000.
            # Max divisors for b<50000 is small (few hundreds).
            divs = get_divisors_sq(b)
            
            # Filter and check
            for u in divs:
                if u < lo_bound or u >= hi_bound:
                    continue
                if (u + b) % a != 0:
                    continue
                y = (u + b) // a
                if y <= x:
                    continue
                v = b_sq // u
                if (v + b) % a != 0:
                    continue
                z = (v + b) // a
                if z <= y:
                    continue
                if x == y or x == z or y == z:
                    continue
                if x > 1000000000000 or y > 1000000000000 or z > 1000000000000:
                    continue
                found = [x, y, z]
                break
            if found:
                break
        answers[idx] = found

    return answers
