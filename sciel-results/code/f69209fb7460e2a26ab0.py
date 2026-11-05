import time
import math

def construct():
    # Deadline guard: must finish before 90s. We use 85s to be safe.
    deadline = time.time() + 85.0
    answers = [None] * 254
    
    # Pre-compute SPF (Smallest Prime Factor) up to a reasonable limit.
    # Max B = n * x_max = 256 * (3*256//4) = 256 * 192 = 49152.
    # B^2 can be up to ~2.4e9.
    # We only need to factorize B, not B^2 directly, then square the exponents.
    # Max B is ~50k. So SPF up to 100k is sufficient and fast.
    LIMIT = 100_000
    spf = list(range(LIMIT + 1))
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def factorize(n):
        """Returns dict of prime: exponent for n"""
        if n <= 1:
            return {}
        factors = {}
        while n > 1:
            p = spf[n]
            cnt = 0
            while n % p == 0:
                n //= p
                cnt += 1
            factors[p] = cnt
        return factors

    def generate_divisors_from_factors(factors):
        """
        Generates divisors from a dictionary of prime: exponent.
        """
        divs = [1]
        for p, exp in factors.items():
            new_divs = []
            p_pow = 1
            for _ in range(exp + 1):
                for d in divs:
                    new_divs.append(d * p_pow)
                p_pow *= p
            divs = new_divs
        return divs

    # Iterate through n = 3..256
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
            
        found = None
        
        # x bounds:
        # 1/x + 1/y + 1/z = 4/n
        # Assume x < y < z
        # 3/x > 4/n => x < 3n/4
        # 1/x < 4/n => x > n/4
        # So x is in [floor(n/4) + 1, floor(3n/4)]
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # If x_min > x_max, no solution in this range (shouldn't happen for n>=3)
        if x_min > x_max:
            answers[idx] = None
            continue

        # To find a solution quickly, we iterate x.
        # Strategy: Try x values that make A = 4x - n small.
        # Small A implies larger y, z, which are easier to fit in bounds and distinct.
        # However, we must ensure y > x.
        # Let's sort candidates by A (ascending).
        
        candidates = []
        for x in range(x_min, x_max + 1):
            A = 4 * x - n
            if A > 0:
                candidates.append((A, x))
        
        # Sort by A
        candidates.sort()
        
        for A, x in candidates:
            if time.time() > deadline:
                break
                
            B = n * x
            # We need d | B^2 such that:
            # y = (d + B) / A is integer
            # z = (B^2/d + B) / A is integer
            # x < y < z
            # y > x => d + B > A*x => d > A*x - B
            # z > y => B^2/d + B > d + B => B^2/d > d => d^2 < B^2 => d < B
            # Also d must be positive integer.
            
            D_min = A * x - B + 1
            D_max = B - 1
            
            if D_min > D_max:
                continue
            
            # Factorize B
            fac_B = factorize(B)
            # Factorize B^2
            fac_B2 = {p: 2*e for p, e in fac_B.items()}
            
            # Generate divisors of B^2
            # Optimization: We only care about divisors in [D_min, D_max].
            # Generating all divisors is fast for small B.
            divs = generate_divisors_from_factors(fac_B2)
            
            # Filter and check
            for d in divs:
                if d < D_min or d > D_max:
                    continue
                
                # Check divisibility for y
                num_y = d + B
                if num_y % A != 0:
                    continue
                y = num_y // A
                
                # Check divisibility for z
                d_prime = (B * B) // d
                num_z = d_prime + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                
                # Verify distinctness and order (should be guaranteed by math, but double check)
                if x < y < z:
                    # Check bounds
                    if y <= 10**12 and z <= 10**12:
                        found = [x, y, z]
                        break
                        
            if found:
                break
        
        answers[idx] = found

    return answers
