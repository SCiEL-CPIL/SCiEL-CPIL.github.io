import time

def construct():
    MAX_DENOM = 10**12
    DEADLINE = time.time() + 85  # 85s safety margin
    answers = [None] * 254  # for n=3..256
    
    def factorize(num):
        """Return prime factorization as dict {prime: exponent}"""
        factors = {}
        if num % 2 == 0:
            cnt = 0
            while num % 2 == 0:
                cnt += 1
                num //= 2
            factors[2] = cnt
        d = 3
        while d * d <= num:
            if num % d == 0:
                cnt = 0
                while num % d == 0:
                    cnt += 1
                    num //= d
                factors[d] = cnt
            d += 2
        if num > 1:
            factors[num] = factors.get(num, 0) + 1
        return factors
    
    def get_divisors(factors):
        """Generate all divisors from prime factorization"""
        divisors = [1]
        for prime, exp in factors.items():
            new_divs = []
            p_power = 1
            for e in range(exp + 1):
                for d in divisors:
                    new_divs.append(d * p_power)
                p_power *= prime
            divisors = new_divs
        return divisors
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > DEADLINE:
            break
        found = None
        # Try x values starting from small values
        # x must satisfy 1/x < 4/n, so x > n/4
        # Also x should be reasonably small for efficiency
        min_x = n // 4 + 1
        # Upper bound on x: 1/x + 1/y + 1/z = 4/n with x<=y<=z
        # So 3/x >= 4/n => x <= 3n/4
        # But we want distinct, so let's be more careful
        # Actually, for distinct x<y<z: 1/x > 1/y > 1/z
        # 1/x < 4/n (since other terms are positive)
        # And 3/x > 4/n => x < 3n/4
        max_x = min(3 * n // 4, 2 * n)  # reasonable upper bound
        
        for x in range(min_x, max_x + 1):
            if time.time() > DEADLINE:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            b_sq = b * b
            
            # Factorize b_sq
            # b = n*x, so b_sq = n^2 * x^2
            # We can factorize n and x separately
            factors_n = factorize(n)
            factors_x = factorize(x)
            
            # Combine: b_sq has factors from n^2 and x^2
            factors_b_sq = {}
            for p, e in factors_n.items():
                factors_b_sq[p] = factors_b_sq.get(p, 0) + 2 * e
            for p, e in factors_x.items():
                factors_b_sq[p] = factors_b_sq.get(p, 0) + 2 * e
            
            divisors = get_divisors(factors_b_sq)
            
            # For each divisor d of b_sq, check if d = a*y - b gives integer y
            # y = (d + b) / a must be integer
            # z = (b_sq/d + b) / a must be integer
            # Need x, y, z all distinct and positive, and <= MAX_DENOM
            for d in divisors:
                if time.time() > DEADLINE:
                    break
                # y = (d + b) / a
                num_y = d + b
                if num_y % a != 0:
                    continue
                y = num_y // a
                if y <= 0 or y > MAX_DENOM:
                    continue
                # z = (b_sq//d + b) / a
                other = b_sq // d
                num_z = other + b
                if num_z % a != 0:
                    continue
                z = num_z // a
                if z <= 0 or z > MAX_DENOM:
                    continue
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                # Verify the solution
                # 1/x + 1/y + 1/z should equal 4/n
                # Cross multiply: n*(y*z + x*z + x*y) == 4*x*y*z
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            if found:
                break
        
        answers[idx] = found
    
    return answers
