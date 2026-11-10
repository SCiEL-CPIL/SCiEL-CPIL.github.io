def construct():
    MAX_DENOM = 10**12
    answers = []
    
    def factorize(num):
        factors = {}
        d = 2
        while d * d <= num:
            while num % d == 0:
                factors[d] = factors.get(d, 0) + 1
                num //= d
            d += 1
        if num > 1:
            factors[num] = factors.get(num, 1)
        return factors
    
    def get_divisors_from_factors(factors):
        if not factors:
            return [1]
        primes = list(factors.keys())
        divisors = [1]
        for p in primes:
            e = factors[p]
            new_divs = []
            for d in divisors:
                for k in range(e + 1):
                    new_divs.append(d * (p ** k))
            divisors = new_divs
        return divisors
    
    for n in range(3, 257):
        found = None
        # A = 4x - n, so x = (A + n) // 4, need A > 0 and (A+n) % 4 == 0
        # Also x <= 3n/4 => A <= 2n
        # Try small A values first
        rem = (-n) % 4
        for A in range(rem, 2 * n + 1, 4):
            if A <= 0:
                continue
            x = (A + n) // 4
            if x < 1 or x > MAX_DENOM:
                continue
            # C = n^2 * x^2
            C = n * n * x * x
            # Factor C
            fac_n = factorize(n)
            fac_x = factorize(x)
            fac_C = {}
            all_primes = set(fac_n.keys()) | set(fac_x.keys())
            for p in all_primes:
                exp = 2 * fac_n.get(p, 0) + 2 * fac_x.get(p, 0)
                if exp > 0:
                    fac_C[p] = exp
            
            divisors = get_divisors_from_factors(fac_C)
            
            for u in divisors:
                v = C // u
                num_y = u + n * x
                num_z = v + n * x
                if num_y % A != 0 or num_z % A != 0:
                    continue
                y = num_y // A
                z = num_z // A
                if y <= 0 or z <= 0:
                    continue
                if y > MAX_DENOM or z > MAX_DENOM:
                    continue
                if x == y or x == z or y == z:
                    continue
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            if found:
                break
        answers.append(found)
    
    return answers
