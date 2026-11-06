import time
from math import gcd

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254
    BOUND = 10**12
    
    # Precompute factorizations for B = n*x where x is in a reasonable range
    # We'll factor on demand but cache results
    factor_cache = {}
    
    def get_factors(num):
        if num in factor_cache:
            return factor_cache[num]
        temp = num
        factors = {}
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                factors[d] = factors.get(d, 0) + 1
                temp //= d
            d += 1
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        factor_cache[num] = factors
        return factors
    
    def generate_divisors_b_sq(factors):
        """Generate divisors of B^2 from prime factorization of B"""
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            for dv in divs:
                val = dv
                for _ in range(2 * e + 1):
                    new_divs.append(val)
                    val *= p
            divs = new_divs
        divs.sort()
        return divs
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        
        found = None
        
        # Strategy: Try x values that are likely to yield solutions
        # For 1/x + 1/y + 1/z = 4/n, we need x > n/4
        # Try x in increasing order from n//4+1
        
        x_start = n // 4 + 1
        x_end = min(4 * n + 200, 20000)
        
        for x in range(x_start, x_end + 1):
            if time.time() > deadline:
                break
            
            A = 4 * x - n
            if A <= 0:
                continue
            
            B = n * x
            B_sq = B * B
            
            factors = get_factors(B)
            divs = generate_divisors_b_sq(factors)
            
            for u in divs:
                if u > B:
                    break
                if (u + B) % A != 0:
                    continue
                v = B_sq // u
                if (v + B) % A != 0:
                    continue
                
                y = (u + B) // A
                z = (v + B) // A
                
                if y <= 0 or z <= 0:
                    continue
                if y > BOUND or z > BOUND:
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
        
        # If not found in the primary range, try a secondary range
        if not found and time.time() < deadline:
            # Try x values that are multiples or divisors of n
            candidates = set()
            for k in range(1, 50):
                if k * n > 100000:
                    break
                candidates.add(k * n)
                if n % k == 0:
                    candidates.add(n // k)
            # Also try x = n + k for small k
            for k in range(1, 100):
                candidates.add(n + k)
            
            for x in sorted(candidates):
                if time.time() > deadline:
                    break
                if x <= x_start - 1 or x >= x_end + 1:
                    continue
                if x < n // 4 + 1:
                    continue
                
                A = 4 * x - n
                if A <= 0:
                    continue
                
                B = n * x
                B_sq = B * B
                
                factors = get_factors(B)
                divs = generate_divisors_b_sq(factors)
                
                for u in divs:
                    if u > B:
                        break
                    if (u + B) % A != 0:
                        continue
                    v = B_sq // u
                    if (v + B) % A != 0:
                        continue
                    
                    y = (u + B) // A
                    z = (v + B) // A
                    
                    if y <= 0 or z <= 0:
                        continue
                    if y > BOUND or z > BOUND:
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
        
        answers[idx] = found
    
    return answers
