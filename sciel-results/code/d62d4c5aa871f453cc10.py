import time
import math

def construct():
    deadline = time.time() + 80.0
    answers = [None] * 254
    
    def factorize(num):
        """Return prime factorization as dict {prime: exponent}"""
        factors = {}
        d = 2
        while d * d <= num:
            while num % d == 0:
                factors[d] = factors.get(d, 0) + 1
                num //= d
            d += 1
        if num > 1:
            factors[num] = factors.get(num, 0) + 1
        return factors
    
    # Precompute factorizations for all possible b = n*x values
    # n is 3..256, x is n//4+1 .. 3n//4
    # b = n*x, max b = 256 * 192 = 49152
    # Precompute factorizations for 1..50000
    max_b = 256 * 192 + 10  # 49162
    factor_cache = {}
    for i in range(1, max_b + 1):
        factor_cache[i] = factorize(i)
    
    def divisors_of_square(b):
        """Generate all positive divisors of b^2 using cached prime factorization of b"""
        if b <= 0:
            return [1]
        factors = factor_cache[b]
        divs = [1]
        for p, e in factors.items():
            exp = 2 * e
            new_divs = []
            for d in divs:
                val = d
                for _ in range(exp + 1):
                    new_divs.append(val)
                    val *= p
            divs = new_divs
        return divs
    
    for n in range(256, 2, -1):
        idx = n - 3
        if time.time() > deadline:
            break
        
        n_start = time.time()
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            if time.time() - n_start > 2.0:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            
            b = n * x
            
            # Get divisors of b^2
            divs = divisors_of_square(b)
            
            # Sort divisors to try smaller ones first (smaller k gives smaller y, 
            # but we need y > x, so actually we want k large enough that y > x)
            # y = (k + b) / a > x means k + b > a*x = x*(4x-n) = 4x² - nx
            # k > 4x² - nx - b = 4x² - nx - nx = 4x² - 2nx = 2x(2x - n)
            # For x > n/2, 2x - n > 0, so k > 2x(2x-n)
            # For x <= n/2, 2x - n <= 0, so k > negative number, always true for positive k
            
            for k in divs:
                if (k + b) % a != 0:
                    continue
                
                y = (k + b) // a
                if y <= x:
                    continue
                
                # k must divide b^2, which it does by construction
                bq = b * b // k
                if (bq + b) % a != 0:
                    continue
                
                z = (bq + b) // a
                
                if z <= y:
                    continue
                
                if z > 10**12 or y > 10**12:
                    continue
                
                # Verify with cross multiplication
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
