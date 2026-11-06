import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill
    
    answers = [None] * 254
    
    # Precompute primes for trial division up to a reasonable limit
    limit = 200000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    def factorize(num):
        factors = {}
        temp = num
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                factors[p] = count
        if temp > 1:
            factors[temp] = 1
        return factors
    
    def get_divisors_of_square(factors):
        # factors is dict of p -> e for B
        # We need divisors of B^2, so exponents are 2*e
        divisors = [1]
        for p, e in factors.items():
            new_divs = []
            exp_max = 2 * e
            for d in divisors:
                val = d
                for k in range(exp_max + 1):
                    new_divs.append(val)
                    val *= p
            divisors = new_divs
        return divisors
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
            
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Strategy: For each x, use the divisor-based approach (faster for large n)
        # but also try a quick linear scan for small ranges as a fast path
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
                
            B = n * x
            
            # Quick bounds check for y
            # 1/y + 1/z = A/B, with y <= z
            # y must satisfy: B/A < y <= 2B/A
            y_min_bound = B // A + 1
            y_max_bound = (2 * B) // A
            
            # Also y > x for distinctness (assuming we search x < y < z)
            y_min = max(x + 1, y_min_bound)
            y_max = y_max_bound
            
            if y_max < y_min:
                continue
            
            # If the range is small, do a linear scan (fast)
            range_size = y_max - y_min + 1
            if range_size < 5000:
                for y in range(y_min, y_max + 1):
                    D = A * y - B
                    if D <= 0:
                        continue
                    N = B * y
                    if N % D == 0:
                        z = N // D
                        if z > y:
                            if z <= 1000000000000:
                                found = [x, y, z]
                                break
            else:
                # For larger ranges, use the divisor approach
                # We need u | B^2 such that y = (u+B)/A is integer and in range
                # and z = (B^2/u + B)/A is integer and > y
                b_factors = factorize(B)
                b_sq_divs = get_divisors_of_square(b_factors)
                
                # Sort divisors to process in order (optional, but helps find solutions faster)
                # Actually, we just need to check each one
                for u in b_sq_divs:
                    if (u + B) % A != 0:
                        continue
                    
                    y = (u + B) // A
                    if y <= x or y < y_min:
                        continue
                    if y > y_max:
                        continue
                    
                    v = (B * B) // u
                    if (v + B) % A != 0:
                        continue
                    
                    z = (v + B) // A
                    if z <= y:
                        continue
                    if z > 1000000000000:
                        continue
                    
                    # Verify distinctness
                    if x == y or x == z or y == z:
                        continue
                    
                    # Final verification
                    if n * (y * z + x * z + x * y) == 4 * x * y * z:
                        found = [x, y, z]
                        break
            
            if found:
                break
                
        answers[idx] = found
        
    return answers
