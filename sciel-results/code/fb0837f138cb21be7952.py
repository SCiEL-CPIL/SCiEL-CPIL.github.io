import time
import math

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254
    BOUND = 10**12
    
    def factorize(num):
        """Return prime factorization as dict {prime: exponent}."""
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
    
    def get_divisors_from_factors(factors):
        """Generate all divisors from prime factorization dict."""
        divisors = [1]
        for prime, exp in factors.items():
            new_divs = []
            for d in divisors:
                for e in range(exp + 1):
                    new_divs.append(d * (prime ** e))
            divisors = new_divs
        return divisors
    
    def find_solution(n, x_start, x_end, max_divisors=200, check_time=True):
        """Find a distinct unit fraction triple for 1/x+1/y+1/z=4/n using divisor enumeration."""
        for x in range(x_start, x_end + 1):
            if check_time and time.time() > deadline - 3.0:
                return None
            
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # (a*y - b)(a*z - b) = b^2
            # So a*y - b must be a positive divisor of b^2
            # y = (d + b) / a where d | b^2
            # z = (b^2/d + b) / a
            
            b_sq = b * b
            
            # Get divisors of b^2, but limit the count for performance
            factors = factorize(b_sq)
            all_divs = get_divisors_from_factors(factors)
            
            # Sort divisors to find small y first
            all_divs.sort()
            
            # Try divisors in order, limiting to max_divisors for performance
            for d in all_divs[:max_divisors]:
                if d <= 0:
                    continue
                # y = (d + b) / a
                if (d + b) % a != 0:
                    continue
                y = (d + b) // a
                if y <= 0 or y <= x:
                    continue
                if y > BOUND:
                    continue
                
                # z = (b_sq // d + b) / a
                if b_sq % d != 0:
                    continue
                z_num = b_sq // d + b
                if z_num % a != 0:
                    continue
                z = z_num // a
                if z <= 0 or z <= y:
                    continue
                if z > BOUND:
                    continue
                
                # Verify with cross multiplication
                if n * y * z + n * x * z + n * x * y == 4 * x * y * z:
                    if x != y and y != z and x != z:
                        return [x, y, z]
            
            # Also try from the other side: iterate larger divisors
            for d in all_divs[max_divisors:]:
                if d <= 0:
                    continue
                if (d + b) % a != 0:
                    continue
                y = (d + b) // a
                if y <= 0 or y <= x:
                    continue
                if y > BOUND:
                    break  # Divisors are sorted, so we can stop
                
                if b_sq % d != 0:
                    continue
                z_num = b_sq // d + b
                if z_num % a != 0:
                    continue
                z = z_num // a
                if z <= 0 or z <= y:
                    continue
                if z > BOUND:
                    continue
                
                if n * y * z + n * x * z + n * x * y == 4 * x * y * z:
                    if x != y and y != z and x != z:
                        return [x, y, z]
        
        return None
    
    # First pass: quick search for each n
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline - 5.0:
            break
        
        x_min = n // 4 + 1
        found = find_solution(n, x_min, x_min + 200, max_divisors=50)
        answers[idx] = found
    
    # Second pass: extended search for unsolved n's
    for n in range(3, 257):
        idx = n - 3
        if answers[idx] is not None:
            continue
        if time.time() > deadline - 10.0:
            break
        
        x_min = n // 4 + 1
        found = find_solution(n, x_min, x_min + 1000, max_divisors=200)
        answers[idx] = found
    
    # Third pass: even more extended search for remaining unsolved n's
    for n in range(3, 257):
        idx = n - 3
        if answers[idx] is not None:
            continue
        if time.time() > deadline - 2.0:
            break
        
        x_min = n // 4 + 1
        found = find_solution(n, x_min, x_min + 5000, max_divisors=1000)
        answers[idx] = found
    
    return answers
