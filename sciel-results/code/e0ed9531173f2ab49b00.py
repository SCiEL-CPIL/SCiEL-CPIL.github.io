import time
import math

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254
    BOUND = 10**12
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        found = None
        
        x_min = n // 4 + 1
        
        # Try x values starting from x_min
        max_x_tries = 2000
        for x in range(x_min, x_min + max_x_tries):
            if time.time() > deadline:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # y must satisfy:
            # 1) y > b/a
            # 2) y < 2*b/a (so z > y)
            # 3) y > x
            # 4) z = b*y/(a*y - b) <= BOUND
            
            y_low_1 = b // a + 1
            y_low_3 = x + 1
            
            denom = BOUND * a - b
            if denom <= 0:
                continue
            y_low_4 = (BOUND * b + denom - 1) // denom
            
            y_low = max(y_low_1, y_low_3, y_low_4)
            y_high = (2 * b + a - 1) // a - 1
            
            if y_high < y_low:
                continue
            
            # Try several y candidates: y_low, y_low+1, ..., and also try
            # values derived from small divisors of b
            candidates = set()
            candidates.add(y_low)
            for dy in range(1, 6):
                y_candidate = y_low + dy
                if y_candidate <= y_high:
                    candidates.add(y_candidate)
            
            # Also try y values where (a*y - b) is a small divisor of b
            # (a*y - b) | b  =>  y = (d + b) / a for d | b
            # Try small divisors d of b
            b_small = b
            for d in range(1, min(100, b_small) + 1):
                if b_small % d == 0:
                    y_candidate = (d + b_small) // a
                    if a * y_candidate == d + b_small:  # exact division
                        if y_low <= y_candidate <= y_high:
                            candidates.add(y_candidate)
                    # Also try the other divisor pair
                    d2 = b_small // d
                    y_candidate2 = (d2 + b_small) // a
                    if a * y_candidate2 == d2 + b_small:
                        if y_low <= y_candidate2 <= y_high:
                            candidates.add(y_candidate2)
            
            for y in sorted(candidates):
                if y <= y_high:
                    z_num = b * y
                    z_den = a * y - b
                    if z_den <= 0:
                        continue
                    if z_num % z_den == 0:
                        z = z_num // z_den
                        if z > y and y > x and z <= BOUND and y <= BOUND and x <= BOUND:
                            lhs = n * y * z + n * x * z + n * x * y
                            rhs = 4 * x * y * z
                            if lhs == rhs:
                                found = [x, y, z]
                                break
            if found:
                break
        
        answers[idx] = found
    
    # Second pass: try to solve remaining unsolved n's with extended search
    for n in range(3, 257):
        idx = n - 3
        if answers[idx] is not None:
            continue
        if time.time() > deadline - 5.0:
            break
        
        x_min = n // 4 + 1
        found = None
        
        # Extended x search with divisor-based y finding
        max_x_tries = 5000
        for x in range(x_min, x_min + max_x_tries):
            if time.time() > deadline - 2.0:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            y_low_1 = b // a + 1
            y_low_3 = x + 1
            
            denom = BOUND * a - b
            if denom <= 0:
                continue
            y_low_4 = (BOUND * b + denom - 1) // denom
            
            y_low = max(y_low_1, y_low_3, y_low_4)
            y_high = (2 * b + a - 1) // a - 1
            
            if y_high < y_low:
                continue
            
            # Enumerate divisors of b up to a limit, check if (d+b)/a is valid y
            # where d | b and d ≡ -b (mod a)
            # Simplification: try d from small divisors
            limit = min(200, b)
            for d in range(1, limit + 1):
                if b % d == 0:
                    # d is a divisor, check if (d + b) % a == 0
                    if (d + b) % a == 0:
                        y = (d + b) // a
                        if y_low <= y <= y_high and y > x:
                            z_num = b * y
                            z_den = a * y - b
                            if z_den > 0 and z_num % z_den == 0:
                                z = z_num // z_den
                                if z > y and z <= BOUND:
                                    lhs = n * y * z + n * x * z + n * x * y
                                    rhs = 4 * x * y * z
                                    if lhs == rhs:
                                        found = [x, y, z]
                                        break
                if time.time() > deadline - 2.0:
                    break
            if found:
                break
        
        answers[idx] = found
    
    return answers
