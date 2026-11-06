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
        # For each x, find valid y range
        max_x_tries = 2000
        for x in range(x_min, x_min + max_x_tries):
            if time.time() > deadline:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # y must satisfy:
            # 1) y > b/a  (so that a*y - b > 0, i.e., z > 0)
            # 2) y < 2*b/a  (so that z > y)
            # 3) y > x  (for distinctness, assuming x < y < z)
            # 4) z = b*y/(a*y - b) <= BOUND
            #    => b*y <= BOUND*(a*y - b)
            #    => b*y <= BOUND*a*y - BOUND*b
            #    => BOUND*b <= y*(BOUND*a - b)
            #    => y >= BOUND*b / (BOUND*a - b)
            
            # y_low from condition 1: y > b/a, so y >= floor(b/a) + 1
            y_low_1 = b // a + 1
            
            # y_low from condition 3: y > x, so y >= x + 1
            y_low_3 = x + 1
            
            # y_low from condition 4: y >= ceil(BOUND*b / (BOUND*a - b))
            denom = BOUND * a - b
            if denom <= 0:
                # This shouldn't happen for reasonable x, but skip if it does
                continue
            y_low_4 = (BOUND * b + denom - 1) // denom  # ceil division
            
            y_low = max(y_low_1, y_low_3, y_low_4)
            
            # y_high from condition 2: y < 2*b/a, so y <= ceil(2*b/a) - 1
            y_high = (2 * b + a - 1) // a - 1  # ceil(2b/a) - 1
            
            if y_high < y_low:
                continue
            
            # Pick y = y_low (smallest valid y, gives largest z but still <= BOUND)
            y = y_low
            z = b * y // (a * y - b)
            
            # Verify all constraints
            if z > y and y > x and z <= BOUND and y <= BOUND and x <= BOUND:
                # Cross-multiply to verify: n*y*z + n*x*z + n*x*y == 4*x*y*z
                lhs = n * y * z + n * x * z + n * x * y
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
        
        answers[idx] = found
    
    return answers
