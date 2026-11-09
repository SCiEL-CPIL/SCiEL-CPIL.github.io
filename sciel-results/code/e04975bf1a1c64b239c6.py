import math
import time

def construct():
    deadline = time.time() + 85  # 85s safety margin for 90s limit
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        target = 4 / n
        
        # Try x values starting from a reasonable minimum
        # We need 1/x < 4/n, so x > n/4
        # Also x, y, z should be distinct and positive
        min_x = n // 4 + 1
        
        for x in range(min_x, min_x + 5000):  # Try up to 5000 values of x
            if time.time() > deadline:
                break
            if found:
                break
            
            a = 4 * x - n  # numerator of (4/n - 1/x)
            if a <= 0:
                continue
            b = n * x       # denominator
            
            # Reduce a/b
            g = math.gcd(a, b)
            a_red = a // g
            b_red = b // g
            
            # Now solve 1/y + 1/z = a_red / b_red
            # => a_red * y * z = b_red * (y + z)
            # => (a_red * y - b_red)(a_red * z - b_red) = b_red^2
            # Let u = a_red * y - b_red, v = a_red * z - b_red
            # u * v = b_red^2, and y = (u + b_red) / a_red, z = (v + b_red) / a_red
            # Need u, v positive divisors of b_red^2 with u <= v (WLOG)
            # and (u + b_red) % a_red == 0, (v + b_red) % a_red == 0
            
            b_sq = b_red * b_red
            
            # Use arithmetic progression: u = k*a_red - b_red for k >= 1
            # u must be positive and divide b_red^2
            # k_min: u > 0 => k*a_red > b_red => k > b_red/a_red
            # k_max: u <= b_red^2 => k*a_red - b_red <= b_red^2 => k <= (b_red^2 + b_red)/a_red
            
            k_min = b_red // a_red + 1
            k_max = (b_sq + b_red) // a_red + 1
            
            found_for_x = False
            for k in range(k_min, k_max + 1):
                if time.time() > deadline:
                    break
                
                u = k * a_red - b_red
                if u <= 0:
                    continue
                if u > b_sq:
                    break
                
                if b_sq % u != 0:
                    continue
                
                v = b_sq // u
                if u > v:
                    continue
                
                # y = (u + b_red) / a_red
                if (u + b_red) % a_red != 0:
                    continue
                y = (u + b_red) // a_red
                
                # z = (v + b_red) / a_red
                if (v + b_red) % a_red != 0:
                    continue
                z = (v + b_red) // a_red
                
                if y <= 0 or z <= 0:
                    continue
                
                # Ensure distinctness and positivity
                if x == y or x == z or y == z:
                    continue
                if y > 10**12 or z > 10**12:
                    continue
                
                # Verify the solution
                # 1/x + 1/y + 1/z == 4/n
                # Cross multiply: n*(y*z + x*z + x*y) == 4*x*y*z
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    # Ensure all are positive and within bounds
                    if x > 0 and y > 0 and z > 0 and x <= 10**12 and y <= 10**12 and z <= 10**12:
                        found = [x, y, z]
                        found_for_x = True
                        break
            
            if found_for_x:
                break
        
        answers.append(found)
    
    return answers
