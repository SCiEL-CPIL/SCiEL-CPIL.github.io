import time
import math

def construct():
    answers = []
    deadline = time.time() + 55  # safety margin
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        
        if n % 2 == 0:
            # Formula for even n:
            # x = n/2, y = n/2 + 1, z = n(n+2)/4
            x = n // 2
            y = x + 1
            z = (n * (n + 2)) // 4
            
            # Verify distinctness and bounds
            if x != y and y != z and x != z and x > 0 and y > 0 and z > 0:
                if x <= 10**12 and y <= 10**12 and z <= 10**12:
                    # Verify equation
                    # 1/x + 1/y + 1/z == 4/n
                    # n*(y*z + x*z + x*y) == 4*x*y*z
                    lhs = n * (y * z + x * z + x * y)
                    rhs = 4 * x * y * z
                    if lhs == rhs:
                        found = [x, y, z]
        
        if found is None:
            # Search for odd n (and fallback for even if formula failed, though it shouldn't)
            # x must be in (n/4, 3n/4]
            start_x = n // 4 + 1
            end_x = (3 * n) // 4
            
            for x in range(start_x, end_x + 1):
                if time.time() > deadline:
                    break
                
                # 1/y + 1/z = 4/n - 1/x
                # Let A = 4x - n. Must be > 0.
                A = 4 * x - n
                if A <= 0:
                    continue
                
                # 1/y + 1/z = A / (nx)
                # y > x
                # y <= 2nx / A
                y_max = (2 * n * x) // A
                if y_max < x + 1:
                    continue
                
                # To keep search fast, we can limit y_max if it's huge, 
                # but for n<=256, y_max is small.
                # Iterate y
                for y in range(x + 1, y_max + 1):
                    # 1/z = A/(nx) - 1/y = (Ay - nx) / (nxy)
                    # z = nxy / (Ay - nx)
                    num = n * x * y
                    den = A * y - n * x
                    if den <= 0:
                        continue
                    if num % den != 0:
                        continue
                    z = num // den
                    if z <= 0:
                        continue
                    if z < y:
                        continue # We assumed y <= z by bound, but check just in case
                    # Check distinctness
                    if x == y or y == z or x == z:
                        continue
                    # Check bounds
                    if x > 10**12 or y > 10**12 or z > 10**12:
                        continue
                    # Verify
                    lhs = n * (y * z + x * z + x * y)
                    rhs = 4 * x * y * z
                    if lhs == rhs:
                        found = [x, y, z]
                        break
                if found:
                    break
        
        answers.append(found)
        
    return answers
