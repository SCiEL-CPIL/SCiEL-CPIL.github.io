import time

def construct():
    start = time.time()
    deadline = start + 85.0
    LIMIT = 10**12
    answers = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        if x_max < x_min:
            continue
        
        # Try x values, prioritizing smaller a = 4x-n (smaller x)
        # Since a = 4x - n, smaller x gives smaller a, which gives smaller y range
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # (ay - b)(az - b) = b^2
            # Let u = ay - b, v = az - b, then u*v = b^2
            # u ≡ -b (mod a), v ≡ -b (mod a)
            # y = (u+b)/a, z = (v+b)/a
            # Need y, z positive integers, x < y < z, all <= LIMIT
            
            # For small a, we can iterate u over divisors of b^2
            # b <= 256 * 192 = 49152, so b^2 <= ~2.4e9
            # Number of divisors of b^2 is manageable
            
            # Find all divisors of b^2
            b2 = b * b
            
            # For performance, limit the approach
            if a <= 100:
                # Use divisor enumeration
                # Find all divisors u of b2 such that u ≡ -b (mod a)
                # and u < b2/u (to ensure y < z)
                # u = ay - b, so u > 0 means y > b/a
                # u < b2 means v > b, so z > b/a + b/a = 2b/a... 
                
                # Just iterate u from 1 to sqrt(b2)
                import math
                sqrt_b2 = int(math.isqrt(b2))
                
                for u in range(1, sqrt_b2 + 1):
                    if b2 % u != 0:
                        continue
                    v = b2 // u
                    # Check u ≡ -b (mod a) and v ≡ -b (mod a)
                    if (u + b) % a != 0:
                        continue
                    if (v + b) % a != 0:
                        continue
                    
                    y = (u + b) // a
                    z = (v + b) // a
                    
                    # Need y, z positive, x < y < z, all <= LIMIT
                    if y <= 0 or z <= 0:
                        continue
                    if y <= x or z <= x:
                        continue
                    if y >= z:
                        continue
                    if y > LIMIT or z > LIMIT:
                        continue
                    if x == y or x == z or y == z:
                        continue
                    
                    # Verify
                    if n * (y * z + x * z + x * y) == 4 * x * y * z:
                        found = [x, y, z]
                        break
                
                if found:
                    break
            else:
                # Direct y search with cap
                y_min = b // a + 1
                y_max_val = 2 * b / a
                y_max = int(y_max_val) if y_max_val != int(y_max_val) else int(y_max_val) - 1
                
                if y_max < y_min:
                    continue
                
                max_search = 200000
                search_y_max = min(y_max, y_min + max_search - 1)
                
                for y in range(y_min, search_y_max + 1):
                    denom = a * y - b
                    if denom <= 0:
                        continue
                    num = b * y
                    if num % denom != 0:
                        continue
                    z = num // denom
                    if z <= y or z > LIMIT:
                        continue
                    if x == y or x == z or y == z:
                        continue
                    if n * (y * z + x * z + x * y) == 4 * x * y * z:
                        found = [x, y, z]
                        break
                
                if found:
                    break
        
        answers[idx] = found
    
    return answers
