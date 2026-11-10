import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    LIMIT = 10**12
    answers = []

    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        
        # x must satisfy: 1/x < 4/n => x > n/4
        # And 1/x + 1/y + 1/z = 4/n with x < y < z implies 3/x > 4/n => x < 3n/4
        # So x_min = floor(n/4) + 1
        # x_max = floor(3n/4)
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            
            b = n * x
            b2 = b * b
            
            # We need to find divisors u of b2 such that:
            # 1. (u + b) % a == 0
            # 2. (b2//u + b) % a == 0
            # 3. y = (u + b) // a, z = (b2//u + b) // a are distinct from x and each other
            # 4. y, z <= LIMIT
            
            # Iterate u from 1 to sqrt(b2) = b
            # To speed up, we can iterate u and check if b2 % u == 0
            # This is O(b) per x. b is roughly n*x.
            # Max b ~ 256 * 192 ~ 50000.
            # This loop is acceptable if we break early.
            
            # Optimization: We can just iterate u up to b.
            # If we find a valid u, we compute y, z.
            # We should check both u and b2//u if they are different?
            # Actually, if u is a divisor, v = b2/u is also a divisor.
            # The pair (y,z) derived from u is the same set as the pair derived from v (swapped).
            # So checking u from 1 to b is sufficient to find all unique sets {y,z}.
            
            for u in range(1, b + 1):
                if b2 % u != 0:
                    continue
                
                v = b2 // u
                
                # Check divisibility for y
                if (u + b) % a != 0:
                    continue
                y = (u + b) // a
                
                # Check divisibility for z
                if (v + b) % a != 0:
                    continue
                z = (v + b) // a
                
                # Check positivity (implied by u,v > 0 and a > 0)
                if y <= 0 or z <= 0:
                    continue
                
                # Check bounds
                if y > LIMIT or z > LIMIT:
                    continue
                
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                
                # Found a valid solution
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
