import time

def construct():
    deadline = time.time() + 80.0
    answers = [None] * 254
    BOUND = 10**12
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4 + 1
        
        for x in range(x_min, min(x_max, 5000) + 1):
            if time.time() > deadline:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            
            b = n * x
            # y must satisfy: b/a < y < 2b/a for z > y > x
            y_lo = b // a + 1
            y_hi = 2 * b // a - 1
            
            if y_lo <= x:
                y_lo = x + 1
            if y_hi < y_lo:
                continue
            
            # Cap y range to avoid excessive computation
            if y_hi - y_lo > 100000:
                y_hi = y_lo + 100000
            
            for y in range(y_lo, y_hi + 1):
                if time.time() > deadline:
                    break
                
                denom = a * y - b
                if denom <= 0:
                    continue
                
                # Check if denom divides b*y (so z is integer)
                if (b * y) % denom == 0:
                    z = (b * y) // denom
                    if z > y and z <= BOUND:
                        # Verify distinctness and the original equation
                        if x != y and x != z and y != z:
                            # Verify: 1/x + 1/y + 1/z == 4/n
                            # Cross multiply: n(yz + xz + xy) == 4xyz
                            lhs = n * (y * z + x * z + x * y)
                            rhs = 4 * x * y * z
                            if lhs == rhs:
                                found = [x, y, z]
                                break
            
            if found:
                break
        
        # If not found with small x, try larger x values
        if not found and time.time() < deadline:
            for x in range(max(x_max, 5000), 20000):
                if time.time() > deadline:
                    break
                
                a = 4 * x - n
                if a <= 0:
                    continue
                
                b = n * x
                y_lo = b // a + 1
                y_hi = 2 * b // a - 1
                
                if y_lo <= x:
                    y_lo = x + 1
                if y_hi < y_lo:
                    continue
                
                if y_hi - y_lo > 50000:
                    y_hi = y_lo + 50000
                
                for y in range(y_lo, y_hi + 1):
                    if time.time() > deadline:
                        break
                    
                    denom = a * y - b
                    if denom <= 0:
                        continue
                    
                    if (b * y) % denom == 0:
                        z = (b * y) // denom
                        if z > y and z <= BOUND:
                            if x != y and x != z and y != z:
                                lhs = n * (y * z + x * z + x * y)
                                rhs = 4 * x * y * z
                                if lhs == rhs:
                                    found = [x, y, z]
                                    break
                
                if found:
                    break
        
        answers[idx] = found
    
    return answers
