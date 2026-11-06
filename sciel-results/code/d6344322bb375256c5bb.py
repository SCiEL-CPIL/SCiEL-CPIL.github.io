import time
import math

def construct():
    deadline = time.time() + 85
    results = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            B2 = B * B
            
            y_min = max(x + 1, B // A + 1)
            y_max = (2 * B) // A - 1
            
            if y_max < y_min:
                continue
            
            for y in range(y_min, y_max + 1):
                if time.time() > deadline:
                    break
                t = A * y - B
                if t <= 0:
                    continue
                if B2 % t != 0:
                    continue
                z_num = B2 // t + B
                if z_num % A != 0:
                    continue
                z = z_num // A
                if z <= y:
                    continue
                if x == y or x == z or y == z:
                    continue
                if z > 10**12:
                    continue
                found = [x, y, z]
                break
            
            if found:
                break
        
        results[idx] = found
    
    return results
