import time
import math

def construct():
    deadline = time.time() + 85
    results = [None] * 254
    BOUND = 10**12
    
    # Precompute by iterating x, y and solving for z for each n.
    # z = n*x*y / (4*x*y - n*(x+y))
    # Need: 4*x*y - n*(x+y) > 0, z integer, z > y, z <= BOUND, all distinct.
    
    for x in range(1, 400):
        if time.time() > deadline - 5:
            break
        for y in range(x + 1, x + 5000):
            if time.time() > deadline - 5:
                break
            
            n_max = (4 * x * y - 1) // (x + y)
            if n_max < 3:
                continue
            if n_max > 256:
                n_max = 256
            
            for n in range(3, n_max + 1):
                idx = n - 3
                if results[idx] is not None:
                    continue
                
                z_den = 4 * x * y - n * (x + y)
                if z_den <= 0:
                    continue
                
                z_num = n * x * y
                if z_num % z_den != 0:
                    continue
                
                z = z_num // z_den
                if z <= y or z > BOUND:
                    continue
                if x == z or y == z:
                    continue
                
                results[idx] = [x, y, z]
    
    return results
