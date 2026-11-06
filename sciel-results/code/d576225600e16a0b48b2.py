import time
import math

DENOM_BOUND = 10**12

def _divisors(n):
    divs = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return divs

def construct():
    deadline = time.time() + 85.0
    results = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
            
        found = None
        n_start = time.time()
        
        # x must satisfy 1/x < 4/n => x > n/4
        x_min = n // 4 + 1
        
        # Try x values starting from x_min
        # With divisor method, each x is fast.
        # We can try a few hundred x values if needed.
        for x in range(x_min, x_min + 1000):
            if time.time() > deadline:
                break
            
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
                
            B2 = B * B
            divs = _divisors(B2)
            
            for U in divs:
                V = B2 // U
                if U > V:
                    continue
                    
                # y = (U+B)/A, z = (V+B)/A
                num_y = U + B
                num_z = V + B
                
                if num_y % A != 0 or num_z % A != 0:
                    continue
                    
                y = num_y // A
                z = num_z // A
                
                if y <= 0 or z <= 0:
                    continue
                    
                if y == z:
                    continue
                    
                if x == y or x == z:
                    continue
                    
                if y > DENOM_BOUND or z > DENOM_BOUND:
                    continue
                
                found = [x, y, z]
                break
                
            if found:
                break
                
        results[idx] = found
        
    return results
