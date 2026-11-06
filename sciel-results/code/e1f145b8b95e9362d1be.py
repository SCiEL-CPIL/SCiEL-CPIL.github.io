import time
import math

DENOM_BOUND = 10**12

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254

    for i, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break

        found = None
        # x must be > n/4 and <= 3n/4
        # 1/x < 4/n => x > n/4
        # 1/x >= 4/(3n) => x <= 3n/4 (since 1/x is the largest term, sum <= 3/x)
        # Actually 1/x + 1/y + 1/z <= 3/x => 4/n <= 3/x => x <= 3n/4
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            # 1/y + 1/z = 4/n - 1/x = (4x - n) / nx
            # Let K = (4x - n) / nx
            # We need y, z such that 1/y + 1/z = K
            # y > 1/K = nx / (4x - n)
            # y <= 2/K = 2nx / (4x - n)
            
            num_k = 4 * x - n
            if num_k <= 0:
                continue
                
            den_k = n * x
            
            y_min = den_k // num_k + 1
            y_max = (2 * den_k) // num_k
            
            # We want x < y < z
            # So y must be > x
            if y_min <= x:
                y_min = x + 1
                
            if y_min > y_max:
                continue
                
            for y in range(y_min, y_max + 1):
                # Check if z is integer
                # 1/z = K - 1/y = (K*y - 1)/y
                # z = y / (K*y - 1) = y * den_k / (num_k * y - den_k)
                # Let D = num_k * y - den_k
                # z = y * den_k / D
                # D must be > 0 and divide y * den_k
                
                D = num_k * y - den_k
                if D <= 0:
                    continue
                    
                # Check divisibility
                N = y * den_k
                if N % D != 0:
                    continue
                    
                z = N // D
                
                # Check z > y
                if z <= y:
                    continue
                    
                # Check bound
                if z > DENOM_BOUND:
                    continue
                    
                # Found a solution x < y < z
                found = [x, y, z]
                break
                
            if found:
                break
                
        answers[i] = found
        
    return answers
