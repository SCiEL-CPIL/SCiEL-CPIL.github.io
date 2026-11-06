import time

DENOM_BOUND = 10**12

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254
    
    for i, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
            
        found = None
        
        # x must be > n/4 and <= 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            # Denominator for y bound: 4x - n
            denom_y = 4 * x - n
            if denom_y <= 0:
                continue
                
            # y > nx / (4x - n)
            # y_min = floor(nx / (4x - n)) + 1
            # Also y >= x
            y_lower = (n * x) // denom_y + 1
            if y_lower < x:
                y_lower = x
                
            # y <= 2nx / (4x - n)
            y_upper = (2 * n * x) // denom_y
            
            if y_lower > y_upper:
                continue
                
            for y in range(y_lower, y_upper + 1):
                if time.time() > deadline:
                    break
                    
                D = 4 * x * y - n * x - n * y
                # D is positive by construction of y_lower
                
                numerator = n * x * y
                if numerator % D != 0:
                    continue
                    
                z = numerator // D
                
                # We assumed x <= y <= z. Check z >= y.
                if z < y:
                    continue
                    
                # Check distinctness
                if x == y or y == z or x == z:
                    continue
                    
                # Check bound
                if z > DENOM_BOUND:
                    continue
                
                # Valid solution
                found = [x, y, z]
                break
            
            if found:
                break
                
        answers[i] = found
        
    return answers
