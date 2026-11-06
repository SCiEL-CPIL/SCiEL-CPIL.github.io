import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    answers = [None] * 254
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
            
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Try x from largest to smallest: larger x gives smaller y ranges
        for x in range(x_max, x_min - 1, -1):
            if time.time() > deadline:
                break
                
            num_yz = 4 * x - n
            if num_yz <= 0:
                continue
                
            den_yz = n * x
            
            y_min = max(x + 1, den_yz // num_yz + 1)
            y_max = (2 * den_yz) // num_yz
            
            if y_max < y_min:
                continue
                
            for y in range(y_min, y_max + 1):
                D = num_yz * y - den_yz
                if D <= 0:
                    continue
                
                N = den_yz * y
                if N % D == 0:
                    z = N // D
                    if z > y:
                        found = [x, y, z]
                        break
            
            if found:
                break
                
        answers[idx] = found
        
    return answers
