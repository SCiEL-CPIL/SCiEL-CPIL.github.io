import time

def construct():
    deadline = time.time() + 75.0
    answers = [None] * 254
    BOUND = 10**12
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        found = None
        
        x_min = n // 4 + 1
        x_max = (3 * n - 1) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            a = 4 * n - x
            if a <= 0:
                continue
            b = n * x
            
            y_min = max(x + 1, b // a + 1)
            y_max = (2 * b - 1) // a  # largest y with y < 2b/a
            
            if y_min > y_max:
                continue
                
            for y in range(y_min, y_max + 1):
                denom = a * y - b
                if denom <= 0:
                    continue
                num = b * y
                if num % denom != 0:
                    continue
                z = num // denom
                if z <= y or z > BOUND:
                    continue
                if x != y and y != z and x != z:
                    found = [x, y, z]
                    break
            if found:
                break
        answers[idx] = found
    
    return answers
