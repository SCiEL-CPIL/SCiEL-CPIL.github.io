import time
import math

def _find_solution(n):
    """Find distinct positive integers x,y,z with 1/x+1/y+1/z=4/n, x<y<z, all<=10^12."""
    MAX_DENOM = 10**12
    # x > n/4, x <= 3n/4
    x_min = n // 4 + 1
    x_max = (3 * n) // 4
    
    for x in range(x_min, x_max + 1):
        A = 4 * x - n  # must be positive
        B = n * x
        if A <= 0:
            continue
        
        # 1/y + 1/z = A/B
        # (Ay-B)(Az-B) = B^2
        # y in range (B/A, 2B/A), y > x
        y_min = max(x + 1, B // A + 1)
        y_max = (2 * B) // A
        
        if y_max < y_min:
            continue
        
        for y in range(y_min, y_max + 1):
            denom = A * y - B
            if denom <= 0:
                continue
            num = B * y
            if num % denom != 0:
                continue
            z = num // denom
            if z <= y:
                continue
            if z > MAX_DENOM:
                continue
            if x == y or y == z or x == z:
                continue
            # Verify
            if n * (y * z + x * z + x * y) == 4 * x * y * z:
                return [x, y, z]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    answers = []
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        sol = _find_solution(n)
        answers.append(sol)
    
    return answers
