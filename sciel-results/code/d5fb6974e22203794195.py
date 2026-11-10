"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    LIMIT = 10**12
    answers = [None] * 254

    for n in range(3, 257):
        if time.time() > deadline:
            break
        found = None
        x_min = n // 4 + 1
        x_max = 3 * n // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            if (2 * B) % A == 0:
                y_max = (2 * B) // A - 1
            else:
                y_max = (2 * B) // A
            
            y_min = max(x + 1, B // A + 1)
            
            if y_min > y_max:
                continue
                
            for y in range(y_min, y_max + 1):
                denom = A * y - B
                if denom <= 0:
                    continue
                if (B * y) % denom == 0:
                    z = (B * y) // denom
                    if z > y and z <= LIMIT:
                        if x != y and y != z and x != z:
                            found = [x, y, z]
                            break
            if found:
                break
        
        answers[n - 3] = found

    return answers
# EVOLVE-BLOCK-END
