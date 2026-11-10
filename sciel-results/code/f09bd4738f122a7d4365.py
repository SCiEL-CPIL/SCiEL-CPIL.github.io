import time

def construct():
    answers = []
    start_time = time.time()
    MAX_DENOM = 10**12
    TIME_LIMIT = 85.0  # Safety margin

    for n in range(3, 257):
        if time.time() - start_time > TIME_LIMIT:
            answers.append(None)
            continue

        found = None
        # x must satisfy n/4 < x < 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() - start_time > TIME_LIMIT:
                break
                
            A = 4 * x - n
            B = n * x
            
            # We need A > 0, which is guaranteed by x > n/4
            # y must satisfy B/A < y < 2B/A
            # And y > x
            
            y_min = max(x + 1, B // A + 1)
            y_max = (2 * B - 1) // A
            
            for y in range(y_min, y_max + 1):
                D = A * y - B
                # D must be positive, which is guaranteed by y > B/A
                if (B * y) % D == 0:
                    z = (B * y) // D
                    if z > y and z <= MAX_DENOM:
                        found = [x, y, z]
                        break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
