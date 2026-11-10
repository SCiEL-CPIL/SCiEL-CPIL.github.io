import time

def construct():
    answers = []
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        x_min = n // 4 + 1
        x_max = min(x_min + 100000, 3 * n // 4 + 1)
        
        # Try x from larger values first (smaller y ranges)
        for x in range(x_max, x_min - 1, -1):
            if time.time() > deadline:
                break
                
            a = 4 * x - n
            b = n * x
            
            if a <= 0:
                continue
                
            y_low = b // a + 1
            y_high = min(y_low + 5000, 2 * b // a + 1)
            
            for y in range(y_low, y_high + 1):
                if y == x:
                    continue
                
                denominator = a * y - b
                if denominator <= 0:
                    continue
                
                num = b * y
                if num % denominator == 0:
                    z = num // denominator
                    if z == x or z == y:
                        continue
                    if z > 1000000000000:
                        continue
                    if z > 0:
                        found = [x, y, z]
                        break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
