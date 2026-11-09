import time

def construct():
    answers = []
    deadline = time.time() + 55  # Safety margin before 60s limit
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        start_x = n // 4 + 1
        end_x = min(3 * n // 4, 10**12)
        
        for x in range(start_x, end_x + 1):
            if time.time() > deadline:
                break
            
            a = 4 * x - n
            b = n * x
            
            if a <= 0:
                continue
            
            y_min = max(x + 1, b // a + 1)
            y_max = min(2 * b // a, 10**12)
            
            if y_min > y_max:
                continue
            
            # Limit inner loop to avoid excessive time on one x
            y_search_limit = min(y_max, y_min + 50000)
            
            for y in range(y_min, y_search_limit + 1):
                if time.time() > deadline:
                    break
                
                denom = a * y - b
                if denom <= 0:
                    continue
                
                if (b * y) % denom != 0:
                    continue
                
                z = (b * y) // denom
                
                if z <= y or z > 10**12:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
