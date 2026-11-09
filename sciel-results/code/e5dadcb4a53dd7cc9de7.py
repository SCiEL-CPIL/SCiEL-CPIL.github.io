import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        min_x = n // 4 + 1
        max_x = (3 * n) // 4
        
        for x in range(min_x, max_x + 1):
            A = 4 * x - n
            B = n * x
            
            # U = mA - B
            # U > 0 => m > B/A
            # U <= B => mA - B <= B => mA <= 2B => m <= 2B/A
            m_min = B // A + 1
            m_max = (2 * B) // A
            
            for m in range(m_min, m_max + 1):
                U = m * A - B
                # U must divide B^2
                if (B * B) % U != 0:
                    continue
                V = (B * B) // U
                if V < U:
                    continue
                # Check y, z integrality
                # y = (U+B)/A
                if (U + B) % A != 0:
                    continue
                y = (U + B) // A
                # z = (V+B)/A
                if (V + B) % A != 0:
                    continue
                z = (V + B) // A
                
                if x < y < z:
                    found = [x, y, z]
                    break
            
            if found:
                break
                
        answers.append(found)
        
    return answers
