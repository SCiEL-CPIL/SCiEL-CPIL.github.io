import time

def construct():
    answers = []
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    for n in range(3, 257):
        if time.time() > deadline:
            # Fill remaining with None to maintain shape
            answers.extend([None] * (256 - n + 1))
            break
            
        found = None
        # x must satisfy 1/x < 4/n => x > n/4
        start_x = n // 4 + 1
        # x must satisfy 3/x > 4/n => x < 3n/4
        # We iterate x upwards. Usually small x works.
        # We limit the number of x values tried to ensure speed, 
        # but for n<=256, even full range is small.
        # Max x is approx 3n/4.
        max_x = (3 * n) // 4 - 1
        
        for x in range(start_x, max_x + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
                
            B = n * x
            
            # 1/y + 1/z = A/B
            # y > x
            # y > B/A
            # y < 2*B/A
            y_min = max(x + 1, B // A + 1)
            y_max = (2 * B) // A
            
            # Optimization: if range is too large, we might spend too much time.
            # But for n<=256, B/A is roughly n, so range is O(n).
            # n=256, range ~ 256. Very fast.
            
            for y in range(y_min, y_max + 1):
                denom = A * y - B
                if denom <= 0:
                    continue
                if (B * y) % denom == 0:
                    z = (B * y) // denom
                    if z > y and z != x and y != x:
                        # Check distinctness
                        if len(set([x, y, z])) == 3:
                            found = [x, y, z]
                            break
            if found:
                break
        
        answers.append(found)
        
    return answers
