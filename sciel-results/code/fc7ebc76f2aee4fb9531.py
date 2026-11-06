import time

def construct():
    deadline = time.time() + 85.0
    answers = []
    
    # We iterate n from 3 to 256
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        
        # x must satisfy: n/4 < x < 3n/4
        # x_min = floor(n/4) + 1
        # x_max = floor(3n/4)
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            if found:
                break
                
            A = 4 * x - n
            # A must be positive. Since x > n/4, 4x > n, so A > 0.
            
            B = n * x
            
            # y range: B/A < y <= 2B/A
            # y must be integer, so y >= floor(B/A) + 1
            # y <= floor(2B/A)
            y_lo = B // A + 1
            y_hi = (2 * B) // A
            
            # Enforce x < y
            if y_lo <= x:
                y_lo = x + 1
                
            if y_lo > y_hi:
                continue
                
            # Iterate y
            # To prevent extremely long loops if no solution exists in this x,
            # we can cap the iterations, but for n <= 256 the ranges are small.
            # Max range size is approx B/A = nx/(4x-n).
            # For x ~ n/4, A ~ 1, B ~ n^2/4. Range ~ n^2/4.
            # For n=256, range ~ 16000.
            # Total x iterations ~ n/2 = 128.
            # Total ops ~ 128 * 16000 = 2M per n.
            # Total for all n: sum of n^2/4 * n/2 ~ n^3/8.
            # Sum n^3 for 3..256 is ~ 256^4/4 = 1e9.
            # This might be tight in Python if not optimized.
            # However, we break as soon as we find a solution.
            # Most n have solutions with small x.
            
            for y in range(y_lo, y_hi + 1):
                if time.time() > deadline:
                    break
                    
                denom = A * y - B
                # denom > 0 is guaranteed by y > B/A
                
                num = B * y
                
                if num % denom == 0:
                    z = num // denom
                    
                    # Check z > y
                    if z > y:
                        # Check z <= 10^12
                        if z <= 1000000000000:
                            # Distinctness: x < y < z ensures x!=y, y!=z, x!=z
                            found = [x, y, z]
                            break
                            
        answers.append(found)
        
    return answers
