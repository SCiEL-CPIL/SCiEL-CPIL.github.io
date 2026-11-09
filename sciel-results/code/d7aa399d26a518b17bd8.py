import time
import math

def construct():
    answers = []
    start_time = time.time()
    deadline = start_time + 55  # 55s margin out of 90s
    
    # Precompute or just loop
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x range
        # 1/x < 4/n => x > n/4
        # 3/x >= 4/n => x <= 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            B = n * x
            
            # A must be positive for y, z to be positive in this form
            # Since x > n/4, 4x > n, so A > 0.
            
            # Reduce A/B
            g = math.gcd(A, B)
            A_red = A // g
            B_red = B // g
            
            # 1/y + 1/z = A_red / B_red
            # y > B_red / A_red
            # y <= 2 * B_red / A_red
            y_min = B_red // A_red + 1
            if y_min <= x:
                y_min = x + 1
                
            y_max = (2 * B_red) // A_red
            
            if y_min > y_max:
                continue
                
            # Iterate y
            # We can also check if y_max is huge, but for n<=256 it's small.
            for y in range(y_min, y_max + 1):
                # 1/z = A_red/B_red - 1/y = (A_red*y - B_red) / (B_red*y)
                num = A_red * y - B_red
                if num <= 0:
                    continue
                den = B_red * y
                if den % num == 0:
                    z = den // num
                    if z > y:
                        # Check distinctness x < y < z is satisfied by construction
                        # Check bound
                        if z <= 1000000000000:
                            found = [x, y, z]
                            break
            if found:
                break
        
        answers.append(found)
        
    return answers
