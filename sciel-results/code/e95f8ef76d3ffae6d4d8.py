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
            B2 = B * B
            
            # y_min, y_max logic
            y_min = max(x + 1, B // A + 1)
            y_max = (2 * B) // A - 1
            
            if y_min > y_max:
                continue
            
            Y_min = A * y_min - B
            Y_max = A * y_max - B
            
            # Y must be > 0
            if Y_min <= 0:
                # Adjust Y_min to be the first positive value congruent to -B mod A
                # Y = kA - B
                # kA > B => k > B/A
                # k_min = B//A + 1
                # Y_min = k_min * A - B
                k_min = B // A + 1
                Y_min = k_min * A - B
                # Also ensure y > x
                # y = (Y+B)/A > x => Y+B > Ax => Y > Ax - B
                # Ax - B = x(4x-n) - nx = 4x^2 - 2nx
                # So Y_min = max(Y_min, 4x^2 - 2nx + 1)
                Y_min = max(Y_min, 4*x*x - 2*n*x + 1)
                
                # Re-calculate Y_min to satisfy congruence
                # Y_min % A == -B % A
                # Let r = (-B) % A
                # We need Y_min >= current_Y_min and Y_min % A == r
                r = (-B) % A
                if Y_min % A != r:
                    Y_min += (r - Y_min) % A
            
            if Y_min > Y_max:
                continue
            
            # Iterate Y
            # Step is A
            # Y = Y_min, Y_min + A, ... <= Y_max
            for Y in range(Y_min, Y_max + 1, A):
                if B2 % Y == 0:
                    Z = B2 // Y
                    # z = (Z + B) / A
                    # Check if Z + B is divisible by A
                    if (Z + B) % A == 0:
                        z = (Z + B) // A
                        y = (Y + B) // A
                        # Check distinctness and bounds
                        if y > x and z > y and z <= LIMIT:
                            # x, y, z are distinct
                            found = [x, y, z]
                            break
            if found:
                break
        
        answers[n - 3] = found

    return answers
