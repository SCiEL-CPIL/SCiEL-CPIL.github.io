import time

def construct():
    deadline = time.time() + 55.0
    answers = [None] * 254
    BOUND = 10**12
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        found = None
        
        # x must be > n/4
        x_min = n // 4 + 1
        # x must be < 3n/4 for y, z to be > x
        # 4/n = 1/x + 1/y + 1/z < 3/x => x < 3n/4
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            if found:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # 1/y + 1/z = A/B
            # y > x
            # y > B/A
            # z > y => 1/z < 1/y => 1/y + 1/z < 2/y => A/B < 2/y => y < 2B/A
            y_min = max(B // A + 1, x + 1)
            y_max = (2 * B) // A
            
            # Optimization: If y_min > y_max, no solution for this x
            if y_min > y_max:
                continue
                
            for y in range(y_min, y_max + 1):
                # D = A*y - B
                # z = B*y / D
                # Need D | B*y
                D = A * y - B
                if D <= 0:
                    continue
                if (B * y) % D != 0:
                    continue
                z = (B * y) // D
                # Check distinctness and bounds
                # We know y > x. We need z > y and z != x (implied by z > y > x)
                if z > y and z <= BOUND:
                    # Verify equation just in case
                    # 1/x + 1/y + 1/z == 4/n
                    # n(yz + xz + xy) == 4xyz
                    lhs = n * (y * z + x * z + x * y)
                    rhs = 4 * x * y * z
                    if lhs == rhs:
                        found = [x, y, z]
                        break
        answers[idx] = found
        
    return answers
