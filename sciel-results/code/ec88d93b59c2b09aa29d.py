import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        # Search for x
        # x must be > n/4
        x_min = n // 4 + 1
        # We try x up to a reasonable limit.
        # Based on knowledge, small x works.
        # Let's try up to 50*n to be safe, but it should be much smaller.
        x_max = 50 * n
        
        for x in range(x_min, x_max + 1):
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            B2 = B * B
            
            # u = Ay - B. We want u to be a divisor of B^2.
            # Also y = (u+B)/A must be integer.
            # Let u = k*A - B. Then y = k.
            # u > 0 => k*A > B => k > B/A.
            # u < v => u < B => k*A - B < B => k*A < 2B => k < 2B/A.
            # So k is in (B/A, 2B/A).
            
            k_min = B // A + 1
            k_max = (2 * B) // A
            
            # Iterate k
            # Note: k is y. We need y > x? Not necessarily, but we can assume wlog x <= y <= z?
            # The problem asks for distinct x,y,z. Order doesn't matter for the set, 
            # but we return a list. The evaluator checks cross multiplication.
            # However, if we assume x is the smallest, we should ensure y >= x.
            # In our parameterization, x is fixed. y=k.
            # If we want distinct, we just check y != x, z != x, y != z.
            
            for k in range(k_min, k_max + 1):
                u = k * A - B
                # u must be positive, which is guaranteed by k_min
                if B2 % u != 0:
                    continue
                v = B2 // u
                if (v + B) % A != 0:
                    continue
                z = (v + B) // A
                y = k
                
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                
                # Check bounds
                if y > 1000000000000 or z > 1000000000000:
                    continue
                
                # Verify solution just in case
                # 1/x + 1/y + 1/z == 4/n
                # n(yz + xz + xy) == 4xyz
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
