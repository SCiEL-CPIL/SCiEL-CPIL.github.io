import time

def construct():
    answers = []
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    for n in range(3, 257):
        found = None
        # We search for x in a reasonable range.
        # x must be > n/4.
        # Usually small x works. Let's try up to n + 100.
        # If not found, we might need to go higher, but small x is most likely.
        
        x_start = n // 4 + 1
        # To ensure we don't miss solutions, we might need to go higher, 
        # but for n <= 256, solutions with small x are abundant.
        # Let's iterate x up to n + 1000 just in case, but break early if found.
        
        for x in range(x_start, n + 1000):
            if time.time() > deadline:
                break
            
            A = 4 * x - n
            B = n * x
            
            # We need (Ay - B)(Az - B) = B^2
            # Let U = Ay - B. Then V = B^2 / U.
            # y = (U + B) / A, z = (V + B) / A
            # Constraints:
            # 1. y >= x  => (U+B)/A >= x => U >= Ax - B
            #    Ax - B = x(4x-n) - nx = 4x^2 - 2nx
            #    So U >= 4x^2 - 2nx
            # 2. z >= y => V >= U => B^2/U >= U => U^2 <= B^2 => U <= B
            # 3. U must divide B^2
            # 4. (U+B) % A == 0
            # 5. (V+B) % A == 0
            # 6. x, y, z distinct
            
            lower_bound = 4 * x * x - 2 * n * x
            if lower_bound < 1:
                lower_bound = 1
            
            # Iterate U from lower_bound to B
            # This loop is O(B) which is O(nx). For n=256, x~64, B~16000. Very fast.
            for U in range(lower_bound, B + 1):
                if (B * B) % U != 0:
                    continue
                
                V = (B * B) // U
                
                # Check divisibility for y
                if (U + B) % A != 0:
                    continue
                y = (U + B) // A
                
                # Check divisibility for z
                if (V + B) % A != 0:
                    continue
                z = (V + B) // A
                
                # Check distinctness and positivity
                # y, z are positive because U, V > 0 and B > 0.
                # U >= 1, V >= 1.
                # A > 0 since x > n/4.
                
                if x == y or x == z or y == z:
                    continue
                
                # Check bounds
                if y > 10**12 or z > 10**12:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
