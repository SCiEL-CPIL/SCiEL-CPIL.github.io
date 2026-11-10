import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    LIMIT = 10**12
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        # x must satisfy 1/x < 4/n => x > n/4
        # and 3/x >= 4/n => x <= 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            A = 4 * x - n
            B = n * x
            if A <= 0:
                continue
                
            # We need (Ay - B)(Az - B) = B^2
            # Let u = Ay - B, v = Az - B. u * v = B^2.
            # y = (u + B) / A, z = (v + B) / A.
            # We require x < y < z.
            # y > x => (u + B) / A > x => u > Ax - B.
            # z > y => v > u.
            # Also u, v must be such that y, z are integers.
            # u + B must be divisible by A.
            # u = k * A - B for some integer k? No, u ≡ -B (mod A).
            
            # Iterate through divisors of B^2
            # To speed up, we can just generate divisors of B^2
            # B is at most 256 * 192 ~ 50000. B^2 ~ 2.5e9.
            # Generating divisors is fast.
            
            # Optimization: u must be >= A*x - B + 1 (since y > x)
            # And u < B^2 (since v > u implies u < sqrt(B^2))
            
            lower_u = A * x - B + 1
            if lower_u < 1:
                lower_u = 1
                
            # Find divisors of B^2
            # We can iterate i from 1 to sqrt(B^2)
            # But B^2 can be large. 
            # Alternatively, factor B, then generate divisors of B^2.
            
            # Simpler: just iterate y in the range?
            # Range of y is B/A to 2B/A.
            # If B/A is small, this is fine.
            # B/A = nx / (4x-n).
            # For x = n/4 + 1, 4x-n = 4(n/4+1)-n = 4.
            # B/A = n(n/4+1)/4 ~ n^2/16.
            # For n=256, n^2/16 = 4096.
            # Iterating 4096 values is very fast.
            # So the y-iteration is perfectly fine.
            
            y_min = max(x + 1, B // A + 1)
            y_max = (2 * B) // A
            
            for y in range(y_min, y_max + 1):
                den = A * y - B
                if den <= 0:
                    continue
                if (B * y) % den == 0:
                    z = (B * y) // den
                    if z > y and z <= LIMIT:
                        found = [x, y, z]
                        break
            if found:
                break
        answers.append(found)
    return answers
