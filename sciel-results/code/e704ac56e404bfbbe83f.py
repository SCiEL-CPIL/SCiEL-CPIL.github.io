"""
Rewritten program for distinct three-term unit-fraction representations of 4/n.
Uses a smarter search: for each n, iterate x from n//4+1 upward, and for each x,
iterate y from a lower bound upward, checking if z is a valid integer > y.
Includes a deadline guard to ensure completion within the time limit.
"""

import time

def construct():
    deadline = time.time() + 85.0  # 5-second safety margin before 90s kill
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x must satisfy: 1/x < 4/n => x > n/4
        # Also, for distinct positive denominators with x <= y <= z:
        # 3/x >= 4/n => x <= 3n/4 (since 1/x+1/y+1/z <= 3/x)
        # So x is in range (n/4, 3n/4]
        
        x_start = n // 4 + 1
        x_end = (3 * n) // 4 + 1  # inclusive upper bound
        
        for x in range(x_start, min(x_end, x_start + 10000)):  # cap x iterations
            if time.time() > deadline:
                break
            
            # We need: 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
            # Let A = 4x - n, B = nx
            # So 1/y + 1/z = A/B
            # This means (y+z)/(yz) = A/B => B(y+z) = A*y*z
            # => A*y*z - B*y - B*z = 0
            # => (A*y - B)(A*z - B) = B^2
            # Let u = A*y - B, v = A*z - B, then u*v = B^2
            # y = (u+B)/A, z = (v+B)/A
            # We need u, v to be positive divisors of B^2 such that
            # (u+B) % A == 0 and (v+B) % A == 0
            # Also y > x (distinct, and we're searching in order x < y < z)
            # and z > y
            
            A = 4 * x - n
            if A <= 0:
                continue
            
            B = n * x
            B_sq = B * B
            
            # y must be > x, so (u+B)/A > x => u > A*x - B = A*x - n*x = x*(A-n) = x*(4x-2n)
            # Also from 1/y + 1/z = A/B and y < z, we have 2/y > A/B => y < 2B/A
            # So y is in range (x, 2B/A)
            
            # Lower bound for y: y > x
            # Upper bound for y: y < 2*B/A (since 1/y > A/(2B) when y<z)
            y_start = x + 1
            y_end = (2 * B) // A  # y must be < 2B/A, so y <= 2B/A - 1, but let's use floor
            
            # Cap the search range for y to avoid infinite loops
            max_y_iters = 100000
            if y_end - y_start > max_y_iters:
                y_end = y_start + max_y_iters
            
            for y in range(y_start, y_end + 1):
                if time.time() > deadline:
                    break
                
                # Check if z = B*y / (A*y - B) is an integer > y
                denom = A * y - B
                if denom <= 0:
                    continue
                
                if (B * y) % denom == 0:
                    z = (B * y) // denom
                    if z > y and z <= 1000000000000:
                        found = [x, y, z]
                        break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
