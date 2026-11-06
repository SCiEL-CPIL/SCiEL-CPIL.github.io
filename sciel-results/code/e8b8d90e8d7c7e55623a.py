"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd

def construct():
    deadline = time.time() + 85.0
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        x_min = n // 4 + 1
        
        # We need x < y < z (distinct, positive)
        # 1/x + 1/y + 1/z = 4/n
        # For a given x, we have 1/y + 1/z = 4/n - 1/x = (4x - n) / (nx)
        # Let a = 4x - n, b = nx
        # 1/y + 1/z = a/b => b(y+z) = ayz => ayz - by - bz = 0
        # Multiply by a: a^2 yz - ab y - ab z = 0
        # Add b^2: (ay - b)(az - b) = b^2
        # Let A = ay - b, B = az - b. Then A*B = b^2.
        # A < B implies y < z.
        # Also y > x implies ay - b > ax - b => A > ax - b.
        # Since x < y, we need A > ax - b.
        # Also y, z are integers, so A and B must be divisors of b^2.
        # y = (A + b) / a, z = (B + b) / a.
        # So (A + b) % a == 0 and (B + b) % a == 0.
        
        # To maximize chance of finding a solution quickly, iterate x from x_min upwards.
        # For each x, iterate divisors A of b^2 such that A < b (since y < z => A < B => A^2 < b^2 => A < b).
        # Actually, A*B = b^2 and A <= B implies A <= b.
        # We need A > ax - b.
        
        # However, iterating all divisors of b^2 for each x might be slow if b is large.
        # Alternative: iterate y from x+1 to some limit.
        # 1/y + 1/z = a/b
        # 1/z = a/b - 1/y = (ay - b) / (by)
        # z = by / (ay - b)
        # We need ay - b > 0 => y > b/a.
        # Also z > y => by / (ay - b) > y => b / (ay - b) > 1 => b > ay - b => 2b > ay => y < 2b/a.
        # So y ranges from floor(b/a) + 1 to ceil(2b/a) - 1.
        # This range is roughly b/a = nx / (4x - n).
        # For x = x_min = n//4 + 1, 4x - n is small (1 or 2), so b/a is large.
        # Specifically, if n is even, n=4k, x_min=k+1, a=4(k+1)-4k=4, b=4k(k+1). b/a = k(k+1).
        # If n is odd, n=4k+1, x_min=k+1, a=4(k+1)-(4k+1)=3, b=(4k+1)(k+1). b/a ~ 4k^2/3.
        # The range size is about b/a.
        # For n=256, x_min=65, a=4, b=256*65=16640. b/a=4160. Range ~4160.
        # For n=3, x_min=1, a=1, b=3. b/a=3. Range ~3.
        # This loop is very fast.
        
        for x in range(x_min, 10000): # x won't be that large, but safe upper bound
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # y must satisfy: b/a < y < 2b/a
            # y > b/a => y >= floor(b/a) + 1
            # y < 2b/a => y <= ceil(2b/a) - 1 = (2b + a - 1) // a - 1
            
            y_min = b // a + 1
            y_max = (2 * b + a - 1) // a - 1
            
            # Also y > x
            if y_min <= x:
                y_min = x + 1
            
            if y_min > y_max:
                continue
                
            # Iterate y
            for y in range(y_min, y_max + 1):
                denom = a * y - b
                if denom <= 0:
                    continue
                if (b * y) % denom == 0:
                    z = (b * y) // denom
                    if z > y and z <= 10**12:
                        # Check distinctness
                        if x != y and y != z and x != z:
                            found = [x, y, z]
                            break
            if found:
                break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
