import time
from math import gcd

def construct():
    deadline = time.time() + 75.0
    answers = [None] * 254
    BOUND = 10**12

    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        found = None
        x_min = n // 4 + 1
        
        # We need 1/x + 1/y + 1/z = 4/n
        # Multiply by xyz: n(yz + xz + xy) = 4xyz
        # Rearranging for z:
        # nyz + nxz + nxy = 4xyz
        # nxy = z(4xy - ny - nx)
        # z = nxy / (4xy - n(x+y))
        
        # For z to be an integer, the denominator D = 4xy - n(x+y) must divide nxy.
        # Also D > 0.
        # 4xy - nx - ny > 0
        # (4x - n)(4y - n) = 16xy - 4nx - 4ny + n^2
        # D = 4xy - nx - ny = (16xy - 4nx - 4ny + n^2 - n^2) / 4 = ((4x-n)(4y-n) - n^2) / 4
        # So 4D = (4x-n)(4y-n) - n^2.
        
        # Let A = 4x - n. Since x >= n//4 + 1, A >= 1 (actually A >= 1 if n%4==0, else higher, but generally positive).
        # A = 4x - n
        # 4D = A(4y - n) - n^2
        # Let B = 4y - n. Then 4D = AB - n^2.
        # So AB = 4D + n^2.
        # Also D = nxy / z. Since z is integer, D | nxy.
        # Note that gcd(D, x) = gcd(4xy - nx - ny, x) = gcd(-ny, x) = gcd(ny, x).
        # gcd(D, y) = gcd(4xy - nx - ny, y) = gcd(-nx, y) = gcd(nx, y).
        
        # A more efficient search strategy:
        # Iterate x. For each x, iterate y.
        # To speed up, we can restrict the range of y.
        # Since x <= y <= z, we have 3/x >= 4/n => x <= 3n/4.
        # Also 1/x + 1/y + 1/z = 4/n.
        # If y is too small, z becomes too large or negative.
        # D = 4xy - nx - ny > 0 => y(4x - n) > nx => y > nx / (4x - n).
        # Also z > y => nxy / D > y => nx / D > 1 => D < nx.
        # 4xy - nx - ny < nx => 4xy - ny < 2nx => y(4x - n) < 2nx => y < 2nx / (4x - n).
        
        # So for a fixed x, y is in range (nx/(4x-n), 2nx/(4x-n)).
        # Let L = floor(nx / (4x - n)) + 1
        # Let R = floor(2nx / (4x - n))
        
        # Also we need y >= x.
        
        # To further optimize, we can use the divisibility condition.
        # D = 4xy - nx - ny.
        # We need D | nxy.
        
        # Let's iterate x from x_min up to some limit.
        # The limit for x: since x <= y <= z, 3/x >= 4/n => x <= 3n/4.
        # So x_max = 3*n // 4.
        
        # For each x, we iterate y.
        # To make this fast, we should check the divisibility quickly.
        
        # Alternative approach:
        # 1/z = 4/n - 1/x - 1/y = (4xy - nx - ny) / nxy = D / nxy
        # z = nxy / D.
        # We need D | nxy.
        
        # Let's try iterating x up to a reasonable bound, and for each x, iterate y in the derived range.
        # If the range is small, this is fast.
        
        x_limit = 3 * n // 4
        # But we also know x must be at least n//4 + 1.
        
        # To ensure we don't miss solutions with large x, we might need to go higher?
        # No, x <= 3n/4 is a hard bound for sorted x<=y<=z.
        
        # However, if n is small, 3n/4 is small. If n is large, it's large.
        # For n=256, x can be up to 192.
        # The range of y for each x is roughly of size nx/(4x-n).
        # For x close to n/4, 4x-n is small, so the range is large.
        # For x close to 3n/4, 4x-n is large, range is small.
        
        # Let's implement this.
        
        solved = False
        for x in range(x_min, x_limit + 1):
            if solved:
                break
            A = 4 * x - n
            if A <= 0:
                continue
            
            # y > nx / A
            y_min = (n * x) // A + 1
            if y_min < x:
                y_min = x
            
            # y < 2nx / A
            y_max = (2 * n * x) // A
            
            # We also need z > y, which is handled by y < 2nx/A.
            # We also need z <= BOUND.
            # z = nxy / D.
            # D = 4xy - nx - ny.
            # If z > BOUND, we can break?
            # As y increases, D increases, so z decreases.
            # So if z > BOUND for the smallest y, it might be even larger for smaller y (but we start at min).
            # Actually, z is decreasing with y. So the maximum z is at y_min.
            # If z_min_y > BOUND, then no solution for this x?
            # Let's check z at y_min.
            
            for y in range(y_min, y_max + 1):
                if time.time() > deadline:
                    return answers
                
                D = 4 * x * y - n * x - n * y
                # D is positive by construction of y_min
                if D <= 0:
                    continue
                
                # Check divisibility
                num = n * x * y
                if num % D != 0:
                    continue
                
                z = num // D
                
                if z > BOUND:
                    continue
                
                # Check distinctness and order
                if x != y and y != z and x != z:
                    # We assumed x <= y <= z. 
                    # The problem doesn't require order, just distinct.
                    # But our derivation assumed x <= y <= z to bound the search.
                    # If we find a solution with x <= y <= z, it is a valid set of distinct denominators.
                    found = [x, y, z]
                    solved = True
                    break
        
        answers[idx] = found

    return answers
