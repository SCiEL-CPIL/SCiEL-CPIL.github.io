import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    LIMIT = 10**12
    answers = [None] * 254
    
    def verify(n, x, y, z):
        if x is None or y is None or z is None:
            return False
        if x <= 0 or y <= 0 or z <= 0:
            return False
        if x > LIMIT or y > LIMIT or z > LIMIT:
            return False
        if x == y or y == z or x == z:
            return False
        lhs = (y*z + x*z + x*y) * n
        rhs = 4 * x * y * z
        return lhs == rhs

    def find_pair_given_x(n, x):
        """Find y,z distinct from x and each other such that 1/y+1/z = (4x-n)/(nx)"""
        c = 4 * x - n
        d = n * x
        if c <= 0:
            return None
        
        # y range: d/c < y < 2d/c
        # y must be integer, so y_min = floor(d/c) + 1, y_max = ceil(2d/c) - 1
        # But we also need y > x for distinctness (or handle ordering)
        
        # 1/y + 1/z = c/d => z = d*y / (c*y - d)
        # Need c*y - d > 0 => y > d/c
        # Need z > y => d*y/(c*y-d) > y => d/(c*y-d) > 1 => d > c*y - d => 2d > c*y => y < 2d/c
        
        y_min = d // c + 1
        # y_max: y < 2d/c, so y_max = ceil(2d/c) - 1
        if (2 * d) % c == 0:
            y_max = (2 * d) // c - 1
        else:
            y_max = (2 * d) // c
        
        if y_min > y_max:
            return None
        
        # Try y values, prefer ones that give distinct z
        # Also skip y == x
        for y in range(y_min, y_max + 1):
            if y == x:
                continue
            denom = c * y - d
            if denom <= 0:
                continue
            if (d * y) % denom != 0:
                continue
            z = (d * y) // denom
            if z <= 0 or z > LIMIT:
                continue
            if z == y or z == x:
                continue
            # Found a valid pair
            return (y, z)
        
        return None

    for n in range(3, 257):
        if time.time() > deadline:
            break
            
        found = None
        
        # Try x values. x must satisfy n/4 < x < 3n/4 (roughly)
        # More precisely: 1/x < 4/n => x > n/4
        # And 1/x > 4/n - 2/(some large) but for existence, x < 3n/4 works
        # Actually, we need 4/n - 1/x > 0, so x > n/4
        # And 4/n - 1/x must be expressible as sum of two unit fractions
        # Max of 1/y+1/z is 2/min(y,z), so we need 4/n - 1/x <= 2
        # This is always true for x > n/4 when n>=3
        
        x_min = n // 4 + 1
        x_max = 3 * n // 4
        
        # Try x values in an order that's likely to succeed quickly
        # Start with x close to n/4 (small x gives larger c/d, easier to split)
        # Also try x = n/2 if integer
        x_candidates = list(range(x_min, x_max + 1))
        
        # Sort to try promising x first
        # x = n/2 is often good
        if n % 2 == 0:
            x_half = n // 2
            if x_min <= x_half <= x_max:
                x_candidates.remove(x_half)
                x_candidates.insert(0, x_half)
        
        for x in x_candidates:
            if time.time() > deadline:
                break
                
            pair = find_pair_given_x(n, x)
            if pair is not None:
                y, z = pair
                # Ensure all three are distinct and positive
                if x != y and y != z and x != z and x > 0 and y > 0 and z > 0:
                    found = [x, y, z]
                    break
        
        if found:
            if not verify(n, *found):
                found = None
        
        answers[n - 3] = found

    return answers
