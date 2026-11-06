"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0
    answers = []
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        found = None
        # Try multiple values of x starting from the minimum
        x_min = n // 4 + 1
        # x must satisfy 4*x > n for a = 4*x - n > 0
        # Also x should be reasonable; try a range of x values
        for x in range(x_min, x_min + 200):
            if time.time() > deadline:
                break
            a = 4 * x - n  # must be > 0
            b = n * x
            # We need (a*y - b)(a*z - b) = b^2
            # Let d = a*y - b, then a*z - b = b^2/d
            # y = (d + b) / a, z = (b^2/d + b) / a
            # Need d | b^2, d < b^2, y > x, z > y, x,y,z distinct
            # y > x => (d+b)/a > x => d+b > a*x => d > a*x - b = (4x-n)*x - nx = 4x^2 - nx - nx = 4x^2 - 2nx
            # z > y => b^2/d + b > d + b => b^2/d > d => d^2 < b^2 => d < b (since d>0)
            # So we need d | b^2, d < b, and d > 4x^2 - 2nx
            # Also y and z must be positive integers, and all three distinct.
            #
            # Alternative: iterate over divisors of b^2 that are < b
            # But b^2 can be large. Instead, iterate d from max(1, 4*x^2 - 2*n*x + 1) to b-1
            # and check if b^2 % d == 0.
            #
            # Actually, a simpler approach: for each possible y > x, check if z is integer.
            # From a/y + a/z = a/b... wait, let me re-derive.
            # 1/y + 1/z = a/b
            # (y+z)/(yz) = a/b
            # b(y+z) = a*yz
            # a*yz - b*y - b*z = 0
            # (ay-b)(az-b) = b^2
            #
            # Let's just iterate y in a reasonable range.
            # y must be > x. Also from 1/y + 1/z = a/b with z > y:
            # 1/y + 1/z < 2/y, so a/b < 2/y => y < 2b/a
            # And 1/y + 1/z > 1/y, so a/b > 1/y => y > b/a
            # So y is in range (b/a, 2b/a)
            y_lo = b // a + 1
            if y_lo <= x:
                y_lo = x + 1
            y_hi = 2 * b // a
            # Limit the search range to avoid excessive time
            if y_hi - y_lo > 50000:
                y_hi = y_lo + 50000
            for y in range(y_lo, y_hi + 1):
                # Check if (a*y - b) divides (b*y) appropriately
                # From (ay-b)(az-b) = b^2, we have az - b = b^2 / (ay - b)
                # So z = (b^2 / (ay - b) + b) / a
                denom = a * y - b
                if denom <= 0:
                    continue
                if (b * b) % denom != 0:
                    continue
                z_num = (b * b) // denom + b
                if z_num % a != 0:
                    continue
                z = z_num // a
                if z <= y:
                    continue
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                # Check all positive and within bounds
                if x < 1 or y < 1 or z < 1:
                    continue
                if x > 10**12 or y > 10**12 or z > 10**12:
                    continue
                # Verify the equation
                if n * y * z + n * x * z + n * x * y == 4 * x * y * z:
                    found = [x, y, z]
                    break
            if found:
                break
        answers.append(found)
    return answers
# EVOLVE-BLOCK-END
