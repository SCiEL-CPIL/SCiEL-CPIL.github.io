import time

def construct():
    deadline = time.time() + 75.0
    answers = [None] * 254
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        found = None
        # Try various x values starting from the minimum
        x_min = n // 4 + 1
        # x must satisfy 4x > n, so x >= n//4 + 1
        # Also we want x, y, z distinct positive integers
        for x in range(x_min, min(x_min + 200, 1000000000000)):
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            # We need (ay - b)(az - b) = b^2
            # Let d be a divisor of b^2, then y = (d + b) / a, z = (b^2/d + b) / a
            # We need y and z to be positive integers, distinct from each other and from x
            # Also y, z <= 10^12
            b2 = b * b
            # Find divisors of b^2
            # b can be large, so we factor b first
            # For efficiency, iterate over possible d values
            # y = (d + b) / a must be integer, so (d + b) % a == 0
            # z = (b2 // d + b) / a must be integer, so (b2 // d + b) % a == 0
            # And d must divide b2
            # Let's iterate d over divisors of b^2
            # But b^2 can be huge. Instead, iterate over divisors of b, since if d | b^2,
            # we can write d = d1 * d2 where d1 | b and d2 | b, but that's complex.
            # Alternative: iterate y in a smart range.
            # y = (d + b) / a where d | b^2
            # So y * a - b = d, and d | b^2, so (y*a - b) | b^2
            # And z = (b^2 / (y*a - b) + b) / a
            # Let's just try y values in a reasonable range
            # y must be > x (for distinctness, we can assume wlog x < y < z)
            # y > b / a (since d > 0 means y*a > b)
            # And y*a - b must divide b^2
            # Upper bound on y: z > y, so z >= y+1
            # z = (b^2/(y*a-b) + b)/a > y
            # This is getting complex. Let's use the divisor approach on b.
            
            # Factor b to get its divisors
            # b = n * x
            # For small n and x, b is manageable
            # Let's try a different approach: iterate y from x+1 upward
            # and check if (y*a - b) divides b^2
            y_start = max(x + 1, b // a + 1)
            # Upper bound: when y*a - b = 1, y = (b+1)/a, and z = b^2 + b over a
            # As y increases, d = y*a - b increases, and z decreases
            # We need z > y, so (b^2/d + b)/a > y → b^2/d + b > a*y → b^2/d > a*y - b = d → b^2 > d^2 → d < b
            # So we need d < b, meaning y*a - b < b, meaning y*a < 2b, meaning y < 2b/a
            y_end = min(2 * b // a, 1000000000000)
            
            if y_end < y_start:
                continue
                
            for y in range(y_start, y_end + 1):
                d = y * a - b
                if d <= 0:
                    continue
                if b2 % d != 0:
                    continue
                z = (b2 // d + b) // a
                if (b2 // d + b) % a != 0:
                    continue
                if z <= y:
                    continue
                if z == x or y == x:
                    continue
                if z > 1000000000000 or y > 1000000000000 or x > 1000000000000:
                    continue
                # Verify: 1/x + 1/y + 1/z == 4/n
                # Cross multiply: n*(y*z + x*z + x*y) == 4*x*y*z
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            if found:
                break
        answers[idx] = found
    
    return answers
# EVOLVE-BLOCK-END
