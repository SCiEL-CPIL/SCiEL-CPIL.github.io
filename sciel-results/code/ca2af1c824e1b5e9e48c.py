import time
import math

def construct():
    deadline = time.time() + 75.0
    answers = [None] * 254
    BOUND = 10**12
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        found = None
        
        # We need 1/x + 1/y + 1/z = 4/n with x < y < z, all distinct positive integers
        # x must be > n/4, so x_min = n//4 + 1
        x_min = n // 4 + 1
        
        # Try multiple x values
        # For each x, we have 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
        # Let a = 4x - n, b = nx
        # Then 1/y + 1/z = a/b => b(y+z) = a*y*z => a*y*z - b*y - b*z = 0
        # => (a*y - b)(a*z - b) = b^2
        # So we need factor pairs (d, e) of b^2 such that d*e = b^2,
        # y = (d+b)/a, z = (e+b)/a, and both must be positive integers.
        # Also need y > x and z > y for distinctness and ordering.
        
        # Strategy: iterate x from x_min upward, and for each x,
        # find factor pairs of b^2 where d = a*y - b, so y = (d+b)/a must be integer.
        # This means (d+b) % a == 0, i.e., d ≡ -b (mod a), i.e., d ≡ b (mod a) since -b ≡ b mod a only if 2b ≡ 0... 
        # Wait: d ≡ -b (mod a). So d = k*a - b for some k, and d*e = b^2.
        # So (k*a - b)*e = b^2 => e = b^2 / (k*a - b)
        # We need k*a - b > 0, so k > b/a. Also e must be positive integer.
        # And y = (d+b)/a = (k*a - b + b)/a = k. So y = k!
        # And z = (e+b)/a.
        
        # So the approach: for each x, let a = 4x-n, b = nx.
        # We need to find integer k > b/a such that (k*a - b) divides b^2.
        # Then y = k, z = (b^2/(k*a-b) + b)/a.
        # Check: y > x, z > y, all distinct, all <= BOUND.
        
        # k must be > b/a. Also y = k must be > x.
        # Let's iterate k from max(x+1, floor(b/a)+1) upward, but we need to bound it.
        # z = (b^2/(k*a-b) + b)/a. As k increases, k*a-b increases, so b^2/(k*a-b) decreases.
        # z > y = k requires (b^2/(k*a-b) + b)/a > k => b^2/(k*a-b) > a*k - b => b^2 > (a*k-b)^2
        # So (a*k - b) < b, i.e., a*k < 2b, i.e., k < 2b/a.
        # So k ranges from max(x+1, floor(b/a)+1) to ceil(2b/a) - 1.
        
        # For each such k, check if (k*a - b) divides b^2.
        
        solved = False
        # Try x values
        max_x_tries = 200
        for x in range(x_min, x_min + max_x_tries):
            if time.time() > deadline:
                break
            a = 4 * x - n
            b = n * x
            if a <= 0:
                continue
            
            # k range
            k_lo = max(x + 1, b // a + 1)
            k_hi = (2 * b) // a  # k < 2b/a, so k <= floor(2b/a) - 1 if 2b/a not integer, else floor(2b/a) - 1
            # Actually k < 2b/a strictly, so k <= ceil(2b/a) - 1
            k_hi = (2 * b - 1) // a  # largest integer < 2b/a
            
            if k_hi < k_lo:
                continue
            
            # Iterate k
            for k in range(k_lo, k_hi + 1):
                if time.time() > deadline:
                    break
                d = k * a - b
                if d <= 0:
                    continue
                if b * b % d != 0:
                    continue
                e = b * b // d
                z = (e + b) // a
                # Check z is integer
                if (e + b) % a != 0:
                    continue
                y = k
                # Verify: y > x, z > y, all distinct
                if y <= x or z <= y:
                    continue
                if y == z:
                    continue
                # Check bounds
                if x > BOUND or y > BOUND or z > BOUND:
                    continue
                # Verify the equation: 1/x + 1/y + 1/z == 4/n
                # Cross multiply: n*(y*z + x*z + x*y) == 4*x*y*z
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    solved = True
                    break
            if solved:
                break
        
        answers[idx] = found
    
    return answers
