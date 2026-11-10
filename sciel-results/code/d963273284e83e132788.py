"""Solve distinct three-term unit-fraction representations of 4/n for n=3..256."""
import time

def construct():
    MAX_DENOM = 10**12
    answers = [None] * 254  # index 0 corresponds to n=3
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        
        found = None
        
        # We need 1/x + 1/y + 1/z = 4/n with x < y < z (distinct positive integers)
        # From the equation: (4x-n)(4y-n)(4z-n) ... let's use a simpler approach
        # 1/x + 1/y = 4/n - 1/z = (4z-n)/(nz)
        # So for a given z, we need 1/x + 1/y = (4z-n)/(nz)
        # Let A = 4z-n, B = nz. We need 1/x + 1/y = A/B
        # (x+y)/(xy) = A/B => B(x+y) = Axy => Axy - Bx - By = 0
        # (Ax-B)(Ay-B) = B^2
        # So we need factor pairs of B^2 where both factors are divisible by A in the right way.
        
        # Actually, let's use the standard approach:
        # For given x, we have 1/y + 1/z = 4/n - 1/x = (4x-n)/(nx)
        # Let a = 4x-n, b = nx. We need 1/y + 1/z = a/b
        # (ay-b)(az-b) = b^2
        # So (ay-b) and (az-b) are factor pairs of b^2.
        # Let d1 = ay-b, d2 = az-b, with d1*d2 = b^2, d1 < d2 (since y < z)
        # Then y = (d1+b)/a, z = (d2+b)/a
        # We need a | (d1+b) and a | (d2+b), and y, z positive integers with y > x, z > y.
        
        # For x, we need 4x > n (so a > 0), i.e., x > n/4.
        # Also 1/x < 4/n => x > n/4.
        # And we need y > x, so 1/y < 1/x, meaning 1/z < 4/n - 2/x.
        # This requires 4/n - 2/x > 0, i.e., x > n/2.
        # Wait, that's not quite right. Let me reconsider.
        # We need 1/x + 1/y + 1/z = 4/n with x < y < z.
        # So 1/x < 4/n => x > n/4.
        # And 1/x + 1/y < 4/n (since 1/z > 0) => 1/y < 4/n - 1/x.
        # This requires 4/n - 1/x > 0, i.e., x > n/4 (already have this).
        # Also 1/x + 1/y > 4/n - 1/z > 4/n - 1/y (since z > y => 1/z < 1/y)
        # So 1/x + 2/y > 4/n.
        
        # Let's just iterate x from floor(n/4)+1 upward, and for each x,
        # find y > x such that z = 1/(4/n - 1/x - 1/y) is an integer > y.
        
        # For given x: a = 4x - n > 0, b = n*x
        # 1/y + 1/z = a/b
        # (ay-b)(az-b) = b^2
        # d1 = ay - b, d2 = az - b, d1*d2 = b^2, d1 < d2
        # y = (d1+b)/a must be integer > x
        # z = (d2+b)/a must be integer > y
        
        # Since d1*d2 = b^2 and d1 < d2, we have d1 < b < d2.
        # d1 ranges over divisors of b^2 that are < b.
        # But b can be large. Let's bound x.
        
        # x > n/4, and 1/x + 1/(x+1) + 1/(x+2) > 4/n (necessary for a solution with x as smallest)
        # Actually, we need 1/x + 1/y + 1/z = 4/n with y >= x+1, z >= x+2.
        # So 4/n <= 1/x + 1/(x+1) + 1/(x+2)
        # This gives an upper bound on x.
        
        # For small n, x is small. For n=3, x > 3/4, so x >= 1.
        # 4/3 <= 1/x + 1/(x+1) + 1/(x+2). For x=1: 1+1/2+1/3 = 11/6 > 4/3. OK.
        # For x=2: 1/2+1/3+1/4 = 13/12 < 4/3. So x=1 is the only option for n=3.
        
        # Let's compute the max x for each n.
        # 4/n <= 3/x (roughly, since y,z > x)
        # So x <= 3n/4.
        # More precisely, 4/n <= 1/x + 1/(x+1) + 1/(x+2)
        # Let's just iterate x from max(1, n//4+1) to some reasonable upper bound.
        
        x_min = max(1, n // 4 + 1)
        # Upper bound: 4/n <= 1/x + 1/(x+1) + 1/(x+2) < 3/x, so x < 3n/4
        # But also we need 4/n - 1/x > 0 for there to be room for 1/y+1/z.
        # And 4/n - 1/x <= 2/(x+1) for there to be a solution with y >= x+1.
        # So 4/n - 1/x <= 2/(x+1)
        # (4x-n)/(nx) <= 2/(x+1)
        # (4x-n)(x+1) <= 2nx
        # 4x^2 + 4x - nx - n <= 2nx
        # 4x^2 + 4x - 3nx - n <= 0
        # 4x^2 + (4-3n)x - n <= 0
        
        # Let's just try x from x_min to min(3*n//4 + 1, 10000) and see.
        # For n up to 256, 3n/4 is at most 192. So x is at most around 192.
        # That's very manageable.
        
        x_max = min(3 * n // 4 + 2, 200)
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # We need (ay-b)(az-b) = b^2 with y > x, z > y
            # d1 = ay - b, d2 = az - b
            # d1 must be a divisor of b^2, d1 < b, and d1 ≡ -b (mod a), i.e., d1 + b ≡ 0 (mod a)
            # Also y = (d1+b)/a > x => d1+b > ax => d1 > ax - b = a(x) - b = 4x^2 - nx - nx = 4x^2 - 2nx
            # Wait, ax - b = a*x - b = (4x-n)*x - nx = 4x^2 - nx - nx = 4x^2 - 2nx
            # So d1 > 4x^2 - 2nx
            
            # Let's find divisors d1 of b^2 such that:
            # 1. d1 < b
            # 2. d1 + b is divisible by a
            # 3. y = (d1+b)/a > x, i.e., d1 > ax - b
            # 4. z = (d2+b)/a > y where d2 = b^2/d1, i.e., d2 + b > a*y = d1 + b, i.e., d2 > d1 (already true)
            #    and z > y means (d2+b)/a > (d1+b)/a, i.e., d2 > d1 (true)
            #    Also z must be integer: d2 + b must be divisible by a.
            #    Since d1*d2 = b^2 and d1+b ≡ 0 (mod a), we need d2+b ≡ 0 (mod a).
            #    d2 = b^2/d1. We need b^2/d1 + b ≡ 0 (mod a).
            #    Multiply by d1: b^2 + b*d1 ≡ 0 (mod a).
            #    But d1 ≡ -b (mod a), so b*d1 ≡ -b^2 (mod a).
            #    So b^2 + (-b^2) = 0 ≡ 0 (mod a). This is always true!
            #    So if d1 + b ≡ 0 (mod a), then automatically d2 + b ≡ 0 (mod a).
            
            # So we just need: d1 is a divisor of b^2, d1 < b, d1 > ax - b, and d1 ≡ -b (mod a).
            
            # b = n*x, a = 4x-n
            # Let's iterate over possible d1 values.
            # d1 ranges from max(1, ax-b+1) to b-1, with d1 ≡ -b (mod a).
            # And d1 must divide b^2.
            
            # The step is a, so we check d1 = (ax-b+1) adjusted to the right residue class.
            # d1 ≡ -b (mod a), i.e., d1 ≡ a - (b % a) (mod a) if b%a != 0, else 0.
            
            b_mod_a = b % a
            if b_mod_a == 0:
                residue = 0
            else:
                residue = a - b_mod_a
            
            # Find the smallest d1 >= max(1, ax-b+1) with d1 ≡ residue (mod a)
            d1_min = max(1, a * x - b + 1)
            # d1_min + k*a ≡ residue (mod a) => d1_min ≡ residue (mod a) - k*0... 
            # We need d1_min + k*a ≡ residue (mod a), so k*a ≡ residue - d1_min (mod a)
            # k ≡ (residue - d1_min) * a^{-1} (mod a)... this is getting complicated.
            # Simpler: just iterate d1 from d1_min to b-1 with step a, starting at the right offset.
            
            offset = (residue - d1_min) % a
            if offset == 0:
                d1_start = d1_min
            else:
                d1_start = d1_min + offset
            
            if d1_start >= b:
                continue
            
            for d1 in range(d1_start, b, a):
                if time.time() > deadline:
                    break
                # Check if d1 divides b^2
                b_sq = b * b
                if b_sq % d1 != 0:
                    continue
                d2 = b_sq // d1
                if d2 <= d1:
                    continue
                y = (d1 + b) // a
                z = (d2 + b) // a
                # Verify: y > x, z > y, all distinct, all positive
                if y > x and z > y and y >= 1 and z >= 1:
                    # Verify the equation
                    if x * y * z * 4 == n * (y * z + x * z + x * y):
                        found = [x, y, z]
                        break
            if found:
                break
        
        answers[idx] = found
    
    return answers
