import time
import math
from fractions import gcd

def construct():
    answers = []
    deadline = time.time() + 55  # Safety margin before 60s limit
    N_MAX = 10**12
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # We need 1/x + 1/y + 1/z = 4/n with x < y < z (distinct positive integers)
        # Since 1/x + 1/y + 1/z = 4/n and x < y < z, we have 3/x > 4/n => x < 3n/4
        # Also 1/x < 4/n => x > n/4
        # So x is in (n/4, 3n/4)
        
        # For a given x, we need 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
        # Let a = 4x - n, b = nx. We need 1/y + 1/z = a/b => b(y+z) = ayz
        # => ayz - by - bz = 0
        # => (ay - b)(az - b) = b^2
        # Let u = ay - b, v = az - b. Then uv = b^2.
        # y = (u + b)/a, z = (v + b)/a
        # We need y > x, z > y, and y, z integers.
        # So u + b must be divisible by a, and v + b must be divisible by a.
        # Also u < v (since y < z => u < v).
        # And u > 0, v > 0 (since y, z > 0 and a > 0 for x > n/4).
        # Also y > x => (u+b)/a > x => u + b > ax => u > ax - b = ax - nx = x(4 - n)... wait
        # Let me re-derive. a = 4x - n. For x > n/4, a > 0.
        # y > x => (u+b)/a > x => u + b > ax => u > ax - b
        # b = nx, so u > ax - nx = x(a - n) = x(4x - n - n) = x(4x - 2n) = 2x(2x - n)
        # Hmm, let's just check y > x directly.
        
        # For each x in range:
        start_x = n // 4 + 1
        # x < 3n/4, so x <= (3n-1)//4
        end_x = (3 * n - 1) // 4
        
        for x in range(start_x, end_x + 1):
            if time.time() > deadline:
                break
            a = 4 * x - n
            b = n * x
            if a <= 0:
                continue
            
            # Find factors u, v of b^2 with u < v, u*v = b^2
            # y = (u+b)/a, z = (v+b)/a
            # Need y, z integers, y > x, z > y, all <= N_MAX
            
            # b can be large, so we can't enumerate all factors of b^2 naively
            # But we can iterate over possible u values.
            # u must be a divisor of b^2, u < b (since u < v and uv = b^2 => u < b)
            # Actually u < v means u^2 < b^2 => u < b.
            
            # Let's iterate u from 1 to b-1? That's too slow for large b.
            # Alternative: iterate over divisors of b, and for each divisor d of b,
            # u = d * k where... this is getting complicated.
            
            # Better approach: for a fixed x, iterate y from x+1 upward.
            # 1/y + 1/z = a/b => 1/z = a/b - 1/y = (ay - b)/(by)
            # z = by / (ay - b)
            # Need ay - b > 0 => y > b/a = nx/(4x-n)
            # And ay - b divides by.
            
            # y must satisfy: y > x, y > b/a, and (ay - b) | (by)
            # Let d = ay - b. Then z = by/d. We need d | by.
            # d = ay - b => y = (d + b)/a. So d + b must be divisible by a.
            # Also d | by = b(d+b)/a => a*d | b*(d+b) => a*d | b*d + b^2 => a*d | b^2
            # (since a*d | b*d is automatic if a*d | b^2... wait, not quite)
            # a*d | b*(d+b). Let g = gcd(a, b). Then a/g * d | b/g * (d+b).
            # Since gcd(a/g, b/g) = 1, we need a/g | d+b and d | b/g * (d+b)... 
            # This is getting complex. Let me just iterate y.
            
            # y > max(x, ceil(b/a))
            min_y = max(x + 1, b // a + 1)
            # z = by / (ay - b). For z > y, we need by / (ay - b) > y => b > ay - b => 2b > ay => y < 2b/a
            # So y < 2b/a
            max_y = 2 * b // a
            
            if min_y >= max_y:
                continue
            
            # Iterate y from min_y to max_y - 1
            for y in range(min_y, max_y):
                denom = a * y - b
                if denom <= 0:
                    continue
                if (b * y) % denom == 0:
                    z = (b * y) // denom
                    if z > y and z <= N_MAX:
                        found = [x, y, z]
                        break
            if found:
                break
        
        answers.append(found)
    
    return answers
