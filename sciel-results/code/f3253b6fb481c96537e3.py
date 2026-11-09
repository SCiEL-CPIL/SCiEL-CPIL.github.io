import time
from math import gcd

def construct():
    answers = [None] * 254  # index n-3
    deadline = time.time() + 55
    
    for n in range(3, 257):
        if time.time() > deadline:
            break
        
        found = None
        
        # Try known simple solutions first
        # For n divisible by 4: 4/n = 1/(n/4)
        # Then 1/x + 1/y + 1/z = 1/(n/4)
        # Use x = n/4, then 1/y + 1/z = 1/(n/4) - 1/(n/4) = 0 -- that doesn't work.
        # Instead: 4/n with n=4k: 4/(4k) = 1/k
        # 1/k = 1/(2k) + 1/(2k) + ... no, need distinct.
        # 1/k = 1/(k+1) + 1/(k(k+1)) -- two terms.
        # For three distinct: 1/k = 1/(2k) + 1/(3k) + 1/(6k) -- but 2k, 3k, 6k may not be distinct from each other? They are distinct.
        # Wait: 1/(2k) + 1/(3k) + 1/(6k) = 3/(6k) + 2/(6k) + 1/(6k) = 6/(6k) = 1/k. Yes!
        # So for n = 4k: [2k, 3k, 6k] = [n/2, 3n/4, 3n/2]
        
        if n % 4 == 0:
            k = n // 4
            x, y, z = 2*k, 3*k, 6*k
            if x != y and y != z and x != z and 1 <= x <= 10**12 and 1 <= y <= 10**12 and 1 <= z <= 10**12:
                # Verify
                if x*y*z*4 == n*(y*z + x*z + x*y):
                    found = [x, y, z]
        
        if found is None:
            # Try x = n/2 when n is even: 4/n - 2/n = 2/n
            # 1/y + 1/z = 2/n
            # (2y-n)(2z-n) = n^2
            if n % 2 == 0:
                x = n // 2
                # 1/y + 1/z = 2/n
                # Let's use the divisor method
                # a = 2, b = n (from 1/y+1/z = a/b where a=2, b=n after simplifying? No)
                # 4/n - 1/x = 4/n - 2/n = 2/n
                # So 1/y + 1/z = 2/n
                # a=2, b=n: (2y-n)(2z-n) = n^2
                # Wait, general form: 1/y + 1/z = a/b => (ay-b)(az-b) = b^2
                # Here a=2, b=n
                # (2y-n)(2z-n) = n^2
                # Let u = 2y-n, v = 2z-n, uv = n^2, u+v = 2(y+z) -- no
                # Actually u*v = n^2, and y = (u+n)/2, z = (v+n)/2
                # Need y > x = n/2, so (u+n)/2 > n/2 => u > 0
                # Need z > y, so v > u
                # Also need u, v same parity as n (so that y, z are integers)
                
                # Iterate over divisors d of n^2 where d < n^2/d (i.e., d < n)
                # Actually u*v = n^2, u < v, u > 0
                # u must have same parity as n for y to be integer
                n2 = n * n
                found2 = False
                # Get divisors of n^2
                # For efficiency, iterate d from 1 to sqrt(n^2) = n
                # But n can be up to 256, so n^2 up to 65536, sqrt up to 256
                # That's fast
                for d in range(1, int(n**0.5) + 1):
                    if time.time() > deadline:
                        break
                    if n2 % d == 0:
                        u = d
                        v = n2 // d
                        if u >= v:
                            continue
                        # Check parity: (u+n) must be even, (v+n) must be even
                        if (u + n) % 2 != 0 or (v + n) % 2 != 0:
                            continue
                        y = (u + n) // 2
                        z = (v + n) // 2
                        if y > x and z > y and y <= 10**12 and z <= 10**12:
                            # Verify
                            if x*y*z*4 == n*(y*z + x*z + x*y):
                                found = [x, y, z]
                                found2 = True
                                break
                if found2:
                    pass
        
        if found is None:
            # General search using the divisor method
            # 4/n - 1/x = (4x-n)/(nx)
            # 1/y + 1/z = (4x-n)/(nx)
            # a = 4x-n, b = nx
            # (a*y - b)(a*z - b) = b^2
            # Let p = a*y - b, q = a*z - b, p*q = b^2
            # y = (p+b)/a, z = (q+b)/a
            # Need y > x, z > y, all distinct, all positive, <= 10^12
            # p > 0 (since y > b/a = nx/(4x-n), and we need y > x)
            # Actually y > x means (p+b)/a > x, so p > ax - b = a*x - b
            # ax - b = (4x-n)*x - nx = 4x^2 - nx - nx = 4x^2 - 2nx = 2x(2x-n)
            # Since x > n/4, 2x > n/2, but we need 2x > n for this to be positive
            # If x > n/2, then p > 2x(2x-n) > 0
            # If n/4 < x <= n/2, then 2x-n <= 0, so p > negative number, which is always true for p > 0
            
            # Bounds on x: x > n/4 (so 4x-n > 0), and for z to exist, we need the range to be valid
            # x < 3n/4 (from 3/x > 4/n)
            
            x_min = n // 4 + 1
            x_max = min(3 * n // 4, 2000)  # Cap x search for time
            
            for x in range(x_min, x_max + 1):
                if time.time() > deadline:
                    break
                    
                a = 4 * x - n
                if a <= 0:
                    continue
                b = n * x
                b2 = b * b
                
                # Need (p+b) divisible by a, (q+b) divisible by a
                # p*q = b^2
                # y = (p+b)//a > x => p > a*x - b
                # p_min = max(1, a*x - b + 1)
                # Also y > 0 always if p > -b, which is true for p > 0
                # z > y => q > p (since q = b^2/p and if p < sqrt(b^2)=b, q > b > p... not necessarily)
                # Actually if p < b, q = b^2/p > b > p, so z > y automatically
                # If p >= b, q <= b <= p, so z <= y, not allowed
                # So we need p < b for z > y
                # And p < b^2/p => p^2 < b^2 => p < b. Consistent.
                
                # So: iterate p from p_min to min(b-1, some cap), where p | b^2
                # And (p+b) % a == 0
                # And z = (b^2//p + b) // a <= 10^12
                
                # For efficiency, instead of iterating all p, iterate over divisors of b^2
                # b = n*x, b^2 = n^2*x^2
                # Number of divisors of b^2 can be large, but for n<=256, x<=2000, b<=512000, b^2 up to ~2.6e11
                # sqrt(b^2) = b, up to 512000. That's too many to iterate directly.
                
                # Alternative: iterate p in a smaller range
                # p_min = max(1, a*x - b + 1)
                # p must be < b
                # Also (p+b) % a == 0 => p ≡ -b (mod a) => p ≡ b (mod a) since -b ≡ b only if a|2b... no
                # -b mod a = (a - b%a) % a. Let r = (-b) % a. Then p ≡ r (mod a).
                # p = r + k*a for k >= 0
                # p_min <= r + k*a < b
                
                # The range of k: k_min = ceil((p_min - r)/a) if r < p_min else 0
                # k_max = floor((b-1-r)/a)
                
                # For each such p, check if p | b^2
                # This is still potentially many values.
                
                # Better approach for small n: just iterate p and check divisibility
                # But limit the range. For small n, b is small.
                
                p_min = max(1, a * x - b + 1)
                if p_min >= b:
                    continue
                    
                r = (-b) % a  # p ≡ r (mod a)
                if r == 0:
                    r = a  # p = a, 2a, 3a, ... but p must be > 0
                    # Actually if r=0, p is multiple of a: p = a, 2a, ...
                    # But p ≡ 0 (mod a), so p = k*a
                    # Let me handle this case
                    k_min = (p_min + a - 1) // a
                    if k_min < 1:
                        k_min = 1
                    k_max = (b - 1) // a
                    if k_min > k_max:
                        continue
                    for k in range(k_min, k_max + 1):
                        if time.time() > deadline:
                            break
                        p = k * a
                        if p >= b:
                            break
                        if b2 % p == 0:
                            q = b2 // p
                            if q <= p:
                                continue
                            y = (p + b) // a
                            z = (q + b) // a
                            if y > x and z > y and x != y and y != z and x != z:
                                if y <= 10**12 and z <= 10**12:
                                    if x*y*z*4 == n*(y*z + x*z + x*y):
                                        found = [x, y, z]
                                        break
                    if found:
                        break
                else:
                    # p = r + k*a
                    k_min = (p_min - r + a - 1) // a
                    if k_min < 0:
                        k_min = 0
                    k_max = (b - 1 - r) // a
                    if k_max < k_min:
                        continue
                    for k in range(k_min, k_max + 1):
                        if time.time() > deadline:
                            break
                        p = r + k * a
                        if p >= b:
                            break
                        if b2 % p == 0:
                            q = b2 // p
                            if q <= p:
                                continue
                            y = (p + b) // a
                            z = (q + b) // a
                            if y > x and z > y and x != y and y != z and x != z:
                                if y <= 10**12 and z <= 10**12:
                                    if x*y*z*4 == n*(y*z + x*z + x*y):
                                        found = [x, y, z]
                                        break
                    if found:
                        break
        
        answers[n-3] = found
    
    return answers
