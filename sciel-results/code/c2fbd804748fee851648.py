import math
import time

def _divisors_of(n):
    """Return all positive divisors of n."""
    if n <= 0:
        return []
    small = []
    large = []
    i = 1
    limit = int(math.isqrt(n))
    while i <= limit:
        if n % i == 0:
            small.append(i)
            if i != n // i:
                large.append(n // i)
        i += 1
    large.reverse()
    return small + large

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    answers = [None] * 254  # for n=3..256
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        # x must satisfy: 4x - n > 0 and 3/x >= 4/n => x <= 3n/4
        # Also x >= ceil(n/4) for a > 0
        x_min = n // 4 + 1  # smallest x with 4x > n
        x_max = (3 * n) // 4  # largest x where 3/x >= 4/n
        
        if x_max < x_min:
            x_max = x_min  # at least try one
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            a = 4 * x - n
            b = n * x
            # (ay - b)(az - b) = b^2
            b2 = b * b
            # Find divisors d of b2 such that (d + b) % a == 0 and (b2//d + b) % a == 0
            # and resulting y, z are positive, distinct from x and each other
            # d must be positive divisor
            # y = (d + b) / a, z = (b2//d + b) / a
            # Need d + b > 0 (always true for d > 0, b > 0)
            # Need a > 0 (guaranteed by x_min)
            
            # We can iterate over divisors of b2
            # But b2 can be large. Let's be smart: we only need d such that d ≡ -b (mod a)
            # and b2/d ≡ -b (mod a)
            
            # Alternative: iterate d from 1 to sqrt(b2), check divisibility
            # sqrt(b2) = b = nx, which for n=256, x~128 is ~32768. That's fine.
            
            limit_d = int(math.isqrt(b2))
            d = 1
            while d <= limit_d:
                if b2 % d == 0:
                    d1 = d
                    d2 = b2 // d
                    # Check d1 + b divisible by a
                    if (d1 + b) % a == 0:
                        y = (d1 + b) // a
                        if (d2 + b) % a == 0:
                            z = (d2 + b) // a
                            # Check distinctness and positivity
                            if y > 0 and z > 0 and y != z and y != x and z != x:
                                # Also check all <= 10^12
                                if x <= 10**12 and y <= 10**12 and z <= 10**12:
                                    found = [x, y, z]
                                    break
                d += 1
            if found is not None:
                break
        
        answers[idx] = found
    
    return answers
