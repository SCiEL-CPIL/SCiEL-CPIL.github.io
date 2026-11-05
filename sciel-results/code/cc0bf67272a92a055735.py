import time
import math
from math import gcd

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline - 1:
            answers.append(None)
            continue
        
        found = None
        # x must satisfy 4x - n > 0, i.e., x > n/4
        start_x = n // 4 + 1
        
        # Try a range of x values; usually small x works
        # But for some n we may need to go further
        max_x_search = start_x + 200
        
        for x in range(start_x, max_x_search):
            if time.time() > deadline - 0.5:
                break
            
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            M = B * B  # B squared
            
            # We need (Ay - B)(Az - B) = B^2
            # Let d = Ay - B, then Az - B = B^2/d
            # y = (d + B) / A, z = (B^2/d + B) / A
            # Need d | B^2, d > B (so y > 0), and (d+B) % A == 0, (B^2/d + B) % A == 0
            # Also need y > x, z > x, y != z, all <= 10^12
            
            # Find divisors of M = B^2
            # Factor B first, then B^2 has exponents doubled
            B_val = B
            # Get prime factorization of B
            temp = B_val
            factors = {}
            d2 = 2
            while d2 * d2 <= temp:
                while temp % d2 == 0:
                    factors[d2] = factors.get(d2, 0) + 1
                    temp //= d2
                d2 += 1
            if temp > 1:
                factors[temp] = factors.get(temp, 0) + 1
            
            # B^2 has each exponent doubled
            exps = {}
            for p, e in factors.items():
                exps[p] = e * 2
            
            # Generate all divisors of B^2
            # This can be large, so let's be smart:
            # We only need divisors d where d > B and d < 2*B (approximately, for y to be reasonable)
            # Actually y = (d+B)/A, and we want y > x = B/n (since B = nx)
            # y > x means (d+B)/A > B/n, i.e., n(d+B) > AB = A*n*x = (4x-n)*n*x
            # This is getting complex. Let's just enumerate divisors and check.
            
            # For efficiency, let's try a different approach:
            # Iterate over possible y values in a smart range
            # y must be > x, and 1/y + 1/z = A/B
            # z = y*B / (A*y - B), need A*y - B > 0, i.e., y > B/A
            # Also z > y means y*B/(Ay-B) > y, i.e., B > Ay - B, i.e., Ay < 2B, i.e., y < 2B/A
            
            # So y is in range (B/A, 2B/A) and y > x
            y_min = max(x + 1, B // A + 1)
            y_max = min(2 * B // A, 10**12)
            
            # If range is too large, use divisor approach
            if y_max - y_min > 100000:
                # Use divisor enumeration
                # Generate divisors of M
                primes = list(exps.keys())
                divisors = [1]
                for p in primes:
                    e = exps[p]
                    new_divs = []
                    for d in divisors:
                        for k in range(e + 1):
                            new_divs.append(d * (p ** k))
                    divisors = new_divs
                
                # Filter: d > B, (d+B) % A == 0
                for d in divisors:
                    if d <= B:
                        continue
                    if d > M:  # shouldn't happen
                        continue
                    rem = M - d * (M // d)  # check if d divides M
                    if rem != 0:
                        continue
                    other = M // d
                    if other <= B:
                        continue
                    # y = (d + B) / A, z = (other + B) / A
                    if (d + B) % A != 0:
                        continue
                    if (other + B) % A != 0:
                        continue
                    y = (d + B) // A
                    z = (other + B) // A
                    if y <= x or z <= x:
                        continue
                    if y == z:
                        continue
                    if y > 10**12 or z > 10**12:
                        continue
                    # Ensure distinctness of all three
                    if x == y or x == z or y == z:
                        continue
                    found = [x, y, z]
                    break
            else:
                # Linear scan over y
                for y in range(y_min, y_max + 1):
                    denom = A * y - B
                    if denom <= 0:
                        continue
                    if (B * y) % denom != 0:
                        continue
                    z = B * y // denom
                    if z <= y:
                        continue
                    if z > 10**12:
                        continue
                    if x == y or x == z or y == z:
                        continue
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
