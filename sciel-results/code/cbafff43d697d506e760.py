"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    results = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        
        # We need 1/x + 1/y + 1/z = 4/n with x < y < z (distinct positive integers)
        # 1/x + 1/y + 1/z = 4/n
        # Let x be the smallest denominator.
        # Since 1/x < 4/n (because there are 3 terms), we need x > n/4.
        # Also 3/x >= 4/n => x <= 3n/4.
        # So x ranges from floor(n/4) + 1 to floor(3n/4).
        
        for x in range(n // 4 + 1, 3 * n // 4 + 1):
            if time.time() > deadline:
                break
            # 1/y + 1/z = 4/n - 1/x = (4x - n) / (nx)
            # Let A = 4x - n, B = nx
            # 1/y + 1/z = A / B
            # (y + z) / (yz) = A / B
            # B(y + z) = A y z
            # A y z - B y - B z = 0
            # Multiply by A:
            # A^2 y z - A B y - A B z = 0
            # (A y - B)(A z - B) = B^2
            # Let u = A y - B, v = A z - B. Then u v = B^2.
            # u < v (since y < z)
            # y = (u + B) / A, z = (v + B) / A
            # We need y > x and z > y, and y, z integers.
            # Also u, v must be positive (since y > B/A and z > B/A, and A>0).
            # Actually, since 1/y < A/B (because 1/z > 0), we have y > B/A.
            # So u = A y - B > 0.
            
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
                
            # We need to find divisors u of B^2 such that:
            # 1. u < B (so that u < v since u*v = B^2 implies v = B^2/u > B)
            #    Actually u < v means u < B.
            # 2. (u + B) % A == 0
            # 3. (v + B) % A == 0 where v = B^2 // u
            # 4. y = (u + B) // A > x
            # 5. z = (v + B) // A > y
            # 6. x, y, z are distinct (guaranteed by x < y < z)
            # 7. x, y, z <= 10^12
            
            # Iterate over divisors u of B^2 up to B.
            # B can be large (n*x up to 256 * 192 ~ 50000). B^2 is ~ 2.5 * 10^9.
            # Iterating up to B (sqrt(B^2)) is feasible if B is small, but B can be up to ~50000.
            # sqrt(B^2) = B. So we iterate u from 1 to B.
            # But B can be ~50000, so 50000 iterations per x.
            # Number of x values is ~ n/2 ~ 128.
            # Total iterations ~ 128 * 50000 = 6.4 * 10^6 per n.
            # For 254 n's, that's ~ 1.6 * 10^9 iterations. Too slow in Python.
            
            # Better approach: iterate over divisors of B^2.
            # Factorize B^2 and generate divisors.
            # Or, iterate u from 1 to B, but only if u divides B^2.
            # Since B^2 = (n*x)^2, its divisors are related to divisors of n*x.
            
            # Alternative: iterate over possible y values.
            # From (A y - B)(A z - B) = B^2, we have A y - B = u, a divisor of B^2.
            # y = (u + B) / A.
            # So we need u such that (u + B) % A == 0 and u | B^2.
            # Also u < B (for y < z) and y > x.
            
            # Let's iterate u from 1 to B-1 (since u < B for u < v).
            # Check if B^2 % u == 0.
            # If so, v = B^2 // u.
            # Check if (u + B) % A == 0 and (v + B) % A == 0.
            # If so, y = (u + B) // A, z = (v + B) // A.
            # Check if y > x and z > y and z <= 10^12.
            
            # To speed up, we can iterate u over divisors of B^2.
            # Let's find all divisors of B^2.
            # B = n * x.
            
            # Actually, iterating u from 1 to B might be too slow if B is large.
            # Let's estimate B max: n=256, x max ~ 192. B = 256*192 = 49152.
            # So u goes up to 49152. That's fine.
            # For each n, we have ~128 x values.
            # Total u iterations per n: ~ 128 * 49152 / 2 (average) ~ 3 * 10^6.
            # For 254 n's: ~ 7.6 * 10^8. Still too slow in pure Python.
            
            # We need a faster method.
            # Note that u must divide B^2.
            # Let's iterate over divisors of B^2 directly.
            
            def get_divisors(m):
                divs = []
                i = 1
                while i * i <= m:
                    if m % i == 0:
                        divs.append(i)
                        if i != m // i:
                            divs.append(m // i)
                    i += 1
                return divs
            
            # But get_divisors(B^2) requires factoring B^2, which is expensive if done naively.
            # However, B^2 is at most ~ (50000)^2 = 2.5 * 10^9.
            # sqrt(2.5 * 10^9) ~ 50000. So get_divisors takes ~50000 steps.
            # Same as iterating u from 1 to B.
            
            # Let's try a different approach: iterate y.
            # 1/y + 1/z = A/B => z = B*y / (A*y - B)
            # We need A*y - B > 0 => y > B/A.
            # Also z > y => B*y / (A*y - B) > y => B > A*y - B => A*y < 2B => y < 2B/A.
            # So y ranges from floor(B/A) + 1 to floor(2B/A).
            # And A*y - B must divide B*y.
            # Let d = A*y - B. Then d | B*y.
            # Since d = A*y - B, we have A*y = d + B.
            # So d | B*(d+B)/A => A*d | B*(d+B) => A*d | B*d + B^2 => A*d | B^2.
            # So d must be a divisor of B^2, and d = A*y - B.
            # This is the same as before.
            
            # Let's just iterate u from 1 to B-1 and check B^2 % u == 0.
            # To make it faster, we can precompute divisors of B^2 for each (n,x) pair? No.
            
            # Let's try iterating y directly.
            # y from max(x+1, B//A + 1) to 2*B//A.
            # For each y, check if (A*y - B) divides B*y.
            
            y_min = max(x + 1, B // A + 1)
            y_max = 2 * B // A
            
            if y_max < y_min:
                continue
                
            # To avoid too many iterations, we can break early if found.
            for y in range(y_min, y_max + 1):
                if time.time() > deadline:
                    break
                d = A * y - B
                if d <= 0:
                    continue
                if (B * y) % d == 0:
                    z = (B * y) // d
                    if z > y and z <= 1000000000000:
                        found = [x, y, z]
                        break
            if found:
                break
        
        results[idx] = found
    
    return results
# EVOLVE-BLOCK-END
