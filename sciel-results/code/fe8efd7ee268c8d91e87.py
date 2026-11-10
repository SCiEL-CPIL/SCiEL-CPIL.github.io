import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    LIMIT = 10**12
    answers = [None] * 254
    
    # Precompute: for each n, find a solution
    # Strategy:
    # 1. Try to use algebraic identities to construct solutions quickly
    # 2. For remaining, do bounded search
    
    def find_solution(n, time_budget):
        """Find distinct x,y,z with 1/x+1/y+1/z=4/n, all <= LIMIT"""
        # Try several known constructions first
        
        # Construction 1: If n is odd, use 1/n + 1/n + ... doesn't work (need distinct)
        # But we can use: 4/n = 1/(n) + 1/(n) + 1/(n/2) - no, need distinct
        
        # Known identity: 4/n = 1/(n) + 1/(n) + 1/(n/2) - not distinct
        # Better: 4/n = 1/(2n) + 1/(2n) + 1/(n) - not distinct
        
        # Let's use: 4/n = 1/(n) + 1/(2n) + 1/(2n) - not distinct
        
        # Use: 4/n = 1/(n+1) + 1/(n+1) + ... no
        
        # Standard approach: 4/n = 1/x + 1/y + 1/z
        # If we set x = n, then 1/y + 1/z = 4/n - 1/n = 3/n
        # So yz/(y+z) = n/3, meaning 3yz = n(y+z)
        # 3yz - ny - nz = 0
        # (3y-n)(3z-n) = n^2
        # So we need factor pairs of n^2 where both factors are divisible by 3 appropriately
        
        # Let's try: x = n
        # Then we need 1/y + 1/z = 3/n
        # (3y-n)(3z-n) = n^2
        # Let a = 3y - n, b = 3z - n, then ab = n^2
        # y = (a+n)/3, z = (b+n)/3
        # Need a+n ≡ 0 (mod 3) and b+n ≡ 0 (mod 3)
        # Since ab = n^2, and a ≡ -n (mod 3), b ≡ -n (mod 3)
        # So a ≡ b ≡ -n (mod 3), meaning ab ≡ n^2 (mod 9)
        # n^2 mod 9 is 0, 1, or 4
        # -n mod 3: if n≡0, then a,b≡0; if n≡1, a,b≡2; if n≡2, a,b≡1
        
        # Try this construction
        n2 = n * n
        found_via_identity = False
        
        # Find divisors of n^2
        # For efficiency, iterate up to sqrt(n^2) = n
        a_max = n2
        # We need a such that a divides n^2, a ≡ -n (mod 3), and a+n > 0
        # y = (a+n)/3 must be positive integer, so a+n >= 3 and a+n ≡ 0 (mod 3)
        
        # Iterate over possible a values (divisors of n^2)
        # To keep it fast, try a few divisors
        # Actually, let's just try the search approach but be smarter
        
        # Quick algebraic constructions:
        # 1) If n is divisible by 3: let n=3k
        #    4/(3k) = 1/(3k) + 1/(6k) + 1/(6k) - not distinct
        #    4/(3k) = 1/(3k) + 1/(3k) + 1/(3k) - not distinct
        #    Better: 4/(3k) = 1/(6k) + 1/(6k) + 1/(3k) - not distinct
        #    
        # Let me use a different approach. 
        # For any n, try:
        # x = n, then need 1/y+1/z = 3/n
        # Try y = 2n: 1/z = 3/n - 1/(2n) = (6-1)/(2n) = 5/(2n), z = 2n/5
        # Only works if 5 divides 2n, i.e., 5 divides n
        
        # Let me just do the search but with better bounds and early termination
        
        # x must satisfy: 1/x < 4/n => x > n/4
        # Also x >= 2
        x_min = n // 4 + 1
        if x_min < 2:
            x_min = 2
            
        # For each x, we need 1/y + 1/z = 4/n - 1/x = (4x-n)/(nx)
        # Let A = 4x - n, B = nx
        # Need A > 0, so x > n/4 (already ensured)
        # 1/y + 1/z = A/B
        # yz/(y+z) = B/A
        # A*y*z = B*(y+z)
        # Ayz - By - Bz = 0
        # (Ay-B)(Az-B) = B^2
        # Let u = Ay - B, v = Az - B, then uv = B^2
        # y = (u+B)/A, z = (v+B)/A
        # Need u+B ≡ 0 (mod A), v+B ≡ 0 (mod A)
        # u*v = B^2, u ≡ -B (mod A), v ≡ -B (mod A)
        
        # For efficiency, iterate x and for each x, find valid (u,v) pairs
        # This is equivalent to finding divisors u of B^2 with u ≡ -B (mod A)
        
        # But B^2 can be huge. Let's instead iterate y in a bounded range.
        
        for x in range(x_min, min(x_min + 50000, LIMIT + 1)):
            if time.time() > deadline:
                return None
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            # y must satisfy: 1/y < A/B => y > B/A
            # And 1/y + 1/z = A/B with z > y means 1/y > A/(2B) => y < 2B/A
            y_min = B // A + 1
            y_max = (2 * B) // A
            if y_max < y_min:
                continue
            # Ensure y > x for distinctness (assuming x <= y <= z)
            if y_min <= x:
                y_min = x + 1
            if y_min > y_max:
                continue
            
            # Limit y search range for performance
            # y_max - y_min can be large; cap it
            y_search_max = min(y_max, y_min + 5000)
            
            for y in range(y_min, y_search_max + 1):
                if time.time() > deadline:
                    return None
                # z = B*y / (A*y - B)
                denom = A * y - B
                if denom <= 0:
                    continue
                num = B * y
                if num % denom == 0:
                    z = num // denom
                    if z > y and z <= LIMIT:
                        if x != y and y != z and x != z:
                            return [x, y, z]
        
        # If the bounded search didn't find anything, try a wider but still bounded search
        # Focus on smaller x values where solutions are more likely
        for x in range(x_min, min(x_min + 200, LIMIT + 1)):
            if time.time() > deadline:
                return None
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            y_min = B // A + 1
            y_max = (2 * B) // A
            if y_max < y_min:
                continue
            if y_min <= x:
                y_min = x + 1
            if y_min > y_max:
                continue
            
            # Try all y in range
            y_search_max = min(y_max, y_min + 20000)
            for y in range(y_min, y_search_max + 1):
                if time.time() > deadline:
                    return None
                denom = A * y - B
                if denom <= 0:
                    continue
                num = B * y
                if num % denom == 0:
                    z = num // denom
                    if z > y and z <= LIMIT:
                        if x != y and y != z and x != z:
                            return [x, y, z]
        
        return None
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        sol = find_solution(n, deadline - time.time())
        answers[idx] = sol
    
    return answers
