import math
import time

def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    answers = []
    
    for n in range(3, 257):
        found = None
        
        # 1/x + 1/y + 1/z = 4/n
        # Assume x <= y <= z
        # Then 1/x >= 1/3 * (4/n) => x <= 3n/4
        # Also 1/x < 4/n => x > n/4
        # So x is in (n/4, 3n/4]
        
        # For a fixed x:
        # 1/y + 1/z = 4/n - 1/x = (4x - n) / (nx)
        # Let A = 4x - n, B = nx
        # 1/y + 1/z = A/B
        # B(y+z) = Az yz
        # Az yz - B y - B z = 0
        # (Az - B)y - B z = 0
        # (Az - B)y = B z
        # z (Az - B) = By
        # z = By / (Ay - B)
        
        # Alternative approach using divisors:
        # (Ay-B)(Az-B) = B^2
        # Let U = Ay - B, V = Az - B
        # UV = B^2, U < V
        # y = (U+B)/A, z = (V+B)/A
        # Need y, z integers, so U ≡ -B (mod A) and V ≡ -B (mod A)
        
        start_x = n // 4 + 1
        end_x = (3 * n) // 4
        
        for x in range(start_x, end_x + 1):
            if time.time() > deadline:
                # Fallback: return what we have so far, pad with None
                while len(answers) < 254:
                    answers.append(None)
                return answers
            
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # Find divisors of B^2
            # U must be a divisor of B^2 such that:
            # 1. U < B (since y > B/A implies U = Ay - B > 0, and y < 2B/A implies U < B)
            # 2. U ≡ -B (mod A)
            # 3. V = B^2/U ≡ -B (mod A)
            # 4. y = (U+B)/A > x (for distinctness and ordering)
            # 5. z = (V+B)/A > y
            
            B2 = B * B
            
            # Find divisors of B^2
            # We can iterate up to sqrt(B^2) = B
            # But B can be up to ~50000, so this might be slow
            # Alternative: factorize B and generate divisors of B^2
            
            # For now, let's try iterating U from 1 to B-1
            # But this is O(B) per x, which might be slow
            
            # Better: iterate over divisors of B^2
            # Factorize B first
            temp = B
            factors = []
            d = 2
            while d * d <= temp:
                while temp % d == 0:
                    factors.append(d)
                    temp //= d
                d += 1
            if temp > 1:
                factors.append(temp)
            
            # Generate all divisors of B^2
            # B^2 has prime factorization with exponents doubled
            # This can get complex, so let's try a different approach
            
            # Iterate y in range as before, but with early termination
            y_min = B // A + 1
            y_max = (2 * B) // A
            
            for y in range(y_min, y_max + 1):
                D = A * y - B
                if D <= 0:
                    continue
                if (B * y) % D == 0:
                    z = (B * y) // D
                    if z > y:
                        # Check distinctness: x, y, z distinct
                        if x != y and y != z and x != z:
                            found = [x, y, z]
                            break
            if found:
                break
        
        answers.append(found)
        
        if time.time() > deadline:
            while len(answers) < 254:
                answers.append(None)
            return answers

    return answers
