import math
import time

# Precomputed solutions cache
_SOLUTIONS_CACHE = None

def _compute_solutions():
    """
    Precompute solutions for all n in [3, 256].
    Returns a dictionary mapping n to [x, y, z] or None.
    """
    solutions = {}
    for n in range(3, 257):
        found = None
        # x must be such that 1/x < 4/n => x > n/4
        # Also 3/x >= 4/n => x <= 3n/4 if we assume x is the smallest?
        # Not necessarily, but x > n/4 is required for A = 4x-n > 0.
        # If 4x-n <= 0, then 1/y + 1/z <= 0 which is impossible for positive y,z.
        # So x must be > n/4.
        
        # Upper bound for x?
        # If x is very large, A is large, B is large.
        # Usually small x works.
        # Let's iterate x from floor(n/4)+1 upwards.
        # We can stop if x gets too large? 
        # For n=256, n/4 = 64. x starts at 65.
        # Let's try up to some limit, or until found.
        
        start_x = n // 4 + 1
        
        # Heuristic: try x up to n//4 + 100 or so. 
        # If no solution found, we might need to go further, but usually small x works.
        # Actually, for 4/n, solutions are dense.
        
        for x in range(start_x, start_x + 500):
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            # We need (Ay - B)(Az - B) = B^2
            # Let u = Ay - B, v = Az - B. u*v = B^2.
            # u and v are divisors of B^2.
            # y = (u + B) / A, z = (v + B) / A.
            # y, z must be integers => (u+B) % A == 0 and (v+B) % A == 0.
            # Also y != x, z != x, y != z.
            # And y, z <= 10^12.
            
            # Iterate divisors u of B^2
            # To avoid iterating too many, we can just iterate u up to sqrt(B^2) = B
            # But B can be up to 256 * 1000 = 256000. Iterating 256k divisors is fine.
            # Actually, we only need to check u such that (u+B) % A == 0.
            # u = k*A - B.
            # (k*A - B) * (v) = B^2.
            # This substitution is complex. Direct divisor iteration is simpler.
            
            limit = int(math.isqrt(B * B))
            # Optimization: We only need to check u values that satisfy the modulo condition.
            # But checking all divisors is safer.
            # Number of divisors of B^2 is small.
            
            # We can iterate u from 1 to B (since u*v=B^2, if u<=B, v>=B).
            # Actually u can be > B. But if u > B, then v < B.
            # We can just iterate u from 1 to B and check if B^2 % u == 0.
            # Then v = B^2 // u.
            
            for u in range(1, B + 1):
                if (B * B) % u != 0:
                    continue
                v = (B * B) // u
                # Check modulo conditions
                if (u + B) % A != 0:
                    continue
                if (v + B) % A != 0:
                    continue
                
                y = (u + B) // A
                z = (v + B) // A
                
                # Check distinctness and bounds
                if y == x or z == x or y == z:
                    continue
                if y > 10**12 or z > 10**12:
                    continue
                
                # We found a solution
                # Ensure positive integers (implied by construction if A>0, B>0, u,v>0)
                if y > 0 and z > 0:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        solutions[n] = found
    
    return solutions

def construct():
    global _SOLUTIONS_CACHE
    if _SOLUTIONS_CACHE is None:
        # Compute once
        _SOLUTIONS_CACHE = _compute_solutions()
    
    answers = []
    for n in range(3, 257):
        ans = _SOLUTIONS_CACHE.get(n)
        if ans:
            answers.append(ans)
        else:
            answers.append(None)
    return answers
