from fractions import Fraction
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_solution = None
    best_length = 257 # We want to find something shorter than 256
    
    def dfs(rem, min_d, path):
        nonlocal best_solution, best_length
        nonlocal deadline
        
        if time.time() > deadline:
            return False
        
        if rem == 0:
            if len(path) < best_length:
                best_length = len(path)
                best_solution = list(path)
            return True
        
        # Pruning 1: Too many terms
        if len(path) >= best_length:
            return False
        
        # Pruning 2: Lower bound on terms needed
        # Each term <= 1/min_d. Sum of m terms <= m/min_d.
        # We need rem <= m/min_d => m >= rem * min_d.
        min_terms_needed = math.ceil(rem * min_d)
        if len(path) + min_terms_needed >= best_length:
            return False
        
        slots = best_length - len(path)
        if slots <= 0:
            return False
            
        # Pruning 3: Upper bound on d using quadratic bound
        # rem - 1/d <= (slots-1)/(d+1)
        # rem*d^2 + (rem - slots)d - 1 <= 0
        A = rem
        B = rem - slots
        C = -1
        
        # Discriminant
        disc = B*B - 4*A*C
        if disc < 0:
            return False
            
        sqrt_disc = math.sqrt(disc)
        # Roots are (-B +/- sqrt_disc) / (2A)
        # We need d <= larger root
        d_max = int(((-B + sqrt_disc) / (2*A)))
        
        lower_bound = max(min_d, math.ceil(1 / rem))
        
        if lower_bound > d_max:
            return False
            
        # Cap d_max at 10,000,000
        d_max = min(d_max, 10000000)
        
        for d in range(lower_bound, d_max + 1):
            # Check time
            if time.time() > deadline:
                return False
                
            if d < 5:
                continue
                
            # 1/d must be <= rem
            if Fraction(1, d) > rem:
                continue
                
            new_rem = rem - Fraction(1, d)
            if new_rem < 0:
                continue
                
            path.append(d)
            if dfs(new_rem, d + 1, path):
                path.pop()
                return True
            path.pop()
            
        return False
        
    dfs(Fraction(1, 1), 5, [])
    
    if best_solution is None:
        # Fallback greedy
        rem = Fraction(1, 1)
        d = 5
        sol = []
        while rem > 0 and time.time() < deadline:
            next_d = max(d, math.ceil(1 / rem))
            if next_d in sol:
                next_d += 1
            sol.append(next_d)
            rem -= Fraction(1, next_d)
            d = next_d + 1
            if len(sol) > 256:
                break
        if rem == 0 and len(sol) <= 256:
            return sol
        raise RuntimeError("No solution found")

    return best_solution
