"""Combined program for Egyptian fraction of 5/31; minimizes largest denominator."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    target = Fraction(5, 31)
    
    best_max = float('inf')
    best_solution = None
    
    def complete_2_terms(r, min_den, max_den):
        """Find b, c such that 1/b + 1/c = r, min_den <= b < c <= max_den."""
        if r <= 0:
            return None
        n = r.numerator
        d = r.denominator
        # (n*b - d)(n*c - d) = d^2
        # X = n*b - d, Y = n*c - d
        # X*Y = d^2
        # X < Y
        # b = (X+d)/n, c = (Y+d)/n
        # b >= min_den, c <= max_den
        # X < d (since b > d/n, roughly)
        # Actually X < sqrt(d^2) = d is not strictly required, but X < Y implies X < d if Y > d.
        # Since b < c, X < Y.
        
        # Iterate divisors of d^2
        # To be fast, we only check divisors up to d
        # But d can be large.
        # However, r is a fraction from a few steps, so d is not huge.
        
        # Optimization: If d is large, factoring might be slow.
        # But for this problem, d is small.
        
        # We need X such that X | d^2 and X < Y (so X < d)
        # and (X+d) % n == 0
        # and (d^2/X + d) % n == 0
        # and b >= min_den
        # and c <= max_den
        
        # We also need b < c, which is satisfied by X < Y.
        
        # Find divisors of d^2
        # Simple loop up to sqrt(d^2) = d
        # This is O(d). If d is 10^5, this is slow.
        # But d is denominator of remainder.
        
        # Better: Factor d, generate divisors of d^2.
        # For simplicity and given small constraints, we can try a limited range or use a fast factorization.
        # Given the time limit, and that we are searching for small max_den, d will be small.
        
        # Let's just loop X from 1 to d
        # But d might be 10^4. 10^4 iterations is fine.
        
        # Actually, we can bound X.
        # b = (X+d)/n >= min_den => X >= n*min_den - d
        # c = (d^2/X + d)/n <= max_den => d^2/X <= n*max_den - d => X >= d^2 / (n*max_den - d)
        
        x_min = max(1, n * min_den - d)
        # x_max is determined by X < Y => X < d
        # Also c <= max_den => X >= d^2 / (n*max_den - d)
        
        # We iterate X in [x_min, d-1]
        # And check if X divides d^2
        
        # To avoid O(d) loop if d is large, we can factor d.
        # But let's try the simple loop first, as d is likely small in the promising branches.
        
        # If d > 10000, we might need a faster method.
        # For now, assume d is small.
        
        if d > 5000:
             # Fallback to slower method or skip?
             # If d is large, min_den is likely large too, so range is small?
             # No, x_min could be small.
             # Let's use a divisor generation method if d is large.
             pass
             
        # Simple divisor check
        # We need X to be a divisor of d^2.
        # Instead of checking all X, we can find divisors of d^2.
        
        # Let's implement a simple divisor finder for d^2
        # This is O(sqrt(d^2)) = O(d) anyway.
        
        # Let's just loop X
        for x in range(x_min, d):
            if (d * d) % x != 0:
                continue
            y = (d * d) // x
            if x >= y:
                continue
            if (x + d) % n != 0:
                continue
            if (y + d) % n != 0:
                continue
            b = (x + d) // n
            c = (y + d) // n
            if b >= min_den and c <= max_den and b < c:
                return (b, c)
        
        return None

    def dfs(r, min_den, max_den, current, depth, max_depth=256):
        """Bounded DFS with 2-term completion."""
        nonlocal best_max, best_solution
        if time.time() > deadline:
            return False
        if r == 0:
            if current:
                m = max(current)
                if m < best_max:
                    best_max = m
                    best_solution = list(current)
            return True
        if depth >= max_depth or r <= 0:
            return False
        
        # If we have 1 term left, we can't use complete_2_terms (needs 2).
        # But we are looking for distinct denominators.
        # If depth == max_depth - 1, we need 1 term.
        # 1/d = r => d = 1/r.
        if max_depth - depth == 1:
            if r.denominator % r.numerator == 0:
                d = r.denominator // r.numerator
                if d >= min_den and d <= max_den:
                    current.append(d)
                    m = max(current)
                    if m < best_max:
                        best_max = m
                        best_solution = list(current)
                    current.pop()
            return False

        # If we have 2 terms left, use complete_2_terms
        if max_depth - depth == 2:
            pair = complete_2_terms(r, min_den, max_den)
            if pair:
                b, c = pair
                current.append(b)
                current.append(c)
                m = max(current)
                if m < best_max:
                    best_max = m
                    best_solution = list(current)
                current.pop()
                current.pop()
            return False

        d_start = max(min_den, (r.denominator + r.numerator - 1) // r.numerator)
        if d_start > max_den:
            return False
        
        # Pruning
        remaining_terms = max_depth - depth
        # Max sum of remaining terms
        # r <= remaining_terms / (d_start)
        if r > Fraction(remaining_terms, d_start):
            return False
            
        # Min sum of remaining terms
        # r >= remaining_terms / (max_den) ? No, denominators can be larger.
        # But we are bounded by max_den.
        # So r >= remaining_terms / max_den is NOT a valid lower bound for feasibility,
        # it's a lower bound for the sum if we use max_den for all.
        # Actually, if r < remaining_terms / max_den, it's impossible?
        # No, we can use smaller denominators (larger values).
        # The minimum value of 1/d for d <= max_den is 1/max_den.
        # So the minimum sum of k terms is k / max_den.
        # If r < k / max_den, we can't reach r?
        # Wait, 1/d >= 1/max_den.
        # So sum >= k / max_den.
        # If r < k / max_den, then it's impossible to represent r with k terms <= max_den.
        if r < Fraction(remaining_terms, max_den):
            return False

        for d in range(d_start, max_den + 1):
            if time.time() > deadline:
                return False
            new_r = r - Fraction(1, d)
            if new_r < 0:
                continue
            current.append(d)
            if dfs(new_r, d + 1, max_den, current, depth + 1, max_depth):
                current.pop()
                return True
            current.pop()
        
        return False
    
    # Initial best known solution: 8, 31, 248 (Max 248)
    best_solution = [8, 31, 248]
    best_max = 248
    
    # Try to find better
    # Search for solutions with max_den < 248
    # We will search for solutions with up to 256 terms.
    # But we can prune heavily.
    
    # We'll run DFS with max_den = best_max - 1
    # To allow finding any improvement.
    
    # We can also try to increase the depth limit if needed, but 256 is plenty.
    
    # Run DFS
    dfs(target, 1, best_max - 1, [], 0, 256)
    
    if best_solution is not None and len(best_solution) <= 256:
        s = sum(Fraction(1, d) for d in best_solution)
        if s == target and len(set(best_solution)) == len(best_solution):
            return best_solution
    
    # Fallback
    return [8, 31, 248]
# EVOLVE-BLOCK-END
