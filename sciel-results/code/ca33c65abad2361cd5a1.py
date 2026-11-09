"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for the 90s wall-clock limit
    
    one = Fraction(1)
    
    # Pre-computed valid solution: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12
    # Let's verify this sums to 1
    # Actually, let me compute a guaranteed valid solution first using greedy
    # Then try to find a smaller one
    
    def greedy_decompose(remaining, min_d, max_terms=256):
        """Greedy Egyptian fraction decomposition with minimum denominator constraint."""
        result = []
        for _ in range(max_terms):
            if remaining == 0:
                break
            if remaining < 0:
                return None
            # Find smallest d >= min_d such that 1/d <= remaining
            if remaining.numerator > 0:
                d = max(min_d, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
            else:
                d = min_d
            
            # Ensure 1/d <= remaining
            while Fraction(1, d) > remaining:
                d += 1
            
            if d > 10000000:
                return None
            
            result.append(d)
            remaining -= Fraction(1, d)
            min_d = d + 1
        
        if remaining == 0:
            return result
        return None
    
    # Get a valid solution first using greedy (this is our fallback)
    greedy_result = greedy_decompose(one, 5)
    
    # Now try to find a smaller solution using iterative deepening DFS
    # We know from analysis that:
    # - 3 terms: max sum = 1/5+1/6+1/7 ≈ 0.509 < 1 → impossible
    # - 4 terms: max sum ≈ 0.634 < 1 → impossible  
    # - 5 terms: max sum ≈ 0.745 < 1 → impossible
    # - 6 terms: max sum ≈ 0.845 < 1 → impossible
    # - 7 terms: max sum ≈ 0.936 < 1 → impossible
    # - 8 terms: max sum ≈ 1.019 > 1 → possible!
    
    # So we start searching from depth 8
    def search(depth, max_depth, remaining, min_den, current):
        """Try to complete with exactly max_depth terms total (depth so far)."""
        if time.time() > deadline:
            return None
        
        if depth == max_depth:
            if remaining == 0:
                return list(current)
            return None
        
        terms_left = max_depth - depth
        
        if remaining <= 0:
            if remaining == 0 and terms_left == 0:
                return list(current)
            return None
        
        # Max achievable sum with terms_left terms, denominators starting from min_den
        max_possible = sum(Fraction(1, min_den + i) for i in range(terms_left))
        if remaining > max_possible:
            return None
        
        # If we have only 1 term left, check if it's a unit fraction
        if terms_left == 1:
            if remaining > 0:
                d_frac = one / remaining
                if d_frac.denominator == 1:
                    d_int = d_frac.numerator
                    if d_int >= min_den and d_int >= 5 and d_int <= 10000000:
                        current.append(d_int)
                        result = list(current)
                        current.pop()
                        return result
            return None
        
        # Lower bound on next denominator
        d_lower = max(min_den, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        
        # Upper bound: we need remaining - 1/d to be achievable with terms_left-1 terms
        # from d+1. The max sum with terms_left-1 terms from d+1 is:
        # sum(1/(d+1+i) for i in range(terms_left-1))
        # So we need: remaining - 1/d <= max_rest
        # => 1/d >= remaining - max_rest
        # => d <= 1/(remaining - max_rest) if remaining > max_rest
        
        # For the upper bound on d, we need new_remaining = remaining - 1/d >= 0
        # and new_remaining must be <= max_rest
        # So: remaining - 1/d <= max_rest
        # => 1/d >= remaining - max_rest
        # If remaining > max_rest, then d <= 1/(remaining - max_rest)
        
        d_upper = 10000000
        if remaining > max_possible:  # This shouldn't happen due to earlier check
            return None
        
        # Compute a tighter upper bound
        # We need: remaining - 1/d <= sum(1/(d+1+i) for i in range(terms_left-1))
        # For large d, this is approximately remaining - 1/d <= (terms_left-1)/(d+1)
        # So: remaining <= 1/d + (terms_left-1)/(d+1) ≈ terms_left/d
        # Thus d <= terms_left/remaining approximately
        
        # More precisely, let's bound d by trying values
        # d must satisfy: remaining - 1/d <= H(d+terms_left-2) - H(d) where H is harmonic
        # For simplicity, use: d <= floor(terms_left / remaining) as a rough upper bound
        # But this might be too loose. Let's use a better approach.
        
        # Actually, the key constraint is that new_remaining must be positive and 
        # achievable. Let's just iterate and break when new_remaining > max_rest.
        
        for d in range(d_lower, min(d_upper + 1, 10000001)):
            val = Fraction(1, d)
            if val > remaining:
                break
            
            new_remaining = remaining - val
            
            if terms_left - 1 == 0:
                if new_remaining == 0:
                    current.append(d)
                    result = list(current)
                    current.pop()
                    return result
                continue
            
            # Check if new_remaining can be achieved with terms_left-1 terms starting from d+1
            max_rest = sum(Fraction(1, d + 1 + i) for i in range(terms_left - 1))
            if new_remaining > max_rest:
                break  # No point trying larger d
            
            # Also need new_remaining > 0 (already ensured by val <= remaining)
            # and new_remaining must be >= min possible (very small, skip)
            
            current.append(d)
            result = search(depth + 1, max_depth, new_remaining, d + 1, current)
            if result is not None:
                return result
            current.pop()
        
        return None
    
    # Try depths from 8 upward (we know 1-7 are impossible)
    best_result = greedy_result  # Fallback
    
    for max_depth in range(8, 16):
        if time.time() > deadline:
            break
        result = search(0, max_depth, one, 5, [])
        if result is not None:
            best_result = result
            break  # Found the minimum, stop
    
    if best_result is not None:
        return best_result
    
    # Last resort: return greedy result
    if greedy_result is not None:
        return greedy_result
    
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
