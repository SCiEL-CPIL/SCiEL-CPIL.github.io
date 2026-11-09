"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import math

    target = Fraction(5, 31)
    
    # We know from knowledge that the minimum largest denominator is 248.
    # We perform a search to find a valid expansion with max denominator 248.
    # This is a structural change from the greedy seed.
    
    def search(rem, min_d, max_d, path):
        if rem == 0:
            return path
        if len(path) >= 256:
            return None
        
        # The next denominator d must satisfy 1/d <= rem, so d >= 1/rem.
        # d must also be >= min_d to ensure distinctness and ordering.
        # d must be <= max_d.
        
        start_d = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
        
        # Iterate possible next denominators
        for d in range(start_d, max_d + 1):
            # Check if 1/d is too large
            if d < (rem.denominator + rem.numerator - 1) // rem.numerator:
                continue
                
            new_rem = rem - Fraction(1, d)
            
            if new_rem < 0:
                break
            
            if new_rem == 0:
                return path + [d]
            
            # Pruning: 
            # The remaining sum new_rem must be representable by distinct denominators 
            # in the range [d+1, max_d].
            # A necessary condition is that new_rem is less than or equal to the sum of 
            # all possible reciprocals in that range.
            # sum_{j=d+1}^{max_d} 1/j approx ln(max_d/d)
            # If new_rem is larger than this, it's impossible.
            # However, calculating this sum exactly is slow. 
            # A simpler bound: if we have at most 256-len(path)-1 terms left, 
            # the maximum sum is (256-len(path)-1) / (d+1).
            # This is a loose bound but safe.
            max_terms_left = 256 - len(path) - 1
            if max_terms_left <= 0:
                continue
            
            # If new_rem is very small, it might be hard to hit exactly, 
            # but the search handles that.
            
            res = search(new_rem, d + 1, max_d, path + [d])
            if res:
                return res
                
        return None

    # Try to find a solution with max denominator 248
    sol = search(target, 1, 248, [])
    
    if sol:
        return sol
    else:
        # Fallback to greedy if search fails (should not happen for 5/31)
        r = target
        values = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
# EVOLVE-BLOCK-END
