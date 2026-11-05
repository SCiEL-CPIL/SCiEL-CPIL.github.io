from fractions import Fraction

def construct():
    # Greedy Egyptian fraction algorithm
    # Selects the smallest denominator d > current_max_d such that 1/d <= rem
    # This minimizes the remainder at each step.
    
    rem = Fraction(1, 1)
    min_d = 5
    denominators = []
    
    while rem != 0:
        # Find smallest d >= min_d such that 1/d <= rem
        # 1/d <= n/m  => m <= n*d => d >= m/n
        # d_min_possible = ceil(m/n)
        d_candidate = (rem.denominator + rem.numerator - 1) // rem.numerator
        
        if d_candidate < min_d:
            d = min_d
        else:
            d = d_candidate
            
        # Ensure 1/d <= rem
        # If d was chosen as ceil, 1/d <= rem is guaranteed if d >= 1/rem.
        # If d = min_d, we must check.
        if Fraction(1, d) > rem:
            # This case shouldn't happen if min_d is large enough or logic is correct
            # But if min_d is small and rem is very small, 1/min_d might be > rem.
            # We need the smallest d such that 1/d <= rem.
            # If 1/min_d > rem, then d must be larger.
            # d >= 1/rem.
            d = (rem.denominator + rem.numerator - 1) // rem.numerator
            if d < min_d:
                # If even min_d is too small (1/min_d > rem), we can't pick min_d.
                # We must pick d = ceil(1/rem).
                # But if ceil(1/rem) < min_d, then 1/min_d < rem?
                # If ceil(1/rem) < min_d, then 1/rem < min_d => 1/min_d < rem.
                # So if 1/min_d > rem, then min_d < 1/rem => ceil(1/rem) >= min_d.
                # So d = ceil(1/rem) will be >= min_d.
                pass
        
        denominators.append(d)
        rem = rem - Fraction(1, d)
        min_d = d + 1
        
        # Safety check to prevent infinite loops or excessive terms
        if len(denominators) > 256:
            break
            
    return denominators
