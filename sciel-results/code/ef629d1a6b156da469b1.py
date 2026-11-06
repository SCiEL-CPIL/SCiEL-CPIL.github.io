"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import isqrt

def construct():
    deadline = time.time() + 55.0  # Safety margin before 60s limit
    
    target = Fraction(5, 31)
    
    # Start with known solution
    best_solution = [8, 31, 248]
    best_max = 248
    
    # Helper: check if a partial assignment can be completed
    # Given fixed denominators fixed[0..k-1] and remaining count r,
    # check if there exist r more distinct denominators > fixed[-1] 
    # (all <= max_allowed) that complete the sum.
    
    # Strategy: For each number of terms n from 2 to 8,
    # search for solutions with max denominator < best_max.
    # Use algebraic completion: fix n-1 denominators, solve for last.
    
    # For n terms: 1/d1 + ... + 1/dn = 5/31, d1 < d2 < ... < dn
    # Fix d1,...,d_{n-1}, then dn = 1/(5/31 - sum(1/di))
    # dn must be integer > d_{n-1} and < best_max
    
    # Bounds on d1:
    # n/d1 > 5/31 => d1 < 31n/5
    # 1/d1 < 5/31 => d1 > 31/5 = 6.2 => d1 >= 7
    # Also need enough room: sum of 1/(d1+1) + ... + 1/(d1+n-1) >= 5/31 - 1/d1
    
    def search_terms(n_terms, max_denom_limit):
        """Search for n-term solutions with all denominators < max_denom_limit."""
        if n_terms < 2:
            return None
        
        # d1 range
        d1_min = 7
        d1_max = min(int(31 * n_terms / 5), max_denom_limit - 1)
        
        if d1_max < d1_min:
            return None
        
        # We'll do recursive search fixing denominators one by one
        # and using algebraic completion for the last one
        
        def dfs(fixed, remaining_count, min_next):
            """
            fixed: list of chosen denominators so far
            remaining_count: how many more denominators to choose
            min_next: minimum value for next denominator
            Returns a complete solution or None.
            """
            if time.time() > deadline:
                return None
            
            current_sum = sum(Fraction(1, d) for d in fixed)
            remaining = target - current_sum
            
            if remaining <= 0:
                return None
            
            if remaining_count == 1:
                # Last denominator: 1/d_last = remaining
                # d_last = 1/remaining
                if remaining.numerator == 1:
                    d_last = remaining.denominator
                    if d_last > min_next and d_last < max_denom_limit:
                        if d_last not in fixed:
                            return fixed + [d_last]
                return None
            
            # Pruning: max possible sum from remaining_count denominators starting at min_next
            # is sum(1/(min_next) + 1/(min_next+1) + ... + 1/(min_next+remaining_count-1))
            # This is an upper bound on what we can still add
            max_addable = sum(Fraction(1, min_next + i) for i in range(remaining_count))
            if max_addable < remaining:
                return None
            
            # Min possible: we need at least 1/(min_next + remaining_count - 1) for the largest
            # Actually, the minimum sum is when denominators are as large as possible,
            # but they must be < max_denom_limit
            # min_addable = sum(1/(max_denom_limit - 1) + 1/(max_denom_limit - 2) + ...)
            # But this is complex; let's use a simpler bound
            # Each denominator >= min_next, so each 1/d <= 1/min_next
            # But we need sum >= remaining, so remaining_count * 1/min_next >= remaining
            # => min_next <= remaining_count / remaining
            if remaining_count / remaining < min_next:
                # Even the smallest possible denominators can't give enough sum
                return None
            
            # Range for next denominator d
            # 1/d < remaining (since remaining terms must sum to remaining, and all positive)
            # So d > 1/remaining
            d_min = max(min_next, int(1 / remaining) + 1)
            
            # 1/d + max possible from remaining_count-1 more terms >= remaining
            # 1/d + (remaining_count-1)/(d+1) >= remaining (upper bound on remaining sum)
            # Actually: 1/d + sum_{i=1}^{remaining_count-1} 1/(d+i) >= remaining
            # The max sum from remaining_count-1 terms starting at d+1 is sum(1/(d+i) for i in 1..remaining_count-1)
            # We need: 1/d + that_sum >= remaining
            # As d increases, 1/d decreases, so there's a max d
            # 1/d < remaining => d > 1/remaining (already have this)
            # Also: (remaining_count) / d > remaining is not quite right
            # Better: 1/d + (remaining_count-1)/(d+1) > remaining as a loose bound
            # Let's just use: d < remaining_count / remaining (from n/d > remaining => d < n/remaining)
            d_max = min(int(remaining_count / remaining), max_denom_limit - 1)
            
            if d_max < d_min:
                return None
            
            for d in range(d_min, d_max + 1):
                if time.time() > deadline:
                    return None
                new_fixed = fixed + [d]
                result = dfs(new_fixed, remaining_count - 1, d + 1)
                if result is not None:
                    return result
            
            return None
        
        # Try each d1
        for d1 in range(d1_min, d1_max + 1):
            if time.time() > deadline:
                return None
            # Check: can we complete with n_terms-1 more denominators > d1?
            result = dfs([d1], n_terms - 1, d1 + 1)
            if result is not None:
                return result
        
        return None
    
    # Search for solutions with increasing term counts, looking for smaller max
    # Try 3, 4, 5, 6, 7, 8 terms
    for n_terms in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline:
            break
        
        # For n terms, the max denominator must be at least... 
        # We want max < best_max, so search with limit = best_max
        result = search_terms(n_terms, best_max)
        if result is not None:
            new_max = max(result)
            if new_max < best_max:
                best_max = new_max
                best_solution = result
                # Continue searching for even better solutions
            else:
                # Found a solution but not better, try next term count
                pass
    
    # Additional targeted search: try to find 3-term solutions with small max
    # 1/a + 1/b + 1/c = 5/31, a < b < c
    # For a in 7..18:
    #   rem = 5/31 - 1/a
    #   1/b + 1/c = rem
    #   (rem*b*c - c - b) = 0... let's use: bc*rem = b + c
    #   b*c*(rem) - b - c = 0
    #   (b - 1/rem)(c - 1/rem) = 1/rem^2
    #   Let p = rem.numerator, q = rem.denominator, rem = p/q
    #   1/b + 1/c = p/q => q(b+c) = p*b*c => p*b*c - q*b - q*c = 0
    #   (p*b - q)(p*c - q) = q^2
    #   So p*b - q and p*c - q are factor pairs of q^2
    
    for a in range(7, 19):
        if time.time() > deadline:
            break
        rem = target - Fraction(1, a)
        if rem <= 0:
            continue
        p = rem.numerator
        q = rem.denominator
        # (p*b - q)(p*c - q) = q^2
        # Let u = p*b - q, v = p*c - q, u*v = q^2
        # b = (u + q) / p must be integer > a
        # c = (v + q) / p must be integer > b
        q2 = q * q
        # Find factor pairs of q^2
        # u < v (since b < c)
        # u >= 1, v >= 1
        # b > a => (u + q) / p > a => u > p*a - q
        u_min = max(1, p * a - q + 1)
        
        found_better = False
        for u in range(1, isqrt(q2) + 1):
            if time.time() > deadline:
                break
            if q2 % u != 0:
                continue
            v = q2 // u
            if u > v:
                continue
            # Check b = (u + q) / p
            if (u + q) % p != 0:
                continue
            b = (u + q) // p
            if b <= a:
                continue
            # Check c = (v + q) / p
            if (v + q) % p != 0:
                continue
            c = (v + q) // p
            if c <= b:
                continue
            if c < best_max:
                best_max = c
                best_solution = [a, b, c]
                found_better = True
    
    # Also try 4-term with algebraic completion more aggressively
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # Fix a, b, then 1/c + 1/d = rem_bc
    # (p*c - q)(p*d - q) = q^2 where rem_bc = p/q
    
    for a in range(7, 25):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        # b range: 1/b < rem_a => b > 1/rem_a
        # 2/b > rem_a => b < 2/rem_a (for 2 more terms)
        # Actually for 3 more terms (b, c, d): 3/b > rem_a => b < 3/rem_a
        b_min = max(a + 1, int(1 / rem_a) + 1)
        b_max = min(int(3 / rem_a), best_max - 1)
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            rem_bc = rem_a - Fraction(1, b)
            if rem_bc <= 0:
                break
            # 1/c + 1/d = rem_bc
            p = rem_bc.numerator
            q = rem_bc.denominator
            q2 = q * q
            c_min = max(b + 1, int(1 / rem_bc) + 1)
            
            for u in range(1, isqrt(q2) + 1):
                if time.time() > deadline:
                    break
                if q2 % u != 0:
                    continue
                v = q2 // u
                if u > v:
                    continue
                if (u + q) % p != 0:
                    continue
                c = (u + q) // p
                if c < c_min or c <= b:
                    continue
                if (v + q) % p != 0:
                    continue
                d = (v + q) // p
                if d <= c:
                    continue
                if d < best_max:
                    best_max = d
                    best_solution = [a, b, c, d]
    
    # Try 5-term
    for a in range(7, 31):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        b_min = max(a + 1, int(1 / rem_a) + 1)
        b_max = min(int(4 / rem_a), best_max - 1)
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                break
            # Need 3 more: c, d, e with 1/c + 1/d + 1/e = rem_b
            # Fix c, then 1/d + 1/e = rem_bc
            c_min = max(b + 1, int(1 / rem_b) + 1)
            c_max = min(int(3 / rem_b), best_max - 1)
            
            for c in range(c_min, c_max + 1):
                if time.time() > deadline:
                    break
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    break
                # 1/d + 1/e = rem_c
                p = rem_c.numerator
                q = rem_c.denominator
                q2 = q * q
                d_min = max(c + 1, int(1 / rem_c) + 1)
                
                for u in range(1, isqrt(q2) + 1):
                    if time.time() > deadline:
                        break
                    if q2 % u != 0:
                        continue
                    v = q2 // u
                    if u > v:
                        continue
                    if (u + q) % p != 0:
                        continue
                    d = (u + q) // p
                    if d < d_min or d <= c:
                        continue
                    if (v + q) % p != 0:
                        continue
                    e = (v + q) // p
                    if e <= d:
                        continue
                    if e < best_max:
                        best_max = e
                        best_solution = [a, b, c, d, e]
    
    # Try 6-term
    for a in range(7, 38):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        b_min = max(a + 1, int(1 / rem_a) + 1)
        b_max = min(int(5 / rem_a), best_max - 1)
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                break
            c_min = max(b + 1, int(1 / rem_b) + 1)
            c_max = min(int(4 / rem_b), best_max - 1)
            
            for c in range(c_min, c_max + 1):
                if time.time() > deadline:
                    break
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    break
                d_min = max(c + 1, int(1 / rem_c) + 1)
                d_max = min(int(3 / rem_c), best_max - 1)
                
                for d in range(d_min, d_max + 1):
                    if time.time() > deadline:
                        break
                    rem_d = rem_c - Fraction(1, d)
                    if rem_d <= 0:
                        break
                    # 1/e + 1/f = rem_d
                    p = rem_d.numerator
                    q = rem_d.denominator
                    q2 = q * q
                    e_min = max(d + 1, int(1 / rem_d) + 1)
                    
                    for u in range(1, isqrt(q2) + 1):
                        if time.time() > deadline:
                            break
                        if q2 % u != 0:
                            continue
                        v = q2 // u
                        if u > v:
                            continue
                        if (u + q) % p != 0:
                            continue
                        e = (u + q) // p
                        if e < e_min or e <= d:
                            continue
                        if (v + q) % p != 0:
                            continue
                        f = (v + q) // p
                        if f <= e:
                            continue
                        if f < best_max:
                            best_max = f
                            best_solution = [a, b, c, d, e, f]
    
    # Final verification
    s = sum(Fraction(1, d) for d in best_solution)
    assert s == target, f"Sum {s} != {target}"
    assert len(set(best_solution)) == len(best_solution), "Duplicate denominators"
    assert all(1 <= d <= 10000000 for d in best_solution), "Denominator out of range"
    assert 1 <= len(best_solution) <= 256, "Invalid term count"
    
    return best_solution
# EVOLVE-BLOCK-END
