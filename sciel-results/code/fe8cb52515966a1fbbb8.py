import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline with safety margin
    
    # Theoretical minimum k is 8.
    # We search for k = 8, 9, 10...
    for k in range(8, 20):
        if time.time() > deadline:
            break
        
        # Backtracking search for k denominators
        result = _search_k_terms(k, start_time, deadline)
        if result is not None:
            return result
    
    # Fallback: Return a known valid solution if we didn't find optimal
    # A known solution for k=10 (or similar) can be hardcoded or found via simple greedy
    # But given the search is fast, this is unlikely to be reached.
    # For safety, return a valid set. 
    # Example: 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 + ... 
    # Actually, let's just return the result of a quick greedy or a known small set.
    # A valid set of 10 denominators >= 5:
    # 1 = 1/2 + 1/3 + 1/7 + 1/42 + 1/140 + 1/455 + 1/1365 + 1/5460 ... 
    # This uses 2 and 3. We need >= 5.
    # Let's use the fact that 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 is not 1.
    # 1/5+1/6+1/7+1/10+1/14+1/35 = (42+35+30+21+15+6)/210 = 149/210 != 1.
    
    # Safe fallback: use a larger k that is easier to find or a hardcoded valid one.
    # Let's just raise an error if we fail, but we should not.
    # A valid solution with k=11:
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 ?
    # Sum H_15 - H_4 = (1+1/2+...+1/15) - (1+1/2+1/3+1/4) = 3.318 - 2.083 = 1.235 > 1.
    # So we can pick a subset.
    
    # Actually, the search for k=8 should be very fast.
    # If it fails, k=9 is also very fast.
    
    return None

def _search_k_terms(k, start_time, deadline):
    """Search for k distinct denominators >= 5 summing reciprocals to 1."""
    
    # Precompute max sums for pruning? 
    # max_sum(n, start) = sum_{i=0}^{n-1} 1/(start+i)
    # We can compute this on the fly.
    
    def backtrack(remaining_terms, min_denom, remaining_sum, current):
        # remaining_sum is a Fraction
        if time.time() > deadline:
            return None
        
        if remaining_terms == 0:
            if remaining_sum == 0:
                return list(current)
            return None
        
        if remaining_terms == 1:
            # Need 1/d = remaining_sum
            if remaining_sum > 0:
                inv = Fraction(1, 1) / remaining_sum
                if inv.denominator == 1:
                    d = inv.numerator
                    if d >= min_denom and d <= 10000000:
                        current.append(d)
                        return list(current)
            return None
        
        # Pruning: Max possible sum with remaining_terms terms starting from min_denom
        # is 1/min_denom + 1/(min_denom+1) + ... + 1/(min_denom + remaining_terms - 1)
        max_possible = Fraction(0, 1)
        for i in range(remaining_terms):
            max_possible += Fraction(1, min_denom + i)
        if max_possible < remaining_sum:
            return None
        
        # Lower bound on next denominator:
        # 1/d <= remaining_sum => d >= 1/remaining_sum
        # d >= min_denom
        inv = Fraction(1, 1) / remaining_sum
        if inv.denominator == 1:
            low_d = max(min_denom, inv.numerator)
        else:
            low_d = max(min_denom, inv.numerator // inv.denominator + 1)
        
        # Upper bound on next denominator:
        # We need remaining_terms - 1 more terms with denominators > d.
        # Max sum of remaining terms: sum_{i=1}^{remaining_terms-1} 1/(d+i)
        # We need remaining_sum - 1/d <= max_rem_sum
        # 1/d >= remaining_sum - max_rem_sum
        # d <= 1 / (remaining_sum - max_rem_sum)
        
        # Iterate d from low_d upwards
        # To avoid infinite loop, we can bound d.
        # As d increases, 1/d decreases, so remaining_sum - 1/d increases.
        # max_rem_sum decreases as d increases.
        # So the condition (remaining_sum - 1/d) <= max_rem_sum will eventually fail.
        
        # Let's iterate d
        # We can also stop if d is too large.
        # A rough upper bound: d < 1/remaining_sum * remaining_terms
        # Let's just iterate and rely on the pruning check inside.
        
        # Optimization: 
        # If remaining_terms == 2:
        #   1/d1 + 1/d2 = R
        #   1/d2 = R - 1/d1
        #   d2 = 1/(R - 1/d1)
        #   We can just iterate d1 and check if d2 is integer and valid.
        
        if remaining_terms == 2:
            for d1 in range(low_d, 10000000):
                if time.time() > deadline:
                    return None
                if d1 >= min_denom:
                    rem = remaining_sum - Fraction(1, d1)
                    if rem > 0:
                        inv_rem = Fraction(1, 1) / rem
                        if inv_rem.denominator == 1:
                            d2 = inv_rem.numerator
                            if d2 > d1 and d2 <= 10000000:
                                current.append(d1)
                                current.append(d2)
                                return list(current)
                # Break condition: if 1/d1 is so small that even with 1/(d1+1) we can't reach R?
                # No, R is fixed.
                # If 1/d1 < R, we proceed.
                # If 1/d1 >= R, we break (but low_d ensures 1/low_d <= R).
                # Actually, if 1/d1 > R, then rem < 0.
                # Since low_d is ceil(1/R), 1/low_d <= R.
                # As d1 increases, 1/d1 decreases.
                # We must ensure d2 > d1.
                # d2 = 1/(R - 1/d1).
                # As d1 increases, R - 1/d1 increases, so d2 decreases.
                # So d2 < d1 will eventually happen.
                # We can break if d2 <= d1.
                # But we compute d2 inside.
                # Let's add a break condition:
                # If 1/d1 is very small, d2 is large.
                # If d2 <= d1, we can break?
                # d2 = d1 / (R*d1 - 1).
                # d2 > d1 => d1 / (R*d1 - 1) > d1 => 1 > R*d1 - 1 => R*d1 < 2 => d1 < 2/R.
                # So if d1 >= 2/R, then d2 <= d1.
                # So we can break when d1 >= 2/R.
                if d1 * remaining_sum.numerator >= 2 * remaining_sum.denominator:
                    break
            return None

        # General case for remaining_terms > 2
        # Iterate d
        # Upper bound for d:
        # We need R - 1/d <= sum_{i=1}^{remaining_terms-1} 1/(d+i)
        # This is true for small d. For large d, LHS ~ R, RHS ~ (remaining_terms-1)/d.
        # So d ~ (remaining_terms-1)/R.
        
        # Let's just iterate d from low_d.
        # To prevent long loops, we can stop if d > 10000000.
        # Also, if d is large, 1/d is small.
        
        # Let's add a heuristic break:
        # If d is such that 1/d is very small, the remaining sum is close to R.
        # The max sum of remaining terms is approx (remaining_terms-1)/d.
        # So we need R <= (remaining_terms-1)/d => d <= (remaining_terms-1)/R.
        
        max_d_est = int((remaining_terms - 1) * remaining_sum.denominator / remaining_sum.numerator) + 10
        
        for d in range(low_d, max_d_est + 100):  # Add small buffer
            if time.time() > deadline:
                return None
            
            if d > 10000000:
                break
                
            frac_d = Fraction(1, d)
            if frac_d >= remaining_sum:
                continue
            
            new_remaining = remaining_sum - frac_d
            
            # Check if new_remaining is achievable with remaining_terms - 1 terms > d
            # Max possible: sum_{i=1}^{remaining_terms-1} 1/(d+i)
            max_rem = Fraction(0, 1)
            for i in range(1, remaining_terms):
                max_rem += Fraction(1, d + i)
            
            if new_remaining > max_rem:
                # If new_remaining is too big, it means 1/d was too small?
                # No, if 1/d is small, new_remaining is large.
                # max_rem decreases as d increases.
                # So if new_remaining > max_rem, increasing d will make new_remaining larger and max_rem smaller.
                # So we can break?
                # Wait.
                # new_remaining = R - 1/d. As d increases, 1/d decreases, so new_remaining increases.
                # max_rem = sum 1/(d+i). As d increases, max_rem decreases.
                # So if new_remaining > max_rem, it will remain true for larger d.
                # So we can break.
                break
            
            current.append(d)
            result = backtrack(remaining_terms - 1, d + 1, new_remaining, current)
            if result is not None:
                return result
            current.pop()
        
        return None
    
    return backtrack(k, 5, Fraction(1, 1), [])
