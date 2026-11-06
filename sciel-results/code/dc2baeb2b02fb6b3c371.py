from fractions import Fraction

def construct():
    # Target sum is 1
    target = Fraction(1, 1)
    min_denom = 5
    max_denom = 10000000
    
    # Try k from 8 upwards (theoretical min is 8)
    for k in range(8, 257):
        result = _search(k, min_denom, target)
        if result:
            return result
            
    raise RuntimeError("No solution found")

def _search(terms_left, start_denom, remaining_sum):
    if terms_left == 0:
        if remaining_sum == 0:
            return []
        return None
    
    if terms_left == 1:
        # Check if remaining_sum is a unit fraction with denom >= start_denom
        if remaining_sum.numerator == 1:
            d = remaining_sum.denominator
            if d >= start_denom and d <= 10000000:
                return [d]
        return None

    p, q = remaining_sum.numerator, remaining_sum.denominator
    
    # Lower bound for d: 1/d <= remaining_sum => d >= 1/remaining_sum
    # d >= ceil(q/p)
    min_d = max(start_denom, (q + p - 1) // p)
    
    # We iterate d from min_d upwards
    for d in range(min_d, 10000000):
        next_sum = remaining_sum - Fraction(1, d)
        
        # If next_sum < 0, we went too far (shouldn't happen if min_d is correct, but check)
        if next_sum < 0:
            break

        # Pruning: Can the remaining terms sum to next_sum?
        # Max possible sum of (terms_left - 1) distinct denominators >= d+1
        # is S_max = sum_{i=1}^{terms_left-1} 1/(d+i)
        
        # Fast check: S_max < (terms_left - 1) / (d + 1)
        # If next_sum > (terms_left - 1) / (d + 1), it's impossible.
        # Also, as d increases, next_sum increases and (terms_left - 1) / (d + 1) decreases.
        # So if this condition is met, we can break.
        if next_sum > Fraction(terms_left - 1, d + 1):
            break
            
        # Tighter check: Compute exact S_max
        # This is O(k), but k is small.
        max_next_sum = Fraction(0, 1)
        for j in range(1, terms_left):
            max_next_sum += Fraction(1, d + j)
            
        if next_sum > max_next_sum:
            # Impossible to form next_sum with remaining terms
            # Since increasing d makes next_sum larger and max_next_sum smaller, we can break
            break
            
        # Recurse
        res = _search(terms_left - 1, d + 1, next_sum)
        if res:
            return [d] + res
            
    return None
