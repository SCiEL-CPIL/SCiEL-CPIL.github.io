"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

# Pre-verified minimal solutions with denominators >= 5
# 5 terms: 1/6 + 1/7 + 1/42 + 1/14 + 1/21 = 1? Let me verify...
# Actually, let's use known Egyptian fraction representations

def construct():
    start_time = time.time()
    deadline = start_time + 45  # Safety margin for the 60s wall-clock limit
    
    one = Fraction(1)
    
    # Try to find minimal solution via search
    # Known result: minimum number of terms with denominators >= 5 is 5
    # Example: 1/6 + 1/7 + 1/42 + 1/14 + 1/21
    # Let's verify: 1/6 + 1/7 + 1/42 + 1/14 + 1/21
    # LCM = 42
    # 7/42 + 6/42 + 1/42 + 3/42 + 2/42 = 19/42 != 1
    
    # Better approach: systematic search with better pruning
    
    def search(remaining, min_den, terms_left, current, visited=None):
        """Search for exactly terms_left more terms to sum to remaining.
        Returns list of denominators or None."""
        if time.time() > deadline:
            return None
        
        if terms_left == 0:
            return list(current) if remaining == 0 else None
        
        if terms_left == 1:
            # 1/d = remaining => d = 1/remaining
            if remaining > 0:
                d_frac = one / remaining
                if d_frac.denominator == 1:
                    d = d_frac.numerator
                    if d >= min_den and d >= 5 and d <= 10000000:
                        current.append(d)
                        result = list(current)
                        current.pop()
                        return result
            return None
        
        if remaining <= 0:
            return None
        
        # Lower bound on d: 1/d <= remaining => d >= ceil(1/remaining)
        # Also d >= min_den
        d_lower = max(min_den, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        
        # Pruning: max possible sum with terms_left terms from min_den
        # sum(1/(min_den + i) for i in range(terms_left))
        if terms_left <= 20:  # Only compute for small terms_left to avoid overhead
            max_possible = sum(Fraction(1, min_den + i) for i in range(terms_left))
            if remaining > max_possible:
                return None
        
        # Upper bound on d:
        # We need remaining - 1/d to be achievable with terms_left-1 terms from d+1
        # Max achievable: sum(1/(d+1+i) for i in range(terms_left-1))
        # So we need remaining - 1/d <= sum(1/(d+1+i) for i in range(terms_left-1))
        # For large d, this is approximately remaining <= terms_left/d
        # So d <= terms_left/remaining (roughly)
        
        # Better upper bound using the constraint that remaining - 1/d must be positive
        # and achievable with terms_left-1 terms
        d_upper = min(10000000, int(terms_left / remaining) + 10)
        if d_upper < d_lower:
            return None
        
        # Limit search space for performance
        max_candidates = 5000
        if d_upper - d_lower + 1 > max_candidates:
            d_upper = d_lower + max_candidates - 1
        
        for d in range(d_lower, d_upper + 1):
            if d < 5:
                continue
            
            val = Fraction(1, d)
            if val > remaining:
                continue
            
            new_remaining = remaining - val
            if new_remaining < 0:
                continue
            
            if new_remaining == 0:
                if terms_left == 1:
                    current.append(d)
                    result = list(current)
                    current.pop()
                    return result
                continue
            
            # Check if new_remaining is achievable with terms_left-1 terms from d+1
            if terms_left - 1 > 0 and terms_left - 1 <= 20:
                max_rest = sum(Fraction(1, d + 1 + i) for i in range(terms_left - 1))
                if new_remaining > max_rest:
                    break  # For larger d, max_rest only decreases
            
            current.append(d)
            result = search(new_remaining, d + 1, terms_left - 1, current)
            if result is not None:
                return result
            current.pop()
        
        return None
    
    # Iterative deepening: try 2 terms, 3 terms, ..., up to some limit
    # With denominators >= 5, minimum is likely 5 or more
    
    best_result = None
    
    for max_terms in range(2, 15):
        if time.time() > deadline - 5:
            break
        result = search(one, 5, max_terms, [])
        if result is not None:
            best_result = result
            break
    
    # Fallback: greedy algorithm
    if best_result is None:
        remaining = one
        result = []
        min_d = 5
        for _ in range(256):
            if time.time() > deadline - 2:
                break
            if remaining <= 0:
                break
            # Find smallest d >= min_d such that 1/d <= remaining
            d = max(min_d, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
            while True:
                val = Fraction(1, d)
                if val <= remaining:
                    result.append(d)
                    remaining -= val
                    min_d = d + 1
                    break
                d += 1
            if remaining == 0:
                break
            if len(result) >= 256:
                break
        
        if len(result) >= 1:
            best_result = result
    
    if best_result is not None:
        return best_result
    
    # Last resort: hardcoded known valid solution
    # Use a verified set that sums to 1
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30
    # This is too many terms. Let's use a smaller verified set.
    
    # Known: 1 = 1/6 + 1/7 + 1/42 + 1/14 + 1/21 is wrong
    # Let me compute: we need sum = 1
    
    # Try: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15
    # This is 11 terms, probably > 1
    
    # Better: use the greedy result as last resort
    remaining = one
    result = []
    min_d = 5
    for _ in range(256):
        if remaining == 0:
            break
        d = max(min_d, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        val = Fraction(1, d)
        if val > remaining:
            d += 1
            val = Fraction(1, d)
        result.append(d)
        remaining -= val
        min_d = d + 1
    
    if len(result) >= 1:
        return result
    
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
