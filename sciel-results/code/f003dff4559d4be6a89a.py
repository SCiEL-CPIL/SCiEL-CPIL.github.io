import time
from fractions import Fraction
from math import gcd
from functools import lru_cache

def construct():
    deadline = time.time() + 55  # safety margin for 90s wall-clock limit
    
    target = Fraction(5, 31)
    
    # Known solution: [8, 31, 248] gives max=248
    best = [8, 31, 248]
    best_max = 248
    
    def verify(dens):
        s = Fraction(0, 1)
        for d in dens:
            s += Fraction(1, d)
        return s == target
    
    # Branch and bound: search for any valid expansion, tracking min max denominator
    # Key insight: try smaller denominators first to find solutions with small max early
    
    # Precompute: for a given remainder r and current max denominator C,
    # what's the minimum possible max denominator we could achieve?
    
    # Use iterative deepening on max denominator C
    # For each C from lower bound up, check if a valid expansion exists with all dens <= C
    
    # Lower bound: need at least one term, and 1/d <= 5/31, so d >= 31/5 = 6.2, so d >= 7
    # But we also need distinct denominators
    
    # For a given C, use DFS with strong pruning:
    # - denominators must be in [min_d, C]
    # - remaining sum must be achievable with remaining terms
    
    def can_achieve(rem, min_d, C, max_terms):
        """Can we represent rem as sum of distinct reciprocals from [min_d, C] with at most max_terms terms?"""
        if rem == 0:
            return True
        if max_terms <= 0:
            return False
        if rem < 0:
            return False
        
        # Upper bound: max sum with max_terms terms from [min_d, C]
        # is 1/min_d + 1/(min_d+1) + ... (largest reciprocals)
        if max_terms == 1:
            return rem.numerator == 1 and rem.denominator >= min_d and rem.denominator <= C
        
        # Max possible sum: use the largest reciprocals (smallest denominators)
        max_sum = Fraction(0, 1)
        for i in range(max_terms):
            d = min_d + i
            if d > C:
                break
            max_sum += Fraction(1, d)
        if rem > max_sum:
            return False
        
        # Min possible sum: use the smallest reciprocals (largest denominators)
        min_sum = Fraction(0, 1)
        for i in range(max_terms):
            d = C - i
            if d < min_d:
                break
            min_sum += Fraction(1, d)
        if rem < min_sum:
            return False
        
        # If rem is a single unit fraction, check directly
        if rem.numerator == 1:
            d = rem.denominator
            if min_d <= d <= C:
                return True
        
        return True  # Can't easily prune further, let DFS handle it
    
    def find_expansion(rem, min_d, C, max_terms, used, path):
        """Find an expansion of rem using denominators in [min_d, C], at most max_terms terms."""
        if time.time() > deadline:
            return None
        
        if rem == 0:
            return list(path)
        if max_terms <= 0:
            return None
        if rem < 0:
            return None
        
        # Pruning: check if achievable
        if max_terms == 1:
            if rem.numerator == 1:
                d = rem.denominator
                if min_d <= d <= C and d not in used:
                    return list(path) + [d]
            return None
        
        # Try denominators from min_d upward
        # d must satisfy: 1/d <= rem, so d >= 1/rem
        # Also d >= min_d
        if rem.numerator == 1:
            d_lower = max(min_d, rem.denominator)
        else:
            d_lower = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
        
        # Upper bound on d: need enough remaining sum
        # rem - 1/d >= (max_terms - 1) * (1/C)  [loose bound]
        # So 1/d <= rem - (max_terms-1)/C
        # d >= 1 / (rem - (max_terms-1)/C)
        
        for d in range(d_lower, C + 1):
            if time.time() > deadline:
                return None
            if d in used:
                continue
            
            new_rem = rem - Fraction(1, d)
            if new_rem < 0:
                continue
            
            # Prune: can remaining be achieved?
            if new_rem != 0:
                # Check if new_rem can be achieved with max_terms-1 terms from [d+1, C]
                if max_terms - 1 > 0:
                    # Quick check: max possible sum
                    max_possible = Fraction(0, 1)
                    for i in range(min(max_terms - 1, C - d)):
                        dd = d + 1 + i
                        max_possible += Fraction(1, dd)
                    if new_rem > max_possible:
                        continue
                    # Min possible sum
                    min_possible = Fraction(0, 1)
                    for i in range(min(max_terms - 1, C - d)):
                        dd = C - i
                        if dd <= d:
                            break
                        min_possible += Fraction(1, dd)
                    if new_rem < min_possible:
                        continue
            
            used.add(d)
            path.append(d)
            result = find_expansion(new_rem, d + 1, C, max_terms - 1, used, path)
            path.pop()
            used.discard(d)
            
            if result is not None:
                return result
        
        return None
    
    # Search for minimum C
    # Try C from 7 upward, for each C try increasing max_terms
    found_better = False
    for C in range(7, best_max):
        if time.time() > deadline - 5:
            break
        
        # Maximum terms needed: with denominators >= 7, each term <= 1/7
        # So need at least ceil(5/31 / (1/7)) = ceil(35/31) = 2 terms
        # But we need distinct, so let's try up to min(C-6, 20) terms
        max_terms_to_try = min(C - 6, 20)
        
        for terms in range(1, max_terms_to_try + 1):
            if time.time() > deadline - 5:
                break
            
            result = find_expansion(target, 7, C, terms, set(), [])
            if result is not None:
                if verify(result):
                    best = result
                    best_max = C
                    found_better = True
                    break
        
        if found_better:
            break
    
    if not found_better:
        if verify(best):
            return best
    
    return best
