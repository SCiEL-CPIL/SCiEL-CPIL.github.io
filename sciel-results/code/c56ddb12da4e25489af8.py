"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import time

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 80  # 80 second safety margin
    MAX_DEN = 10**7
    
    # Try to find minimum-k Egyptian fraction with denominators in [5, MAX_DEN]
    # using depth-first search with strong pruning.
    
    def dfs(rem, min_den, terms_left, used):
        """
        rem: Fraction remaining to sum to 1
        min_den: next denominator must be >= min_den (and not in used)
        terms_left: how many more terms we can use
        used: frozenset of denominators already chosen
        Returns list of chosen denominators or None
        """
        if rem == 0:
            return []
        if terms_left == 0:
            return None
        
        import math
        # Lower bound on next denominator: 1/d <= rem => d >= 1/rem
        # 1/rem = rem.den / rem.num
        den_lb = max(min_den, math.ceil(rem.denominator / rem.numerator))
        
        if den_lb > MAX_DEN:
            return None
        
        # Pruning: even taking the largest possible terms (smallest denominators),
        # we must be able to reach rem. The max sum with terms_left terms starting 
        # from den_lb is at most terms_left / den_lb (each term <= 1/den_lb).
        if terms_left * Fraction(1, den_lb) < rem:
            return None
        
        # Also: the minimum possible sum with terms_left terms (using largest denominators)
        # is at least terms_left / MAX_DEN. If rem < terms_left / MAX_DEN, impossible.
        if rem < Fraction(terms_left, MAX_DEN):
            return None
        
        # Also, the min sum with terms_left terms: we need at least rem, but each 
        # term is positive, so we just need rem > 0 which is already checked.
        # Another prune: the maximum possible sum using the largest available 
        # denominators... actually the sum must be exactly rem, so we need 
        # sum of remaining terms = rem. Each term >= 1/MAX_DEN.
        # So terms_left / MAX_DEN >= rem is necessary? No, terms can be large,
        # so sum can be small. But we need sum = rem, so rem must be achievable.
        # rem must be >= terms_left * (1/MAX_DEN)? No, we can pick small denominators.
        # Actually rem must be <= terms_left * (1/den_lb) which we already check.
        # And rem must be > 0 which is checked.
        
        # Try each candidate denominator
        # Upper bound on den: 1/den >= rem / terms_left => den <= terms_left / rem
        # terms_left/rem = terms_left * rem.den / rem.num
        den_ub_frac = terms_left * rem.denominator // rem.numerator
        den_ub = min(MAX_DEN, max(den_lb, den_ub_frac))
        
        # Cap the search range for efficiency - use a larger window for early terms
        if terms_left >= 3:
            search_end = min(den_ub, den_lb + 2000000)
        elif terms_left == 2:
            search_end = min(den_ub, den_lb + 1000000)
        else:
            search_end = den_ub
        
        for den in range(den_lb, search_end + 1):
            if time.time() > deadline:
                return None
            if den in used:
                continue
            # Quick check: 1/den <= rem
            if den < rem.denominator // rem.numerator:
                continue
            term = Fraction(1, den)
            new_rem = rem - term
            # Quick prune: new_rem must be achievable with terms_left-1 terms
            if terms_left - 1 > 0:
                # Max possible: (terms_left-1) / (den+1)
                if (terms_left - 1) * Fraction(1, den + 1) < new_rem:
                    continue
                # Min possible: (terms_left-1) / MAX_DEN
                if new_rem < Fraction(terms_left - 1, MAX_DEN):
                    continue
            result = dfs(new_rem, den + 1, terms_left - 1, used | {den})
            if result is not None:
                return [den] + result
        return None
    
    # Try k from 6 upward (k=5 is impossible since 1/5+1/6+1/7+1/8+1/9 < 1)
    for k in range(6, 10):
        if time.time() > deadline:
            break
        result = dfs(Fraction(1, 1), 5, k, frozenset())
        if result is not None:
            # Verify solution
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(set(result)) == len(result) and all(5 <= d <= MAX_DEN for d in result):
                return result
    
    # Fallback to seed approach
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    raise RuntimeError("no valid representation found")

# EVOLVE-BLOCK-END
