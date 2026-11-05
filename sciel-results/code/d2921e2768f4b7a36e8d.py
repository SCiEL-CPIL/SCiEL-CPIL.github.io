"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import time
import math

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 85  # 85 second safety margin
    MAX_DEN = 10**7
    
    # k=8 is the minimum: 1/5+...+1/12 > 1, but 1/5+...+1/11 < 1
    # We'll use integer arithmetic: represent sum as N/LCM
    # Strategy: DFS with integer remainders
    
    def dfs(rem_num, rem_den, min_den, terms_left, used_set):
        """
        rem = rem_num / rem_den (already reduced)
        Find terms_left more distinct denominators >= min_den such that their reciprocal sum = rem
        Returns list of denominators or None
        """
        if rem_num == 0:
            return []
        if terms_left == 0:
            return None
        
        # Lower bound: 1/d <= rem => d >= rem_den / rem_num
        den_lb = max(min_den, (rem_den + rem_num - 1) // rem_num)
        
        if den_lb > MAX_DEN:
            return None
        
        # Upper bound: 1/d >= rem/terms_left => d <= terms_left / rem = terms_left * rem_den / rem_num
        den_ub = min(MAX_DEN, (terms_left * rem_den) // rem_num)
        
        if den_lb > den_ub:
            return None
        
        # Prune: max possible sum with terms_left terms starting from den_lb
        # Each term <= 1/den_lb, so max sum = terms_left / den_lb
        # Need terms_left / den_lb >= rem => terms_left * rem_den >= rem_num * den_lb
        if terms_left * rem_den < rem_num * den_lb:
            return None
        
        # Prune: min possible sum - each term >= 1/MAX_DEN, but we can also bound by
        # the largest denominators. Actually, the sum must be exactly rem, and each
        # denominator is at most MAX_DEN, so min contribution per term is 1/MAX_DEN.
        # But this is very loose. Better: the remaining sum rem must be achievable.
        
        # Try candidate denominators
        # Limit search range
        search_end = min(den_ub, den_lb + 100000)
        
        for den in range(den_lb, search_end + 1):
            if time.time() > deadline:
                return None
            if den in used_set:
                continue
            
            # new_rem = rem - 1/den = (rem_num * den - rem_den) / (rem_den * den)
            new_num = rem_num * den - rem_den
            if new_num < 0:
                continue
            new_den = rem_den * den
            
            # Reduce fraction
            g = math.gcd(new_num, new_den)
            new_num //= g
            new_den //= g
            
            # Prune: can we achieve new_num/new_den with terms_left-1 terms?
            if terms_left - 1 > 0:
                # Max sum: (terms_left-1) / (den+1)
                # Need (terms_left-1)/(den+1) >= new_num/new_den
                # => (terms_left-1)*new_den >= new_num*(den+1)
                if (terms_left - 1) * new_den < new_num * (den + 1):
                    continue
                # Min sum: each term >= 1/MAX_DEN, so min sum = (terms_left-1)/MAX_DEN
                # This is usually not binding
                # Also: 1/d' <= new_rem for some d' >= den+1 => d' >= new_den/new_num
                next_lb = (new_den + new_num - 1) // new_num
                if next_lb > MAX_DEN:
                    continue
                next_ub = ((terms_left - 1) * new_den) // new_num
                if next_lb > next_ub:
                    continue
            
            result = dfs(new_num, new_den, den + 1, terms_left - 1, used_set | {den})
            if result is not None:
                return [den] + result
        
        return None
    
    # Try k=8 first (minimum possible)
    for k in range(8, 12):
        if time.time() > deadline:
            break
        result = dfs(1, 1, 5, k, frozenset())
        if result is not None:
            # Verify solution
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(set(result)) == len(result) and all(5 <= d <= MAX_DEN for d in result):
                return result
    
    # Fallback: try a known construction
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + ... 
    # Let's try a different approach: use the identity
    # 1 = 1/2 + 1/3 + 1/7 + 1/42, but denominators must be >= 5
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + ...
    
    # Use the fallback from seed
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
