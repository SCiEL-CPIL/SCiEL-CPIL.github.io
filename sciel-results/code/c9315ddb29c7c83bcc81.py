"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # k=8,9 are impossible. Try k=10 first.
    # If k=10 fails, try k=11, etc.
    for k in range(10, 16):
        if time.time() > deadline - 2:
            break
        result = _search(k, deadline)
        if result is not None:
            return result
    
    # Fallback: return a known valid solution for k=11 or higher if search fails
    # Known solution for k=11:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/x = 1?
    # Let's just use a pre-computed valid set if search fails.
    # A valid set for k=11:
    # [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] sum > 1
    # We need a set that sums to exactly 1.
    # If search fails, we can return a known set.
    # One known set for 1 with 10 terms is hard to find instantly, but 11 is easier.
    # However, the prompt says "minimize their count".
    # If we can't find 10, we should return the best we found.
    # But construct() must return a valid list.
    # If _search returns None, we need a fallback.
    
    # Let's ensure we have a valid fallback.
    # The previous fallback used a DP on 2520.
    # We can use a known Egyptian fraction for 1.
    # 1 = 1/2 + 1/3 + 1/6 (denoms < 5, invalid)
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 ... no.
    
    # Let's rely on the search for k=10. It should be fast with good pruning.
    # If it times out, we try k=11.
    
    # Fallback solution if all searches fail (should not happen for k=11)
    # A simple valid solution:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 is too big.
    # We can use the identity: 1/n = 1/(n+1) + 1/(n(n+1))
    # Start with 1/5.
    # 1/5 -> 1/6 + 1/30
    # 1/6 -> 1/7 + 1/42
    # 1/7 -> 1/8 + 1/56
    # 1/8 -> 1/9 + 1/72
    # 1/9 -> 1/10 + 1/90
    # 1/10 -> 1/11 + 1/110
    # 1/11 -> 1/12 + 1/132
    # 1/12 -> 1/13 + 1/156
    # 1/13 -> 1/14 + 1/182
    # 1/14 -> 1/15 + 1/210
    # This creates many terms.
    
    # Let's just return the result of the search. If it fails, raise an error or return a hardcoded valid set.
    # Hardcoded valid set for k=11:
    # [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] doesn't work.
    # Let's use the fallback from the previous code which worked (score 0.9).
    # The previous fallback found a solution.
    
    return _fallback()

def _search(k, deadline):
    """Search for k distinct denominators >= 5 summing to 1 using integer arithmetic."""
    
    def max_sum_frac(remaining, start_d):
        """Return (num, den) for sum of 1/(start_d+i) for i in range(remaining)."""
        if remaining == 0:
            return (0, 1)
        num, den = 0, 1
        for i in range(remaining):
            d = start_d + i
            num = num * d + den
            den = den * d
        g = gcd(num, den)
        return (num // g, den // g)
    
    def dfs(idx, num, den, last_denom, chosen):
        """Find k-idx denominators > last_denom, each >= 5, summing to num/den."""
        if time.time() > deadline:
            return False
        
        remaining = k - idx
        
        if remaining == 0:
            return num == 0
        
        if num <= 0 or num > den:
            return False
        
        # Prune: max possible sum of remaining terms with smallest denominators
        # If num/den > max_sum(remaining, last_denom+1), then impossible
        m_num, m_den = max_sum_frac(remaining, last_denom + 1)
        if num * m_den > m_num * den:
            return False
        
        if remaining == 1:
            if den % num == 0:
                d_val = den // num
                if d_val > last_denom and d_val >= 5 and d_val <= 10000000:
                    chosen.append(d_val)
                    return True
                chosen.pop() if False else None # dummy
            return False
        
        # Bounds for next denominator
        # 1/d < num/den => d > den/num
        min_d = max(last_denom + 1, den // num + 1)
        
        # Upper bound:
        # We need 1/d + max_sum(remaining-1, d+1) >= num/den
        # Approx: 1/d + (remaining-1)/(d+1) >= num/den
        # Let m = remaining - 1
        # (d+1 + md) / (d(d+1)) >= num/den
        # den * (m+1)d + den >= num * d^2 + num * d
        # num * d^2 + (num - den*(m+1))d - den <= 0
        # Roots of ax^2 + bx + c = 0
        a = num
        b = num - den * (remaining - 1)
        c = -den
        
        # d <= (-b + sqrt(b^2 - 4ac)) / 2a
        disc = b * b - 4 * a * c
        if disc < 0:
            return False
        sqrt_disc = int(disc ** 0.5)
        if sqrt_disc * sqrt_disc < disc:
            sqrt_disc += 1
            
        max_d = ( -b + sqrt_disc ) // (2 * a)
        max_d = min(10000000, max_d)
        
        if min_d > max_d:
            return False
        
        for d_val in range(min_d, max_d + 1):
            if time.time() > deadline:
                return False
            
            # 1/d_val must be < num/den
            if d_val * num <= den:
                continue
            
            new_num = num * d_val - den
            new_den = den * d_val
            
            if new_num <= 0:
                continue
            
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            chosen.append(d_val)
            if dfs(idx + 1, new_num, new_den, d_val, chosen):
                return True
            chosen.pop()
        
        return False
    
    chosen = []
    if dfs(0, 1, 1, 4, chosen):
        return chosen
    return None

def _fallback():
    # A known valid solution for k=11
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 is not 1.
    # Let's use the DP fallback from the original code which was robust.
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
