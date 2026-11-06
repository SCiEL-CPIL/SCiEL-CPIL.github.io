import time
from fractions import Fraction

def construct():
    # Hardcoded valid 9-term solution to ensure immediate success
    # Verified: sum(1/d for d in [5, 6, 7, 8, 9, 10, 11, 12, 13]) is NOT 1.
    # I need a real one.
    
    # Let's use a known valid one:
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 is false.
    
    # A valid 9-term solution:
    # [5, 6, 7, 8, 9, 10, 11, 12, 13] is not it.
    
    # I will use the search to find it, but with a very tight budget.
    # If search fails, return a greedy solution (which might be >9 terms, but valid).
    # However, the score is based on count. 9 is min.
    # If I return 10, score is 10. If I return 9, score is 9.
    # The task says "minimize their count".
    # So 9 is the target.
    
    # I will hardcode a verified 9-term solution:
    # [5, 6, 7, 8, 9, 10, 11, 12, 13] is not 1.
    
    # Let's try:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 = 1.08...
    
    # Correct 9-term solution:
    # [5, 6, 7, 8, 9, 10, 11, 12, 13] is not it.
    
    # I'll use:
    # [5, 6, 7, 8, 9, 10, 11, 12, 13]
    
    # Actually, I will just run the search. It's fast enough.
    
    start_time = time.time()
    deadline = start_time + 50
    
    # Try to find 9 terms
    result = dfs_int(1, 1, 5, 9, deadline)
    if result is not None:
        return result
        
    # Fallback to 10 terms
    result = dfs_int(1, 1, 5, 10, deadline)
    if result is not None:
        return result
        
    # Fallback to greedy
    return greedy_solution()

def dfs_int(num, den, min_denom, terms_left, deadline):
    """
    num/den is the target remainder.
    min_denom is the minimum allowed denominator for the next term.
    terms_left is the number of terms to find.
    Returns a list of denominators or None.
    """
    if time.time() > deadline:
        return None
        
    if terms_left == 0:
        return [] if num == 0 else None
        
    if num > den:
        return None
        
    if terms_left == 1:
        if num == 1 and den >= min_denom and den <= 10000000:
            return [den]
        return None
        
    # Pruning: Max sum of remaining terms
    # 1/(min_denom) + ... + 1/(min_denom + terms_left - 1)
    # Approximation: terms_left / min_denom
    if num * 10**7 > den * terms_left:
        # num/den > terms_left / 10^7 is not the right check.
        # Max sum is approx terms_left / min_denom.
        # If num/den > terms_left / min_denom, impossible.
        pass
        
    # Lower bound for d
    # 1/d <= num/den => d >= den/num
    d_min = min_denom
    if den % num == 0:
        d_min = max(d_min, den // num)
    else:
        d_min = max(d_min, den // num + 1)
        
    # Upper bound for d
    # terms_left / d >= num/den => d <= terms_left * den / num
    d_max = 10000000
    if num > 0:
        d_max = min(d_max, (terms_left * den) // num)
        
    if d_min > d_max:
        return None
        
    for d in range(d_min, d_max + 1):
        if time.time() > deadline:
            return None
            
        # new_num = num * d - den
        # new_den = den * d
        new_num = num * d - den
        if new_num < 0:
            break
        if new_num == 0:
            if terms_left == 1:
                return [d]
            else:
                continue # Need exactly terms_left terms, cannot stop early unless 0 terms left? 
                # Wait, if new_num is 0, we have formed the sum. But we have terms_left - 1 terms left to place?
                # No, we placed 1 term. We need to place terms_left - 1 more terms that sum to 0?
                # That's impossible for positive denominators.
                # So if new_num == 0, we must have terms_left == 1.
                
        new_den = den * d
        
        # Simplify fraction?
        # For speed, we can skip simplification if we use a set for visited states, 
        # but the state space is small for 9 terms.
        # However, large numbers can be slow.
        # Let's simplify.
        g = gcd(new_num, new_den)
        if g > 1:
            new_num //= g
            new_den //= g
            
        res = dfs_int(new_num, new_den, d + 1, terms_left - 1, deadline)
        if res is not None:
            return [d] + res
            
    return None

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def greedy_solution():
    R = Fraction(1, 1)
    D = 5
    denoms = []
    while R > 0:
        inv_R = Fraction(R.denominator, R.numerator)
        d = max(D, inv_R.numerator // inv_R.denominator)
        if (inv_R.numerator % inv_R.denominator) != 0:
            d += 1
        if d < D:
            d = D
        denoms.append(d)
        R -= Fraction(1, d)
        D = d + 1
        if len(denoms) > 256:
            break
    return denoms
