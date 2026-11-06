import time
from fractions import Fraction

def construct():
    """
    Find 1..256 distinct integer denominators in [5,10000000] whose reciprocal sum is exactly 1.
    Minimize their count.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    # Known minimal solutions for Egyptian fractions summing to 1 with denominators >= 5
    # The minimum number of terms is 8.
    
    # Known 8-term solutions with denominators >= 5:
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/21 + 1/30 + 1/105 + 1/110? No, let's use verified ones.
    
    # A well-known 8-term decomposition:
    # 1 = 1/2 + 1/3 + 1/7 + 1/42 + 1/1806 + 1/... 
    # We need denominators >= 5, so we can't use 1/2 or 1/3 directly.
    
    # Let me think about this differently. We need:
    # 1/a1 + 1/a2 + ... + 1/ak = 1 where 5 <= a1 < a2 < ... < ak <= 10000000
    
    # For k=8, one known solution:
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/21 + 1/35 + 1/42? Let me check...
    # Actually, let me use a systematic approach.
    
    # Known: 1 = 1/6 + 1/7 + 1/8 + 1/12 + 1/14 + 1/24 + 1/28 + 1/56? Let me verify.
    # Or better, use a constructive approach.
    
    # Let me use the following known decomposition:
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/21 + 1/30 + 1/70 + 1/140? 
    # Sum = 1/5 + 1/6 + 1/7 + 1/10 + 1/21 + 1/30 + 1/70 + 1/140
    # = 0.2 + 0.1667 + 0.1429 + 0.1 + 0.0476 + 0.0333 + 0.0143 + 0.0071 ≈ 0.8119, not 1.
    
    # Let me use a different approach. Start with known decompositions.
    
    # 1/2 = 1/3 + 1/6, so 1 = 1/3 + 1/3 + 1/6 (but 3 < 5)
    
    # Let's use: 1 = 1/5 + 1/5 + 1/5 + 1/5 + 1/5 (but denominators must be distinct)
    
    # For distinct denominators >= 5, the minimum is indeed 8 terms.
    # One valid 8-term solution:
    # 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/12 + 1/14 + 1/35? Let me calculate...
    
    # Actually, let me use a verified solution:
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/21 + 1/35 + 1/42
    # Check: 1/5+1/6+1/7+1/10+1/14+1/21+1/35+1/42
    # LCM of 5,6,7,10,14,21,35,42 = 210
    # 42/210 + 35/210 + 30/210 + 21/210 + 15/210 + 10/210 + 6/210 + 5/210
    # = (42+35+30+21+15+10+6+5)/210 = 164/210 ≠ 1
    
    # Let me try another approach. Use the identity:
    # 1/n = 1/(n+1) + 1/(n(n+1))
    # Starting from 1/5:
    # 1/5 = 1/6 + 1/30
    # 1/30 = 1/31 + 1/930
    # This gives: 1 = 1/5 + 1/6 + 1/31 + 1/930 + ... but we need to sum to 1.
    
    # Actually, I need to be more careful. Let me just search for a valid solution.
    
    # A known 8-term Egyptian fraction for 1 with distinct denominators >= 5:
    # 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/12 + 1/14 + 1/140? 
    # Let me compute with Fraction.
    
    # Let me try: [5, 6, 7, 10, 14, 21, 30, 70]
    # 1/5+1/6+1/7+1/10+1/14+1/21+1/30+1/70
    # LCM = 210
    # 42+35+30+21+15+10+7+3 = 163/210, not 1.
    
    # Try [5, 6, 7, 10, 14, 21, 35, 42]:
    # 42+35+30+21+15+10+6+5 = 164/210, not 1.
    
    # Let me use a more systematic approach. I'll do a DFS search.
    
    def dfs(remaining, start_denom, terms, depth, max_depth):
        """
        remaining: Fraction, the remaining sum to achieve
        start_denom: minimum denominator to consider (for distinctness, in increasing order)
        terms: list of denominators chosen so far
        depth: current depth
        max_depth: maximum depth allowed
        """
        if time.time() > deadline:
            return None
        if depth == max_depth:
            if remaining == 0:
                return terms
            return None
        if remaining == 0:
            return terms if len(terms) >= 1 else None
        
        # remaining must be positive
        if remaining <= 0:
            return None
        
        # The next denominator d must satisfy: 1/d <= remaining
        # So d >= 1/remaining
        # Also d > start_denom for distinctness
        # And d >= 5
        min_d = max(start_denom + 1, 5)
        
        # Upper bound: we need (max_depth - depth) * (1/min_d) >= remaining
        # So min_d <= (max_depth - depth) / remaining
        # Thus d <= floor((max_depth - depth) / remaining)
        # But this is a loose bound. A better bound:
        # The largest denominator we can use is bounded by the fact that
        # the remaining (max_depth - depth) terms must sum to 'remaining',
        # and the largest term is 1/d, so we need (max_depth - depth)/d >= remaining
        # => d <= (max_depth - depth)/remaining
        
        max_d_candidate = int((max_depth - depth) / remaining)
        if max_d_candidate < min_d:
            return None
        max_d = min(max_d_candidate, 10000000)
        
        for d in range(min_d, max_d + 1):
            if 1 / Fraction(d) > remaining:
                continue
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                break
            result = dfs(new_remaining, d, terms + [d], depth + 1, max_depth)
            if result is not None:
                return result
        return None
    
    # Try to find solutions with increasing number of terms
    for k in range(8, 12):  # Start from known minimum of 8
        if time.time() > deadline:
            break
        result = dfs(Fraction(1, 1), 4, [], 0, k)
        if result is not None:
            return result
    
    # If DFS didn't find anything, fall back to a known solution or raise
    # Let's try a different strategy: use known decompositions
    # 1 = 1/2 + 1/2, but we need distinct and >= 5
    
    # Another approach: 1 = 1/5 + 1/5 + 1/5 + 1/5 + 1/5 won't work (not distinct)
    
    # Let me try a different decomposition strategy.
    # 1 = 1/5 + 4/5
    # 4/5 = 1/6 + 19/30
    # 19/30 = 1/7 + 81/210 = 1/7 + 27/70
    # 27/70 = 1/8 + 151/560 = ...
    
    # This is getting complex. Let me try a broader search with a time limit.
    
    # Alternative: use a known valid 8-term solution
    # After more thought, let me verify: [6, 7, 8, 9, 10, 12, 14, 35]
    s = Fraction(1,6) + Fraction(1,7) + Fraction(1,8) + Fraction(1,9) + Fraction(1,10) + Fraction(1,12) + Fraction(1,14) + Fraction(1,35)
    if s == 1:
        return [6, 7, 8, 9, 10, 12, 14, 35]
    
    # Try [5, 6, 7, 10, 14, 21, 30, 42]
    s = Fraction(1,5) + Fraction(1,6) + Fraction(1,7) + Fraction(1,10) + Fraction(1,14) + Fraction(1,21) + Fraction(1,30) + Fraction(1,42)
    if s == 1:
        return [5, 6, 7, 10, 14, 21, 30, 42]
    
    # Let me compute some sums to find a valid one
    # I'll do a small search
    
    from itertools import combinations
    
    # This could be too slow, so let's just return a reasonable solution
    # Actually, let me try one more known decomposition:
    # 1/3 = 1/4 + 1/12, 1/4 = 1/5 + 1/20
    # So 1/3 = 1/5 + 1/20 + 1/12
    # 1/2 = 1/3 + 1/6 = 1/5 + 1/20 + 1/12 + 1/6
    # 1 = 1/2 + 1/2 = 2*(1/5 + 1/20 + 1/12 + 1/6) but denominators aren't distinct
    
    # Let me just return a valid solution I can verify:
    # 1 = 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + ...
    # Sum of 1/6 through 1/12: 1/6+1/7+1/8+1/9+1/10+1/11+1/12 ≈ 0.1667+0.1429+0.125+0.1111+0.1+0.0909+0.0833 = 0.82
    
    # For the last resort, return a solution with more terms that I can verify
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + ...
    # This might not sum to exactly 1 with a finite number of distinct denominators.
    
    # Let me use a constructive approach with the greedy method, but ensuring distinct denominators >= 5.
    
    # Greedy Egyptian fraction:
    # At each step, take the smallest denominator d >= max(5, last_denom+1) such that 1/d <= remaining
    # This doesn't guarantee we'll reach exactly 1, but let's try.
    
    remaining = Fraction(1, 1)
    terms = []
    last_denom = 4
    
    while remaining > 0 and len(terms) < 256 and time.time() < deadline:
        # Find smallest d > last_denom, d >= 5, such that 1/d <= remaining
        d = max(5, last_denom + 1)
        while 1 / Fraction(d) > remaining:
            d += 1
        terms.append(d)
        remaining -= Fraction(1, d)
        last_denom = d
    
    if remaining == 0 and len(terms) >= 1:
        return terms
    
    # Final fallback: this should never be reached if we have a valid solution
    raise RuntimeError("no valid representation found")
