"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    start = time.time()
    deadline = start + 85  # safety margin before 90s limit
    
    # Strategy: Greedy Egyptian fraction with forced denominator lower bounds
    # We need sum of 1/d_i = 1, with d_i >= 5, distinct, minimizing count
    
    # Known small solutions:
    # 1/5 + 1/6 + 1/8 + 1/12 + 1/30 = 1? Let's check
    # Actually, let's use a search approach
    
    # The minimum number of terms: 
    # With denominators >= 5, the greedy approach picks the largest possible unit fraction
    # that doesn't exceed the remaining sum.
    
    def greedy_egyptian(target_num, target_den, min_den, used, max_terms=256):
        """
        Greedy algorithm to represent target_num/target_den as sum of unit fractions.
        Each denominator must be >= min_den and not in used.
        Returns list of denominators or None if impossible within max_terms.
        """
        if target_den == 0:
            return None
        if target_num == 0:
            return []
        if target_num < 0:
            return None
        if len(used) >= max_terms:
            return None
            
        # Greedy: pick smallest d >= min_den such that 1/d <= target_num/target_den
        # i.e., d >= target_den/target_num
        # and d not in used
        import math
        min_possible = max(min_den, target_den // target_num + (1 if target_den % target_num else 0))
        if min_possible < 5:
            min_possible = 5
            
        d = min_possible
        while d <= 10000000 and len(used) < max_terms:
            if d not in used:
                # Check if 1/d <= target_num/target_den
                # i.e., target_den <= d * target_num
                if target_den <= d * target_num:
                    new_used = used | {d}
                    rem_num = target_num * d - target_den
                    rem_den = target_den * d
                    # Simplify
                    g = math.gcd(rem_num, rem_den)
                    rem_num //= g
                    rem_den //= g
                    
                    result = greedy_egyptian(rem_num, rem_den, d + 1, new_used, max_terms)
                    if result is not None:
                        return [d] + result
            d += 1
        return None
    
    # Try with target = 1/1, min_den = 5
    result = greedy_egyptian(1, 1, 5, set(), 256)
    if result is not None:
        return result
    
    # Fallback: try known solutions or raise error
    # Known: 1/5 + 1/7 + 1/8 + 1/9 + 1/10 + 1/12 + 1/14 + 1/15 + 1/18 + 1/20 + 1/21 + 1/24 + 1/28 + 1/30 + 1/35 + 1/40 + 1/42 + 1/45 + 1/50 + 1/56 + 1/60 + 1/70 + 1/72 + 1/84 + 1/90 + 1/100 + 1/105 + 1/120 + 1/126 + 1/140 + 1/150 + 1/168 + 1/180 + 1/210 + 1/252 + 1/280 + 1/315 + 1/360 + 1/420 + 1/504 + 1/630 + 1/840 + 1/1260 + 1/2520 = 1?
    # Actually that's not right. Let me just return a valid solution.
    
    # Simple valid solution with denominators >= 5:
    # 1/5 + 1/7 + 1/8 + 1/9 + 1/10 + 1/12 + 1/14 + 1/15 + 1/18 + 1/20 + 1/21 + 1/24 + 1/28 + 1/30 + 1/35 + 1/40 + 1/42 + 1/45 + 1/50 + 1/56 + 1/60 + 1/70 + 1/72 + 1/84 + 1/90 + 1/100 + 1/105 + 1/120 + 1/126 + 1/140 + 1/150 + 1/168 + 1/180 + 1/210 + 1/252 + 1/280 + 1/315 + 1/360 + 1/420 + 1/504 + 1/630 + 1/840 + 1/1260 + 1/2520
    # This is getting complex. Let me use a simpler known identity.
    
    # 1 = 1/5 + 1/6 + 1/8 + 1/12 + 1/30? Let's verify:
    # LCM = 120
    # 1/5 = 24/120, 1/6 = 20/120, 1/8 = 15/120, 1/12 = 10/120, 1/30 = 4/120
    # Sum = 24+20+15+10+4 = 73/120 != 1
    
    # Let me try: 1/5 + 1/6 + 1/8 + 1/12 + 1/30 doesn't work.
    
    # Actually, let me just compute it properly with a different approach.
    # The greedy algorithm should work. Let me verify it runs correctly.
    
    raise RuntimeError("no solution found")
# EVOLVE-BLOCK-END
