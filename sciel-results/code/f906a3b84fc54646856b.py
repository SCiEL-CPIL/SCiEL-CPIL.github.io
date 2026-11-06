"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    deadline = time.time() + 55  # safety margin before 60s limit
    target_num = 5
    target_den = 31
    
    best = [8, 31, 248]
    best_max = 248
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def reduce(n, d):
        g = gcd(n, d)
        return n // g, d // g
    
    def sub_frac(n, d, k):
        # n/d - 1/k = (n*k - d) / (d*k)
        num = n * k - d
        den = d * k
        if num <= 0:
            return None
        return reduce(num, den)
    
    # Recursive backtracking search
    # State: remaining fraction (num, den), minimum allowed denominator, current terms
    # Goal: find terms such that sum of 1/term = target, minimizing max(term)
    
    def search(rem_num, rem_den, min_k, terms, max_den_limit):
        nonlocal best, best_max
        if time.time() > deadline:
            return
        
        # If rem is exactly 1/k, we can finish
        if rem_num == 1:
            k = rem_den
            if k > min_k and k < best_max and k <= max_den_limit:
                candidate = terms + [k]
                if len(candidate) <= 256 and all(1 <= x <= 10000000 for x in candidate):
                    # Verify distinctness
                    if len(set(candidate)) == len(candidate):
                        cm = max(candidate)
                        if cm < best_max:
                            best_max = cm
                            best = candidate
            return
        
        # Upper bound on k: need rem_num/rem_den <= (remaining_terms) / k
        # For at least 1 more term: 1/k >= rem_num/rem_den => k <= rem_den/rem_num
        # But we also need to be able to represent the remainder with more terms
        # If we have m terms left to place, we need: rem_num/rem_den <= m/k
        # So k <= m * rem_den / rem_num
        # Also k > min_k
        
        # Lower bound: k > rem_den/rem_num (since 1/k < rem means k > rem_den/rem_num)
        # Actually 1/k < rem_num/rem_den means k > rem_den/rem_num
        
        # For the current level, we need at least 1 term
        # k_min: k > min_k, and 1/k < rem => k > rem_den/rem_num
        k_min = max(min_k + 1, rem_den // rem_num + 1)
        
        # k_max: we need enough remaining terms to sum to rem
        # With m terms each at most 1/k_min_remaining, sum <= m/k_min_remaining
        # But simpler: for the last term to work, we need 1/k >= rem eventually
        # Conservative: k <= floor(m * rem_den / rem_num) where m is max terms remaining
        # We'll use a generous bound and let the recursion prune
        
        # Max terms we can still add: up to 256 - len(terms), but practically limited
        # Use a reasonable max depth
        max_terms_remaining = min(256 - len(terms), 10)  # cap for efficiency
        
        # k must satisfy: rem_num/rem_den > 1/k (so k > rem_den/rem_num)
        # and we need to be able to complete with remaining terms
        # The largest k we'd try: if we use just 1 more term, k = rem_den/rem_num (if integer)
        # If we use more terms, k can be smaller
        # Upper bound: k < rem_den / rem_num * max_terms_remaining (rough)
        # But actually, for k to be valid, we need 1/k < rem, so k > rem_den/rem_num
        # And we need rem - 1/k >= 0, so k >= rem_den/rem_num (but strict for positive remainder)
        
        # Let's bound k: k must be > rem_den/rem_num and k < best_max
        # Also, the new remainder must be positive and representable
        k_upper = min(best_max - 1, max_den_limit)
        # Also, we need at least 1 more term after this, so the new remainder must be >= 1/(k+1) approximately
        # But that's handled in recursion
        
        if k_min > k_upper:
            return
        
        # Try k values in range [k_min, k_upper]
        # Optimize: try larger k first? No, try to find small max_den
        # Actually, to minimize max denominator, we want all terms small
        # But the last term could be large. Let's just search systematically.
        
        for k in range(k_min, k_upper + 1):
            if time.time() > deadline:
                return
            
            new_rem = sub_frac(rem_num, rem_den, k)
            if new_rem is None:
                continue
            
            new_num, new_den = new_rem
            
            # Pruning: check if new_rem can be represented with remaining terms
            # Minimum possible sum with (max_terms_remaining-1) terms each >= 1/(k+1) is 0
            # Maximum possible sum with (max_terms_remaining-1) terms each <= 1/(k+1) is (max_terms_remaining-1)/(k+1)
            # We need new_rem <= (max_terms_remaining-1)/(k+1)
            if max_terms_remaining - 1 > 0:
                # new_num/new_den <= (max_terms_remaining-1)/(k+1)
                # new_num * (k+1) <= new_den * (max_terms_remaining-1)
                if new_num * (k + 1) > new_den * (max_terms_remaining - 1):
                    continue
            
            # Also, new_rem must be > 0 (already checked)
            # And we need at least 1 more term, so new_rem >= 1/(k+1) is NOT required
            # (the next term could be much larger)
            
            search(new_num, new_den, k, terms + [k], max_den_limit)
    
    # Start search
    search(target_num, target_den, 1, [], 10000000)
    
    return best
# EVOLVE-BLOCK-END
