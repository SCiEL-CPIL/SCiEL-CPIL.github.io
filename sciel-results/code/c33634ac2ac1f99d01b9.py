import time
from fractions import Fraction
from math import ceil

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 10 # We can increase this if needed, but likely 3-5 terms suffice
DEADLINE = 85.0

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def check_time():
        return time.time() > deadline

    # Known solution
    best_solution = [8, 31, 248]
    best_max = 248

    # Precompute reciprocal sums for pruning?
    # We'll just compute on the fly, it's fast enough for small depths.
    
    def max_sum_recip(start, count):
        """Sum of 1/(start), 1/(start+1), ..., 1/(start+count-1)"""
        s = Fraction(0, 1)
        for i in range(count):
            s += Fraction(1, start + i)
        return s

    def min_sum_recip(end, count):
        """Sum of 1/end, 1/(end-1), ..., 1/(end-count+1)"""
        s = Fraction(0, 1)
        for i in range(count):
            s += Fraction(1, end - i)
        return s

    def search(remaining, min_d, max_d, depth, max_terms):
        """
        Search for a solution to 'remaining' using distinct denominators 
        in [min_d, max_d], with at most 'max_terms - depth' terms left.
        Returns list of denominators or None.
        """
        if check_time():
            return None
        
        if remaining == 0:
            return []
        
        if depth >= max_terms:
            return None
            
        if remaining < 0:
            return None
        
        # Pruning 1: Max possible sum with remaining terms
        # We can pick at most (max_terms - depth) terms.
        # To maximize the sum, we pick the smallest available denominators: min_d, min_d+1, ...
        terms_left = max_terms - depth
        # If we use all terms_left, the max sum is sum of reciprocals of min_d ... min_d+terms_left-1
        # But we can also use fewer terms? No, we must sum to exactly 'remaining'.
        # If we use fewer terms, the denominators must be smaller to compensate?
        # Actually, using fewer terms means each term is larger? No.
        # To reach a target 'remaining', if we use fewer terms, we need larger values for 1/d, i.e., smaller d.
        # But d is constrained to be >= min_d.
        # So the *maximum* sum we can achieve with *any* number of terms <= terms_left
        # is actually achieved by using the *most* terms (smallest denominators).
        # Wait. 1/7 + 1/8 > 1/7.
        # So adding more terms (with larger denominators) increases the sum?
        # No, 1/8 < 1/7.
        # Sum of 1/7 + 1/8 = 0.225.
        # Sum of 1/7 = 0.142.
        # So adding terms *increases* the sum.
        # Therefore, the maximum sum we can form with up to `terms_left` terms
        # is the sum of the `terms_left` smallest available denominators.
        
        max_possible_sum = max_sum_recip(min_d, terms_left)
        if remaining > max_possible_sum:
            return None

        # Pruning 2: Min possible sum with remaining terms
        # To minimize the sum, we pick the largest available denominators: max_d, max_d-1, ...
        # But we must pick *distinct* denominators.
        # The minimum sum is achieved by picking the largest possible denominators.
        # We can pick up to `terms_left` terms.
        # To minimize the sum, we should pick the *fewest* terms?
        # 1/100 = 0.01. 1/100 + 1/99 = 0.02.
        # Adding terms increases the sum.
        # So the minimum sum is achieved by picking *one* term, the largest one: max_d.
        # Wait, is that right?
        # We need to sum to `remaining`.
        # If `remaining` is very small, we need a large denominator.
        # If we pick max_d, sum = 1/max_d.
        # If 1/max_d > remaining, we can't use max_d alone. We might need to combine?
        # No, if 1/max_d > remaining, we can't use max_d because all other denominators are smaller, so their reciprocals are larger.
        # So if 1/max_d > remaining, it's impossible?
        # Yes, because min(1/d) for d in [min_d, max_d] is 1/max_d.
        # So if remaining < 1/max_d, we can't form it?
        # Wait. We can sum multiple terms.
        # But all terms are positive.
        # So the smallest positive sum we can form is 1/max_d (using just max_d).
        # Any other combination will be larger.
        # So if remaining < 1/max_d, return None.
        
        if remaining < Fraction(1, max_d):
            return None

        # Pruning 3: Bounds on the next denominator d
        # 1/d <= remaining  => d >= ceil(1/remaining)
        d_min = max(min_d, ceil(remaining.denominator / remaining.numerator) if remaining.numerator != 0 else min_d)
        
        # Also, we need to be able to form (remaining - 1/d) with the remaining terms.
        # Let rem_next = remaining - 1/d.
        # We need rem_next >= 0.
        # And rem_next must be formable by terms in [d+1, max_d].
        # Max sum for rem_next is max_sum_recip(d+1, terms_left - 1).
        # So remaining - 1/d <= max_sum_recip(d+1, terms_left - 1).
        # 1/d >= remaining - max_sum_recip(d+1, terms_left - 1).
        # d <= 1 / (remaining - max_sum_recip(d+1, terms_left - 1)).
        # This depends on d.
        
        # Simpler upper bound for d:
        # The smallest sum we can form for the rest (terms_left - 1 terms from [d+1, max_d])
        # is 1/max_d (if we only use 1 term) or more.
        # Actually, the minimum sum we can form with *up to* terms_left-1 terms from [d+1, max_d]
        # is 1/max_d.
        # So we need remaining - 1/d >= 1/max_d ?
        # No, we need remaining - 1/d to be *at least* the minimum possible sum of the rest.
        # The minimum possible sum of the rest is 1/max_d (using just max_d).
        # So remaining - 1/d >= 1/max_d.
        # 1/d <= remaining - 1/max_d.
        # d >= 1 / (remaining - 1/max_d).
        # This gives a lower bound, which we already have (d_min).
        
        # What about an upper bound for d?
        # If d is too large, 1/d is too small.
        # remaining - 1/d will be large.
        # Can we form a large sum with the remaining terms?
        # The max sum of the remaining terms (terms_left - 1 terms from [d+1, max_d])
        # is max_sum_recip(d+1, terms_left - 1).
        # So we need remaining - 1/d <= max_sum_recip(d+1, terms_left - 1).
        # If this is not satisfied, d is too large.
        
        d_max = max_d
        # We can iterate d from d_min to d_max.
        # But we can break early if d is too large.
        
        for d in range(d_min, d_max + 1):
            if check_time():
                return None
            
            # Check if d is used? We are passing min_d, so d >= min_d.
            # And we recurse with min_d = d+1, so distinctness is handled.
            
            rem_next = remaining - Fraction(1, d)
            if rem_next < 0:
                break # Since d is increasing, 1/d is decreasing, rem_next is increasing?
                # No. 1/d decreases as d increases.
                # remaining - 1/d increases as d increases.
                # So if rem_next < 0, it will be more negative for smaller d?
                # We are iterating d from d_min (small) to d_max (large).
                # At d_min, 1/d is largest, so rem_next is smallest.
                # If rem_next < 0 at d_min, it might become >= 0 for larger d?
                # No, if 1/d_min > remaining, then 1/d < 1/d_min < remaining for all d > d_min?
                # Yes. So if rem_next < 0, it stays < 0?
                # Wait. 1/d decreases. So -1/d increases.
                # So rem_next increases.
                # So if rem_next < 0, it might become >= 0 later.
                # So we should NOT break. We should continue.
                # But d_min was calculated as ceil(1/remaining).
                # So 1/d_min <= remaining.
                # So rem_next >= 0 always.
                pass
            
            if rem_next == 0:
                return [d]
            
            # Pruning for the recursive call
            # Check if rem_next can be formed by terms in [d+1, max_d] with terms_left-1 terms.
            
            # Min sum check: rem_next must be >= 1/max_d
            if rem_next < Fraction(1, max_d):
                # If d is larger, rem_next is larger. So we might be able to form it?
                # No, 1/max_d is the *minimum* sum we can form with *any* number of terms from [d+1, max_d]?
                # Yes, the smallest positive sum is 1/max_d.
                # So if rem_next < 1/max_d, it's impossible.
                # As d increases, rem_next increases.
                # So for larger d, rem_next might be >= 1/max_d.
                # So we cannot break. We just skip this d.
                continue
            
            # Max sum check: rem_next must be <= max_sum_recip(d+1, terms_left - 1)
            # If terms_left - 1 == 0, max sum is 0. But rem_next > 0. So fail.
            if terms_left - 1 == 0:
                continue
                
            max_sum_rest = max_sum_recip(d + 1, terms_left - 1)
            if rem_next > max_sum_rest:
                # If d increases, 1/d decreases, rem_next increases.
                # max_sum_rest decreases (starts from d+2 instead of d+1).
                # So the gap widens.
                # If rem_next > max_sum_rest, it will likely be true for larger d too.
                # So we can break.
                break
            
            res = search(rem_next, d + 1, max_d, depth + 1, max_terms)
            if res is not None:
                return [d] + res
        
        return None

    # Search for minimal M
    # Start from a reasonable lower bound
    # 1/7 + 1/8 + 1/9 + 1/10 = 0.142 + 0.125 + 0.111 + 0.1 = 0.478 > 0.161
    # So we can't use many small terms.
    # Try M from 50 to 247
    
    # Optimization: If we find a solution for M, we can return immediately.
    # We want the *smallest* M.
    
    for M in range(50, 248):
        if check_time():
            break
        
        # We need to ensure we have enough terms.
        # With M=50, min denom 7.
        # 1/7 + 1/8 + ...
        # Let's just use MAX_TERMS = 10.
        
        res = search(TARGET, 7, M, 0, MAX_TERMS)
        if res is not None:
            # Verify
            s = sum(Fraction(1, d) for d in res)
            if s == TARGET:
                best_solution = res
                best_max = M
                break

    return best_solution
