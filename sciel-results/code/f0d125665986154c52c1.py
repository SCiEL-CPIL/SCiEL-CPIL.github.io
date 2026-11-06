import time
from fractions import Fraction

def construct():
    start_time = time.time()
    
    # Fallback: The 12-term solution from the seed program
    def get_fallback():
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
        raise RuntimeError("no baseline representation")

    # Try to find an 11-term solution
    # Target: 1, Min Denom: 5, Count: 11
    # We use backtracking.
    
    target_frac = Fraction(1, 1)
    min_den = 5
    
    def backtrack(remaining, min_d, current):
        if time.time() - start_time > 5.0:
            return None
        if len(current) == 0:
            return None
        if len(current) == 1:
            if remaining > 0:
                d = 1 / remaining
                if d == int(d) and int(d) >= min_d and int(d) not in current:
                    return current + [int(d)]
            return None
        
        # Pruning:
        # Max possible sum with remaining terms (assuming smallest possible denominators)
        # If remaining > sum(1/i for i in range(min_d, min_d + len(current))), impossible?
        # No, we can pick larger denominators, which makes sum smaller.
        # So if remaining < min_possible_sum, impossible?
        # Min possible sum is with largest denominators? No, largest denominators give smallest sum.
        # We need to reach exactly 'remaining'.
        # Max sum we can add is with smallest denominators.
        # If remaining > max_sum_possible, we can't reach it?
        # Wait. If we need to add 'remaining' to the sum, and we have 'k' slots.
        # The maximum value we can add is if we pick the smallest available denominators.
        # So if remaining > max_addable, we are stuck? No, that means we need to add MORE than we can?
        # No, if remaining is large, we need small denominators.
        # If remaining is small, we need large denominators.
        
        # Let's check bounds.
        # Max term is 1/min_d.
        # Min term is 1/max_d (10^7).
        # If remaining > len(current) / min_d, impossible.
        if remaining > len(current) / min_d:
            return None
            
        # If remaining < len(current) / 10000000, impossible (too small).
        if remaining < len(current) / 10000000:
            return None

        # Try denominators
        # Heuristic: try denominators that keep the sum close to target.
        # d must be >= min_d
        # d must be <= 1/remaining (if we could use all remaining slots for this d? No)
        # Actually, 1/d <= remaining is not required if we have more terms.
        # But 1/d must be <= remaining if it's the last term.
        # For intermediate terms, 1/d can be > remaining / len(current)?
        
        # Let's iterate d from min_d upwards.
        # To avoid huge loops, we can estimate the range.
        # If we pick d, the new remaining is remaining - 1/d.
        # The next min_d will be d + 1.
        
        # Limit the range of d to keep search fast.
        # If d is too small, remaining becomes too large for remaining slots?
        # If d is too large, remaining becomes too small?
        
        # Upper bound for d:
        # We need to fit 'len(current)-1' terms into 'remaining - 1/d'.
        # The max sum of 'len(current)-1' terms starting from d+1 is approx (len(current)-1)/(d+1).
        # So we need remaining - 1/d <= (len(current)-1)/(d+1) ?
        # This gives a lower bound on d? No, upper bound on 1/d?
        # remaining - 1/d <= (len(current)-1)/(d+1)
        # remaining <= 1/d + (len(current)-1)/(d+1)
        # If this inequality is violated, d is too small (1/d is too big, sum too big).
        # So we need d to be large enough?
        # No, if d is small, 1/d is big. Sum is big.
        # If Sum > remaining, we overshoot.
        # So we need 1/d <= remaining? Not necessarily, but if 1/d > remaining, we can't subtract it.
        # So d >= 1/remaining.
        
        min_d_candidate = max(min_d, int(1/remaining) + 1)
        
        # Max d?
        # If d is very large, 1/d is very small.
        # We need to make up 'remaining' with 'len(current)-1' terms.
        # The max sum of those terms is approx (len(current)-1) / (d+1).
        # So we need remaining - 1/d <= (len(current)-1) / (d+1).
        # This is always true for large d?
        # remaining <= 1/d + (len(current)-1)/(d+1)
        # For large d, RHS is small. LHS is fixed.
        # So d cannot be too large.
        # Solve for d:
        # remaining * d * (d+1) <= (d+1) + d * (len(current)-1)
        # This is a quadratic.
        
        # Let's just iterate d from min_d_candidate up to a limit.
        # Limit: 1 / (remaining / len(current)) ?
        # Average term is remaining / len(current).
        # So d is around len(current) / remaining.
        
        max_d_candidate = int(len(current) / remaining) * 10 + 100
        # Also cap at 10000000
        max_d_candidate = min(max_d_candidate, 10000000)
        
        for d in range(min_d_candidate, max_d_candidate + 1):
            if d in current:
                continue
            term = Fraction(1, d)
            if term > remaining:
                continue
            new_remaining = remaining - term
            res = backtrack(new_remaining, d + 1, current + [d])
            if res:
                return res
        return None

    # Try to find 11 terms
    # Start with empty set, min_den 5
    # But we need to pick 11 terms.
    # The backtrack function above assumes 'current' is the list of picked terms.
    # I need to modify it to track the count.
    
    def backtrack_count(remaining, min_d, current, k):
        if time.time() - start_time > 5.0:
            return None
        if k == 0:
            if remaining == 0:
                return current
            return None
        
        # Pruning
        if remaining > k / min_d:
            return None
        if remaining < k / 10000000:
            return None
            
        min_d_candidate = max(min_d, int(1/remaining) + 1)
        max_d_candidate = int(k / remaining) * 10 + 100
        max_d_candidate = min(max_d_candidate, 10000000)
        
        for d in range(min_d_candidate, max_d_candidate + 1):
            if d in current:
                continue
            term = Fraction(1, d)
            if term > remaining:
                continue
            res = backtrack_count(remaining - term, d + 1, current + [d], k - 1)
            if res:
                return res
        return None

    # Try k=11
    sol_11 = backtrack_count(Fraction(1, 1), 5, [], 11)
    if sol_11:
        return sol_11

    # Try k=10? Unlikely to exist for min denom 5.
    # If not found, return fallback
    return get_fallback()
