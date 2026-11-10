import time
from fractions import Fraction

def construct():
    """
    Finds 8 distinct integer denominators in [5, 10000000] whose reciprocal sum is exactly 1.
    Returns the list of denominators.
    """
    target = Fraction(1, 1)
    min_denom = 5
    max_denom = 10000000
    target_count = 8
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # DFS to find the solution
    # state: current_denom_start, remaining_sum, current_list
    def dfs(start, rem_sum, current):
        if time.time() > deadline:
            return None
            
        if len(current) == target_count:
            if rem_sum == 0:
                return current
            return None
        
        if rem_sum <= 0:
            return None
            
        # Pruning:
        # 1. Maximum possible sum with remaining slots
        # The largest possible sum is achieved by picking the smallest available denominators:
        # start, start+1, ..., start + (remaining_slots - 1)
        remaining_slots = target_count - len(current)
        
        # If even the max possible sum is less than rem_sum, prune
        max_possible_sum = Fraction(0, 1)
        for i in range(remaining_slots):
            d = start + i
            if d > max_denom:
                break
            max_possible_sum += Fraction(1, d)
        
        if max_possible_sum < rem_sum:
            return None
            
        # 2. Minimum possible sum
        # The smallest possible sum is achieved by picking the largest possible denominators.
        # However, we just need to ensure it's possible to reach rem_sum.
        # A simple lower bound: 1/d_i >= 1/max_denom.
        # Sum >= remaining_slots / max_denom.
        # If rem_sum < remaining_slots / max_denom, it's impossible (though unlikely here).
        
        # 3. Constraint on next denominator
        # 1/d_next <= rem_sum / remaining_slots ? No.
        # 1/d_next >= rem_sum - (sum of max possible for others)
        
        # Standard bound:
        # d_next must satisfy:
        # 1/d_next <= rem_sum  => d_next >= 1/rem_sum
        # Also, the remaining sum after picking d_next must be achievable by remaining_slots - 1 terms.
        # Max sum for remaining: sum(1/(d_next + i)) for i in 0..slots-2
        # Min sum for remaining: 0
        
        # Lower bound for d_next:
        # We need rem_sum - 1/d_next to be <= max_possible_sum(remaining_slots - 1, d_next + 1)
        # And rem_sum - 1/d_next >= 0
        
        # Also d_next >= start
        # And d_next >= 1/rem_sum (since 1/d_next <= rem_sum is not required, 1/d_next can be larger? No, if 1/d_next > rem_sum, sum becomes negative. So 1/d_next <= rem_sum is required if all terms are positive? Yes.)
        # Wait, if 1/d_next > rem_sum, then rem_sum - 1/d_next < 0. Since other terms are positive, this is impossible.
        # So 1/d_next <= rem_sum => d_next >= 1/rem_sum.
        
        lower_bound = max(start, int(1 / rem_sum) + 1)
        if lower_bound < start:
            lower_bound = start
            
        # Upper bound for d_next:
        # We need to be able to form rem_sum - 1/d_next with remaining_slots - 1 terms, all > d_next.
        # The MAXIMUM sum we can get with remaining_slots - 1 terms starting from d_next + 1 is:
        # sum(1/(d_next + 1 + i)) for i in 0..slots-2
        # If this max sum < rem_sum - 1/d_next, then d_next is too small? No, if d_next is small, 1/d_next is large, rem_sum - 1/d_next is small.
        # If d_next is large, 1/d_next is small, rem_sum - 1/d_next is large.
        # We need rem_sum - 1/d_next <= max_sum_remaining.
        # This gives an upper bound on d_next?
        # Actually, if d_next is too large, 1/d_next is too small, so rem_sum - 1/d_next is too large to be formed by the remaining small terms?
        # No, the remaining terms are > d_next, so they are smaller.
        # So if d_next is too large, the remaining required sum is large, but the max sum of remaining terms is small.
        # So yes, there is an upper bound.
        
        # Let's just iterate d_next from lower_bound upwards.
        # To avoid infinite loop, we can estimate an upper bound.
        # rem_sum - 1/d_next <= (remaining_slots - 1) / (d_next + 1)
        # 1/d_next >= rem_sum - (remaining_slots - 1) / (d_next + 1)
        # This is complex. Let's just iterate and prune based on max_sum.
        
        # Optimization: 
        # If rem_sum * d_next < 1, then 1/d_next > rem_sum, impossible.
        # So d_next >= 1/rem_sum.
        
        # Iterate d_next
        # We can limit d_next. If d_next is very large, 1/d_next is very small.
        # The remaining sum is approx rem_sum.
        # Max sum of remaining terms (slots-1) starting from d_next+1 is approx (slots-1)/(d_next+1).
        # We need rem_sum - 1/d_next <= (slots-1)/(d_next+1).
        # For large d_next, 1/d_next is negligible.
        # rem_sum <= (slots-1)/d_next => d_next <= (slots-1)/rem_sum.
        
        upper_bound = int((remaining_slots - 1) / rem_sum) + 10
        if upper_bound < lower_bound:
            return None
            
        # Cap upper bound to max_denom
        if upper_bound > max_denom:
            upper_bound = max_denom
            
        for d_next in range(lower_bound, upper_bound + 1):
            if d_next > max_denom:
                break
            new_rem = rem_sum - Fraction(1, d_next)
            if new_rem < 0:
                continue
            # Prune if new_rem is too large for remaining slots
            # Max sum for remaining_slots - 1 terms starting from d_next + 1
            max_rem_sum = Fraction(0, 1)
            for i in range(remaining_slots - 1):
                max_rem_sum += Fraction(1, d_next + 1 + i)
            if new_rem > max_rem_sum:
                continue
                
            res = dfs(d_next + 1, new_rem, current + [d_next])
            if res:
                return res
        return None

    result = dfs(min_denom, target, [])
    if result:
        return result
    else:
        # Fallback: Return a valid but potentially non-optimal solution if DFS fails (should not happen)
        # Or raise error
        raise RuntimeError("No solution found within time limit")

