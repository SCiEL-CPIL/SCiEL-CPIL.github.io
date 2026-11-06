import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    target = Fraction(5, 31)
    
    def check_solution(C, max_terms=10):
        # Precompute suffix sums of reciprocals for pruning?
        # sum_rec[i] = 1/i + 1/(i+1) + ... + 1/C
        # This allows O(1) check for max possible sum.
        sum_rec = [Fraction(0)] * (C + 2)
        # sum_rec[C+1] = 0
        for i in range(C, 0, -1):
            sum_rec[i] = sum_rec[i + 1] + Fraction(1, i)
            
        def dfs(current_sum, last_d, path, depth):
            if time.time() > deadline:
                return None
            
            remaining = target - current_sum
            
            if remaining == 0:
                if path:
                    return list(path)
                return None
            
            if remaining < 0:
                return None
                
            if depth >= max_terms:
                return None
            
            # Pruning 1: Max possible sum from remaining denominators
            # We can pick any subset of {last_d+1, ..., C}.
            # The max sum is sum_rec[last_d + 1].
            if remaining > sum_rec[last_d + 1]:
                return None
                
            # Pruning 2: Min possible sum?
            # We need to pick at least one more term if remaining > 0.
            # The min sum of 1 term is 1/C.
            # But we can pick more terms.
            # If remaining is very small, we might need large denominators.
            # But denominators are bounded by C.
            # If remaining < 1/C, it's impossible?
            # No, 1/(C-1) + 1/C is larger than 1/C.
            # The smallest positive sum we can form is 1/C (picking just C).
            # But we must use distinct integers.
            # So if remaining < 1/C, we can't form it?
            # Wait, if we pick C, sum is 1/C.
            # If remaining < 1/C, we can't pick C.
            # Can we pick C-1? 1/(C-1) > 1/C.
            # So if remaining < 1/C, no subset of {1..C} sums to remaining?
            # Yes, because the smallest element is 1/C.
            # So if remaining < 1/C, return None.
            if remaining < Fraction(1, C):
                return None

            # Lower bound for next denominator d:
            # 1/d <= remaining => d >= 1/remaining
            min_d = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
            min_d = max(min_d, last_d + 1)
            
            # Upper bound for d:
            # We need remaining - 1/d to be formable by {d+1..C}.
            # Max sum of {d+1..C} is sum_rec[d+1].
            # So we need remaining - 1/d <= sum_rec[d+1].
            # 1/d >= remaining - sum_rec[d+1].
            # d <= 1 / (remaining - sum_rec[d+1])
            # This depends on d, so hard to compute directly.
            # But we can iterate d and break if condition fails.
            
            for d in range(min_d, C + 1):
                if time.time() > deadline:
                    return None
                
                # Check if remaining - 1/d can be formed by {d+1..C}
                rem_after = remaining - Fraction(1, d)
                if rem_after < 0:
                    # 1/d > remaining.
                    # Since d is increasing, 1/d is decreasing.
                    # So for larger d, 1/d will be even smaller.
                    # So we should continue.
                    pass
                else:
                    # Check if rem_after can be formed by {d+1..C}
                    if rem_after > sum_rec[d + 1]:
                        # rem_after is too large to be formed by the tail.
                        # As d increases, 1/d decreases, so rem_after increases.
                        # And sum_rec[d+1] decreases.
                        # So the condition rem_after > sum_rec[d+1] will persist.
                        break
                    
                    # Also check if rem_after is too small to be formed by at least one term?
                    # If rem_after > 0, we need at least one term.
                    # Smallest term is 1/C.
                    # If rem_after < 1/C, we can't form it?
                    # Wait, we can form 0 if we pick no terms.
                    # But rem_after > 0, so we must pick at least one term.
                    # So if rem_after < 1/C, it's impossible?
                    # No, 1/C is the smallest *available* term.
                    # So yes, if 0 < rem_after < 1/C, impossible.
                    if rem_after > 0 and rem_after < Fraction(1, C):
                         # Can we form rem_after with terms > d?
                         # The smallest term is 1/C.
                         # So if rem_after < 1/C, we can't pick any term.
                         # So this d is invalid.
                         # But maybe a larger d? No, larger d means smaller 1/d, larger rem_after.
                         # So if rem_after < 1/C, and we increase d, rem_after increases.
                         # It might become >= 1/C.
                         # So we don't break, we just skip this d?
                         # No, if rem_after < 1/C, we can't pick any term from {d+1..C} to sum to rem_after.
                         # So this d leads to a dead end.
                         # We continue to next d.
                         pass
                    else:
                        path.append(d)
                        res = dfs(current_sum + Fraction(1, d), d, path, depth + 1)
                        path.pop()
                        if res:
                            return res
            
            return None
        
        return dfs(Fraction(0), 0, [], 0)

    for C in range(10, 10000):
        if time.time() > deadline:
            break
        
        sol = check_solution(C)
        if sol:
            return sol
            
    # Fallback
    r = target
    vals = []
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        vals.append(d)
        r -= Fraction(1, d)
    return vals
