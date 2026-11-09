import time
from math import factorial

def construct():
    P = factorial(12)
    
    # Global deadline guard
    start_time = time.time()
    deadline = 50.0  # Leave margin for evaluation overhead
    
    def dfs(rem_prod, num_factors_left, last_factor, max_bound):
        if time.time() > deadline:
            return None
            
        if num_factors_left == 0:
            return [] if rem_prod == 1 else None
            
        if rem_prod == 1:
            return None # Cannot use 1
            
        # Pruning: 
        # The minimum product of the remaining 'num_factors_left' factors, 
        # which must be > last_factor and distinct, is 
        # (last_factor + 1) * (last_factor + 2) * ... * (last_factor + num_factors_left)
        # If rem_prod is less than this, no solution.
        min_prod = 1
        for i in range(1, num_factors_left + 1):
            min_prod *= (last_factor + i)
            if min_prod > rem_prod:
                break
        if min_prod > rem_prod:
            return None
            
        # The maximum product of the remaining 'num_factors_left' factors,
        # which must be <= max_bound, is
        # max_bound * (max_bound - 1) * ... * (max_bound - num_factors_left + 1)
        # If rem_prod is greater than this, no solution.
        max_prod = 1
        for i in range(num_factors_left):
            max_prod *= (max_bound - i)
            if max_prod > rem_prod:
                break
        if max_prod < rem_prod:
            return None

        # Iterate possible factors
        # f must be > last_factor and <= max_bound
        # and must divide rem_prod
        for f in range(last_factor + 1, max_bound + 1):
            if rem_prod % f != 0:
                continue
            
            # Additional pruning:
            # The remaining product is rem_prod // f
            # We need num_factors_left - 1 factors, all > f
            # Min product of these is (f+1)*...*(f + num_factors_left - 1)
            rem = rem_prod // f
            min_rem_prod = 1
            for i in range(1, num_factors_left):
                min_rem_prod *= (f + i)
                if min_rem_prod > rem:
                    break
            if min_rem_prod > rem:
                continue
                
            # Max product of these is max_bound * ... * (max_bound - num_factors_left + 2)
            # Wait, the factors must be <= max_bound.
            # The largest possible distinct factors <= max_bound are max_bound, max_bound-1, ...
            max_rem_prod = 1
            for i in range(num_factors_left - 1):
                max_rem_prod *= (max_bound - i)
                if max_rem_prod > rem:
                    break
            if max_rem_prod < rem:
                continue

            sub_sol = dfs(rem, num_factors_left - 1, f, max_bound)
            if sub_sol is not None:
                return [f] + sub_sol
                
        return None

    # Try M from lower bound upwards
    # Lower bound estimate: (12!)^(1/7) ~ 24
    for M in range(24, 200):
        if time.time() > deadline:
            break
            
        # Try k from 7 down to 1
        for k in range(7, 0, -1):
            sol = dfs(P, k, 12, M)
            if sol is not None:
                return sol
                
    # Fallback: return a valid solution (though suboptimal)
    return [P]
