import time
import sys
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit

    def elapsed():
        return time.time() - start_time

    def time_ok():
        return elapsed() < deadline

    def find_solution(k):
        """
        Finds k distinct integers d_1 < ... < d_k >= 5 such that sum(1/d_i) = 1.
        Returns the list of denominators or None if not found.
        """
        
        def dfs(rem_num, rem_den, min_d, depth, max_depth, path):
            if not time_ok():
                return None
            if rem_num == 0:
                return path
            if depth == max_depth:
                return None
            
            terms_left = max_depth - depth
            
            # Pruning 1: Upper bound on the sum of remaining terms
            # The maximum sum is achieved by the smallest possible denominators: min_d, min_d+1, ...
            # Approximate sum: terms_left / min_d
            # If rem > terms_left / min_d, prune.
            if rem_num * min_d > rem_den * terms_left:
                return None

            # Pruning 2: Lower bound on the sum of remaining terms
            # The minimum sum is achieved by the largest possible denominators (10^7).
            # If rem < terms_left / 10^7, prune.
            if rem_num * 10**7 < rem_den * terms_left:
                return None

            # Determine the range for the next denominator d
            # 1/d <= rem  => d >= rem_den / rem_num
            d_min = max(min_d, rem_den // rem_num)
            if rem_den % rem_num == 0:
                d_min = rem_den // rem_num
            else:
                d_min = rem_den // rem_num + 1

            # Upper bound for d:
            # We need 1/d + sum_{j=1}^{terms_left-1} 1/(d+j) >= rem
            # A loose upper bound is d <= terms_left / rem
            d_max = (terms_left * rem_den) // rem_num
            
            if d_min > d_max:
                return None
            
            # Iterate through possible denominators
            for d in range(d_min, d_max + 1):
                if not time_ok():
                    return None
                # new_rem = rem - 1/d = (rem_num*d - rem_den) / (rem_den*d)
                new_num = rem_num * d - rem_den
                if new_num < 0:
                    break
                new_den = rem_den * d
                res = dfs(new_num, new_den, d + 1, depth + 1, max_depth, path + [d])
                if res is not None:
                    return res
            return None

        return dfs(1, 1, 5, 0, k, [])

    # The minimum possible k is likely around 7-10.
    # We try k=5 first. If found, it's the optimal solution.
    for k in range(5, 21):
        if not time_ok():
            break
        res = find_solution(k)
        if res is not None:
            return res
    
    # Fallback: return a known valid solution if search fails
    # Example: 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 + 1/42 + 1/70 + 1/105 + 1/210 + 1/420 + ...
    # Actually, let's just raise an error if we can't find one, but we should try to return something.
    # A known solution with 10 terms?
    # If we can't find one, we might timeout.
    # Let's try to return a hardcoded solution if we can verify one.
    # For now, if search fails, we return an empty list which is invalid, so we must find one.
    # The search for k=5..20 should find one.
    raise RuntimeError("No solution found within time limit")
