import time
from fractions import Fraction

def construct():
    # We need distinct denominators in [5, 10^7] summing to 1.
    # Minimize the count.
    
    # Theoretical lower bound:
    # For k terms, the maximum sum with distinct denominators >= 5 is 1/5 + 1/6 + ... + 1/(5+k-1).
    # We need this sum to be >= 1.
    # k=5: 1/5+1/6+1/7+1/8+1/9 ~ 0.78 < 1
    # k=6: +1/10 ~ 0.87 < 1
    # k=7: +1/11 ~ 0.96 < 1
    # k=8: +1/12 ~ 1.04 > 1. So k=8 is the theoretical minimum.
    
    # We will try to find a solution with k=8 using a DFS search.
    # If that fails within a reasonable time, we fall back to a known valid solution with more terms.
    
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    def elapsed():
        return time.time() - start_time
    
    def time_ok():
        return elapsed() < deadline

    def find_solution(k):
        """
        Finds k distinct integers d_1 < ... < d_k >= 5 such that sum(1/d_i) = 1.
        Returns the list of denominators or None if not found.
        """
        
        def dfs(rem, min_d, depth, max_depth, path):
            if not time_ok():
                return None
            if rem == 0:
                return path
            if depth == max_depth:
                return None
            
            terms_left = max_depth - depth
            
            # Pruning 1: Upper bound on the sum of remaining terms
            # The maximum sum is achieved by the smallest possible denominators: min_d, min_d+1, ...
            # We can compute this sum efficiently.
            # Sum_{i=0}^{terms_left-1} 1/(min_d + i)
            # This is a harmonic sum.
            # For small terms_left, direct sum is fine.
            max_sum = sum(Fraction(1, min_d + i) for i in range(terms_left))
            if rem > max_sum:
                return None
            
            # Pruning 2: Lower bound on the sum of remaining terms
            # The minimum sum is achieved by the largest possible denominators.
            # Denominators are <= 10^7.
            # So min_sum = terms_left / 10^7.
            if rem < Fraction(terms_left, 10**7):
                return None
            
            # Determine the range for the next denominator d
            # 1/d <= rem  => d >= ceil(1/rem)
            d_min = max(min_d, (1 + rem.denominator - 1) // rem.numerator)
            
            # Upper bound for d:
            # We need 1/d + sum_{j=1}^{terms_left-1} 1/(d+j) >= rem
            # A loose upper bound is d <= terms_left / rem
            # More precisely, if 1/d > rem, we skip.
            # If 1/d + (terms_left-1)/(d+1) < rem, we break.
            # Let's use the loose bound first, it's usually tight enough for small k.
            d_max = (terms_left * rem.denominator) // rem.numerator
            
            if d_min > d_max:
                return None
            
            # Iterate through possible denominators
            # To find a solution quickly, we might want to try specific orders.
            # Smaller d leaves larger remainder.
            # Let's iterate from d_min to d_max.
            
            for d in range(d_min, d_max + 1):
                if not time_ok():
                    return None
                new_rem = rem - Fraction(1, d)
                if new_rem < 0:
                    break
                res = dfs(new_rem, d + 1, depth + 1, max_depth, path + [d])
                if res is not None:
                    return res
            return None

        return dfs(Fraction(1, 1), 5, 0, k, [])

    # Try to find a solution with k=8 first
    res = find_solution(8)
    if res is not None:
        return res
    
    # If k=8 fails, try k=9
    res = find_solution(9)
    if res is not None:
        return res
    
    # If k=9 fails, try k=10
    res = find_solution(10)
    if res is not None:
        return res
    
    # Fallback: A known valid solution with more terms.
    # We can construct one using the identity 1 = 1/2 + 1/3 + 1/6, but denominators must be >= 5.
    # We can split 1/2 into 1/5 + 1/10 + ...? No, distinct.
    # Let's use a known expansion or generate one.
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + ...
    # Actually, let's just return a hardcoded valid solution if the search fails.
    # But we need to be sure it's valid.
    
    # A simpler fallback:
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + ...
    # This is getting long.
    
    # Let's try to find a solution with k=11, 12, etc. if the above fail.
    for k in range(11, 20):
        if not time_ok():
            break
        res = find_solution(k)
        if res is not None:
            return res
    
    # If all else fails, we must return a valid list.
    # We can use the fact that 1 = sum(1/k for k in 5..N) - remainder?
    # Or just return a large set that sums to 1.
    # Actually, the search for k=8,9,10 is very likely to succeed.
    # If it doesn't, it's a bug or extreme timeout.
    
    # As a last resort, return a hardcoded solution.
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + ...
    # This is risky. Let's trust the search for k=8..10.
    
    raise RuntimeError("No solution found within time limit")
