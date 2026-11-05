import time
import sys
from fractions import Fraction
from math import gcd

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s limit
    
    def elapsed():
        return time.time() - start_time
    
    def time_ok():
        return elapsed() < deadline
    
    # We want the minimum number of distinct denominators in [5, 10000000]
    # such that their reciprocal sum is exactly 1.
    
    # k=1: 1/5 != 1. Impossible.
    # k=2: 1/a + 1/b = 1 => (a-1)(b-1)=1 => a=2, b=2. But d>=5. Impossible.
    # k=3: 1/a + 1/b + 1/c = 1.
    #   Max sum with d>=5 is 1/5+1/6+1/7 = 41/105 < 1. Impossible.
    # k=4: 1/a + 1/b + 1/c + 1/d = 1.
    #   Max sum with d>=5 is 1/5+1/6+1/7+1/8 = 205/840 = 41/168 < 1. Impossible.
    # k=5: 1/a + 1/b + 1/c + 1/d + 1/e = 1.
    #   Max sum with d>=5 is 1/5+1/6+1/7+1/8+1/9 = 1429/2520 < 1. Impossible.
    #   Wait, 1/5+1/6+1/7+1/8+1/9 = (504+420+360+315+280)/2520 = 1879/2520 < 1.
    #   So k=5 is impossible?
    #   Let's check k=6.
    #   Max sum with d>=5 is 1/5+1/6+1/7+1/8+1/9+1/10 = 1879/2520 + 252/2520 = 2131/2520 < 1.
    #   Still < 1.
    #   k=7: 1/5..1/11. Sum = 2131/2520 + 1/11 = (2131*11 + 2520) / 27720 = (23441 + 2520)/27720 = 25961/27720 < 1.
    #   k=8: 1/5..1/12. Sum = 25961/27720 + 1/12 = (25961*2 + 2310)/55440 = (51922+2310)/55440 = 54232/55440 < 1.
    #   k=9: 1/5..1/13. Sum = 54232/55440 + 1/13 = (54232*13 + 55440) / 720720 = (705016 + 55440)/720720 = 760456/720720 > 1.
    #   So k=9 is possible.
    
    # Let's search for k=5, 6, 7, 8, 9 using DFS with strong pruning.
    
    def dfs(rem, min_d, depth, max_depth, path):
        if not time_ok():
            return None
        if rem == 0:
            return path
        if depth == max_depth:
            return None
        
        terms_left = max_depth - depth
        
        # Pruning 1: Upper bound on rem
        # If rem > terms_left / min_d, return None.
        if rem > terms_left / min_d:
            return None
        
        # Compute d_min and d_max using integer arithmetic
        # d_min = ceil(1/rem) = ceil(rem.denominator / rem.numerator)
        # d_max = floor(terms_left / rem) = floor(terms_left * rem.denominator / rem.numerator)
        
        d_min = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
        d_max = (terms_left * rem.denominator) // rem.numerator
        
        if d_min > d_max:
            return None
        
        # Additional pruning: if even the maximum possible sum (using smallest denominators) is less than rem, prune
        # Max sum = 1/min_d + 1/(min_d+1) + ... + 1/(min_d+terms_left-1)
        # Approximate: terms_left / min_d
        # If rem > terms_left / min_d, already pruned above
        
        # Iterate d from d_min to d_max
        for d in range(d_min, d_max + 1):
            if not time_ok():
                return None
            # Check if 1/d <= rem, i.e., d >= 1/rem
            # This is guaranteed by d_min >= ceil(1/rem)
            new_rem = rem - Fraction(1, d)
            if new_rem < 0:
                break
            res = dfs(new_rem, d + 1, depth + 1, max_depth, path + [d])
            if res is not None:
                return res
        return None
    
    # Try k=5 first, then 6, etc.
    for k in range(5, 10):
        if not time_ok():
            break
        res = dfs(Fraction(1, 1), 5, 0, k, [])
        if res is not None:
            return res
    
    # If we reach here, we haven't found a solution with k=5..9.
    # Let's try a more aggressive search with k=9 using a different strategy.
    # The key insight: for k=9, we need 9 distinct denominators >= 5 summing to 1.
    # Let's try to find a known solution or use a smarter search.
    
    # Known approach: start with a small set and expand.
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 = 760456/720720 > 1
    # So we need to adjust.
    
    # Let's try a randomized restart with the DFS for k=9.
    import random
    
    best_solution = None
    for _ in range(100):  # Try up to 100 random restarts
        if not time_ok():
            break
        
        # Randomize the search order slightly
        random.seed(_ * 12345 + 67890)
        
        def dfs_random(rem, min_d, depth, max_depth, path):
            if not time_ok():
                return None
            if rem == 0:
                return path
            if depth == max_depth:
                return None
            
            terms_left = max_depth - depth
            
            if rem > terms_left / min_d:
                return None
            
            d_min = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
            d_max = (terms_left * rem.denominator) // rem.numerator
            
            if d_min > d_max:
                return None
            
            # Shuffle the range slightly to explore different branches
            candidates = list(range(d_min, d_max + 1))
            if len(candidates) > 50:
                # Keep the first few and last few, shuffle the middle
                head = candidates[:10]
                tail = candidates[-10:]
                middle = candidates[10:-10]
                random.shuffle(middle)
                candidates = head + middle + tail
            
            for d in candidates:
                if not time_ok():
                    return None
                new_rem = rem - Fraction(1, d)
                if new_rem < 0:
                    continue
                res = dfs_random(new_rem, d + 1, depth + 1, max_depth, path + [d])
                if res is not None:
                    return res
            return None
        
        res = dfs_random(Fraction(1, 1), 5, 0, 9, [])
        if res is not None:
            return res
    
    # If we still fail, try k=10, 11, etc.
    for k in range(10, 21):
        if not time_ok():
            break
        res = dfs(Fraction(1, 1), 5, 0, k, [])
        if res is not None:
            return res
    
    # If we still fail, we have a problem.
    # But we must return a valid solution.
    # Let's try a greedy construction as a last resort.
    
    def greedy_construct():
        # Start with a known base and adjust
        # Try to find 9 denominators that sum to 1
        # Use a simple iterative method
        
        # Start with [5, 6, 7, 8, 9, 10, 11, 12, 13]
        # Sum = 760456/720720, which is > 1
        # We need to increase some denominators to reduce the sum
        
        # This is complex. Let's just return a valid solution if we find one.
        return None
    
    # As a last resort, try to find ANY valid solution
    for k in range(9, 50):
        if not time_ok():
            break
        res = dfs(Fraction(1, 1), 5, 0, k, [])
        if res is not None:
            return res
    
    # If we absolutely cannot find a solution, raise an error
    raise RuntimeError("No solution found")
