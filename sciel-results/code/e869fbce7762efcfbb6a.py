from fractions import Fraction
import time

def construct():
    deadline = time.time() + 85
    
    # Try to find minimal N via optimized backtracking
    # Start from small N and increase
    
    def max_sum(start, k):
        """Max sum of k distinct unit fractions with denominators >= start"""
        s = Fraction(0, 1)
        for i in range(k):
            s += Fraction(1, start + i)
        return s
    
    def search(remaining, min_d, count, path):
        if time.time() > deadline:
            return None
            
        if count == 0:
            if remaining == 0:
                return path[:]
            return None
        
        if count == 1:
            if remaining.numerator == 1:
                d = remaining.denominator
                if d >= min_d and d not in path and d <= 10000000:
                    return path + [d]
            return None
        
        if remaining <= 0:
            return None
        
        # Lower bound for d
        # 1/d <= remaining => d >= ceil(1/remaining)
        # 1/remaining = remaining.denominator / remaining.numerator
        low_d = max(min_d, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        
        d = low_d
        while d <= 10000000:
            if time.time() > deadline:
                return None
            
            # 1/d must be <= remaining
            if remaining.numerator * d < remaining.denominator:
                d += 1
                continue
            
            rest = remaining - Fraction(1, d)
            
            if rest < 0:
                d += 1
                continue
            
            # Check if rest can be formed by count-1 terms starting from d+1
            if rest > max_sum(d + 1, count - 1):
                break
            
            res = search(rest, d + 1, count - 1, path + [d])
            if res is not None:
                return res
            
            d += 1
        
        return None
    
    # Search for minimal N
    for N in range(4, 13):  # Start from 4, go up to 12
        if time.time() > deadline:
            break
        sol = search(Fraction(1, 1), 5, N, [])
        if sol is not None:
            return sol
    
    # Fallback: known valid solution (verified)
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20
    # This sums to more than 1, so we need a different fallback.
    
    # Use a DP approach with a fixed target as ultimate fallback
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    raise RuntimeError("No solution found")
