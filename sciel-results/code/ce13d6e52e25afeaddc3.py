from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s safety margin
    
    # Hardcoded 12-term solution: 1/5+1/6+1/7+1/8+1/9+1/10+1/11+1/12+1/13+1/14+1/15+1/2520 = 1
    # Let me verify: sum 1/5 to 1/15 = 1.1346... and 1/2520 ≈ 0.000397
    # Actually let me compute more carefully.
    # LCM(5..15) = 360360
    # 1/5 = 72072/360360
    # 1/6 = 60060/360360
    # 1/7 = 51480/360360
    # 1/8 = 45045/360360
    # 1/9 = 40040/360360
    # 1/10 = 36036/360360
    # 1/11 = 32760/360360
    # 1/12 = 30030/360360
    # 1/13 = 27720/360360
    # 1/14 = 25740/360360
    # 1/15 = 24024/360360
    # Sum = 72072+60060+51480+45045+40040+36036+32760+30030+27720+25740+24024 = 445007
    # 360360 - 445007 = -84647. That's negative, so sum > 1.
    # So 1/5..1/15 already exceeds 1. Need a different approach.
    
    # The seed's get_fallback uses a different method. Let me replicate it.
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
    
    # First, get the 12-term fallback quickly
    fallback = get_fallback()
    
    # Now try to find a solution with fewer terms (11, 10, ...)
    # Use a more efficient search strategy
    
    def try_find_solution(k, min_den=5, max_den=10000000):
        """Try to find k distinct denominators in [min_den, max_den] whose reciprocal sum is 1."""
        # Use the fact that we can scale: if we want sum of 1/d_i = 1,
        # we can use a common denominator approach.
        # For k terms, we need sum(1/d_i) = 1.
        # 
        # Strategy: Pick k-1 denominators, then the last one is determined.
        # 1/d_k = 1 - sum(1/d_i for i in 1..k-1)
        # d_k = 1 / (1 - sum(1/d_i))
        # 
        # We need d_k to be an integer > max(d_1..d_{k-1}) and <= max_den.
        
        # Better strategy: Use the Egyptian fraction expansion algorithm.
        # Greedy: at each step, pick the largest possible denominator (smallest 1/d)
        # such that the remaining sum can be expressed with the remaining number of terms.
        
        # Let's use a backtracking approach with strong pruning.
        
        best = [None]
        
        def backtrack(remaining, min_d, current, k_left):
            if time.time() > deadline:
                return
            if k_left == 0:
                if remaining == 0:
                    best[0] = current[:]
                return
            
            if k_left == 1:
                # Need 1/d = remaining, so d = 1/remaining
                if remaining > 0:
                    num, den = remaining.numerator, remaining.denominator
                    if den % num == 0:
                        d = den // num
                        if d >= min_d and d <= max_den and d not in current:
                            best[0] = current + [d]
                return
            
            # Pruning:
            # Max sum with k_left terms starting from min_d:
            # Each term <= 1/min_d, so max sum = k_left / min_d
            if remaining > k_left / min_d:
                return
            # Min sum with k_left terms: each term >= 1/max_den
            if remaining < k_left / max_den:
                return
            
            # Lower bound on d: 1/d <= remaining, so d >= 1/remaining
            d_low = max(min_d, int(1 / float(remaining)) + 1)
            if d_low < min_d:
                d_low = min_d
            
            # Upper bound on d:
            # After picking d, we need k_left-1 terms from d+1 to sum to remaining - 1/d
            # Max sum of k_left-1 terms from d+1: (k_left-1)/(d+1)
            # So we need remaining - 1/d <= (k_left-1)/(d+1)
            # This gives a lower bound on d (d must be large enough)
            # For upper bound: remaining - 1/d >= (k_left-1)/max_den
            # 1/d <= remaining - (k_left-1)/max_den
            # d >= 1/(remaining - (k_left-1)/max_den)
            # This is the same as d_low roughly.
            
            # For the upper bound on d:
            # The minimum possible sum of k_left-1 terms starting from d+1 is (k_left-1)/max_den
            # So we need remaining - 1/d >= (k_left-1)/max_den
            # 1/d <= remaining - (k_left-1)/max_den
            # If remaining - (k_left-1)/max_den <= 0, no solution
            if remaining - (k_left - 1) / max_den <= 0:
                return
            d_high = int(1 / (remaining - (k_left - 1) / max_den))
            d_high = min(d_high, max_den)
            
            if d_low > d_high:
                return
            
            # Try d from d_low to d_high
            # To make this tractable, limit the range
            # Use a heuristic: try d values that are "reasonable"
            # The average term should be remaining/k_left
            # So d should be around k_left/remaining
            avg_d = k_left / float(remaining)
            
            # Try a limited range around avg_d
            search_start = max(d_low, int(avg_d * 0.5))
            search_end = min(d_high, int(avg_d * 2) + 100)
            
            # Also include d_low and d_high
            candidates = set()
            for d in range(search_start, search_end + 1):
                candidates.add(d)
            candidates.add(d_low)
            candidates.add(d_high)
            
            for d in sorted(candidates):
                if d < min_d or d > max_den:
                    continue
                if d in current:
                    continue
                term = Fraction(1, d)
                if term >= remaining:
                    continue
                new_remaining = remaining - term
                if new_remaining < 0:
                    continue
                current.append(d)
                backtrack(new_remaining, d + 1, current, k_left - 1)
                current.pop()
                if best[0] is not None:
                    return
        
        backtrack(Fraction(1, 1), min_den, [], k)
        return best[0]
    
    # Try k=11, 10, 9, ...
    for k in range(11, 4, -1):
        if time.time() > deadline - 5:
            break
        sol = try_find_solution(k)
        if sol is not None:
            # Verify
            s = sum(Fraction(1, d) for d in sol)
            if s == 1 and len(set(sol)) == len(sol) and all(5 <= d <= 10000000 for d in sol):
                return sol
    
    return fallback
