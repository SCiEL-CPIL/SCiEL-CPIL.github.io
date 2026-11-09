import math
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55 seconds to be safe within 60s limit
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def search(num, den, k, min_d, path):
        """
        DFS to find k distinct denominators >= min_d such that sum(1/d_i) = num/den.
        Returns list of denominators or None.
        """
        # Check time
        if time.time() > deadline:
            return None
        
        if k == 0:
            return path if num == 0 else None
            
        if k == 1:
            # We need num/den = 1/d => d = den/num
            if num == 0:
                return None # Should not happen if k>0 and num/den > 0
            if den % num == 0:
                d = den // num
                if d >= min_d and d <= 10000000:
                    return path + [d]
            return None

        # Lower bound for d:
        # 1/d <= num/den => d >= den/num
        d_min = max(min_d, math.ceil(den / num))
        
        # Upper bound for d:
        # num/den <= k/d => d <= k * den / num
        d_max = int((k * den) / num)
        
        if d_max > 10000000:
            d_max = 10000000
            
        if d_min > d_max:
            return None

        # Pruning 1: Max possible sum with k terms starting from d_min
        # Sum_{i=0}^{k-1} 1/(d_min + i) must be >= num/den
        # If this max sum is less than target, we can't reach the target.
        # Approximate or exact? Exact is better for small k.
        max_sum = 0.0
        for i in range(k):
            max_sum += 1.0 / (d_min + i)
        if max_sum * den < num:
            return None

        # Pruning 2: Min possible sum with k terms starting from d_min
        # Sum_{i=0}^{k-1} 1/(d_min + i) is actually the MAX sum because denominators are smallest.
        # The MIN sum is with the largest possible denominators?
        # No, the denominators are chosen in increasing order.
        # The "smallest" contribution we can get from k distinct denominators >= d_min
        # is not simply 1/d_max + ... because we don't know d_max yet.
        # However, if we pick the *largest* possible denominators, the sum is smallest.
        # But we are constrained by d <= d_max.
        # A simpler lower bound check:
        # The minimum sum we can form with k distinct integers >= d_min is
        # 1/d_min + 1/(d_min+1) + ... + 1/(d_min+k-1).
        # Wait, this is the MAXIMUM sum.
        # The MINIMUM sum is achieved by picking the largest possible denominators.
        # But we don't have a fixed upper bound for the *set* of denominators other than 10^7.
        # However, we know each d_i <= d_max.
        # So the minimum sum is approx 1/d_max + 1/(d_max-1) + ... ? No, they must be distinct.
        # Actually, if max_sum < target, we prune.
        # What if min_sum > target?
        # The minimum sum is formed by the largest possible denominators.
        # The largest possible denominators are d_max, d_max-1, ..., d_max-k+1?
        # No, they must be >= d_min.
        # If we pick d_max, d_max-1, ..., d_max-k+1, the sum is minimal.
        # If this minimal sum is > num/den, then we can't form the sum (we would overshoot).
        min_sum = 0.0
        for i in range(k):
            min_sum += 1.0 / (d_max - i)
        if min_sum * den > num:
            return None

        # Iterate d from d_min to d_max
        for d in range(d_min, d_max + 1):
            # Calculate remaining fraction: num/den - 1/d
            # = (num*d - den) / (den*d)
            n2 = num * d - den
            d2 = den * d
            
            if n2 <= 0:
                continue
            
            # Reduce fraction
            g = gcd(n2, d2)
            n2_reduced = n2 // g
            d2_reduced = d2 // g
            
            # Additional Pruning:
            # The remaining fraction n2_reduced/d2_reduced must be representable
            # by k-1 unit fractions with denominators > d.
            
            result = search(n2_reduced, d2_reduced, k - 1, d + 1, path + [d])
            if result is not None:
                return result
        return None

    # First, try to get a quick valid solution using greedy to have a fallback
    def greedy_egyptian(target_frac, min_denom, max_terms):
        remaining = target_frac
        chosen = []
        used_denoms = set()
        
        for _ in range(max_terms):
            if remaining <= 0:
                break
            if time.time() > deadline:
                return chosen if remaining == 0 else None
            
            if remaining.numerator == 0:
                break
                
            # d_min from constraint: 1/d <= remaining => d >= ceil(1/remaining)
            d_min_from_frac = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
            d_min = max(min_denom, d_min_from_frac)
            
            found = False
            # Search a limited range for speed
            for d in range(d_min, min(d_min + 50000, 10000001)):
                if d in used_denoms:
                    continue
                if d > 10000000:
                    break
                unit = Fraction(1, d)
                if unit <= remaining:
                    chosen.append(d)
                    used_denoms.add(d)
                    remaining -= unit
                    min_denom = d + 1
                    found = True
                    break
            
            if not found:
                return None
        
        if remaining == 0:
            return chosen
        return None

    best_solution = greedy_egyptian(Fraction(1, 1), 5, 256)
    best_k = len(best_solution) if best_solution else 256
    
    # Now try to find a better (shorter) solution with DFS
    # We know k=4 is impossible for denominators >= 5?
    # 1/5+1/6+1/7+1/10 = 0.2+0.166+0.142+0.1 = 0.608 < 1.
    # 1/5+1/6+1/7+1/8 = 0.2+0.166+0.142+0.125 = 0.633 < 1.
    # Max sum for k=4 with min 5 is 1/5+1/6+1/7+1/8 < 1.
    # So k must be at least 5?
    # 1/5+1/6+1/7+1/8+1/9 = 0.633 + 0.111 = 0.744 < 1.
    # 1/5+1/6+1/7+1/8+1/9+1/10 = 0.744 + 0.1 = 0.844 < 1.
    # 1/5+1/6+1/7+1/8+1/9+1/10+1/11 = 0.844 + 0.09 = 0.934 < 1.
    # 1/5+1/6+1/7+1/8+1/9+1/10+1/11+1/12 = 0.934 + 0.083 = 1.017 > 1.
    # So k=8 is possible.
    
    # Try k from 5 up to best_k-1
    # Start from 5 to save time, as 4 is impossible.
    for k in range(5, best_k):
        if time.time() > deadline:
            break
            
        solution = search(1, 1, k, 5, [])
        if solution is not None:
            best_solution = solution
            best_k = k
            break  # We found a solution with minimal k, so we can stop
            
    # If no solution found with minimal k, return the best found (greedy)
    if best_solution is None:
        # Fallback to a known valid solution if greedy failed (unlikely)
        # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/20 + 1/42 + 1/70 + 1/105 + 1/210 + 1/420 is not 1.
        # Let's use the greedy result if it exists, otherwise a hardcoded one.
        # The greedy result is valid.
        if best_solution is None:
            # Hardcoded valid solution:
            # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 + 1/70 + 1/140 + 1/280 + 1/560 + 1/1120 + 1/2240 + 1/4480 + 1/8960 + 1/17920 + 1/35840 + 1/71680 + 1/143360 + 1/286720 + 1/573440 + 1/1146880 + 1/2293760 + 1/4587520 + 1/9175040
            # This is getting long.
            # Let's just return the greedy solution. It should have worked.
            # If it didn't, we have a problem.
            # But greedy_egyptian returns None if it fails.
            # Let's try a different greedy or DP fallback.
            pass

    return best_solution
# EVOLVE-BLOCK-END
