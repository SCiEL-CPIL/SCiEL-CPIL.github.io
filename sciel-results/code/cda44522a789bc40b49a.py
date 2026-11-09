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

    def verify_solution(denoms):
        """Verify that sum of reciprocals equals exactly 1."""
        if len(denoms) < 1 or len(denoms) > 256:
            return False
        if len(set(denoms)) != len(denoms):
            return False
        for d in denoms:
            if d < 5 or d > 10000000:
                return False
        s = Fraction(0, 1)
        for d in denoms:
            s += Fraction(1, d)
        return s == Fraction(1, 1)

    def greedy_egyptian(target_frac, min_denom, max_terms):
        """Greedy algorithm: repeatedly pick the largest unit fraction <= remaining."""
        remaining = target_frac
        chosen = []
        used_denoms = set()
        
        for _ in range(max_terms):
            if remaining <= 0:
                break
            if time.time() > deadline:
                return None
            
            if remaining.numerator == 0:
                break
                
            d_min_from_frac = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
            d_min = max(min_denom, d_min_from_frac)
            
            found = False
            for d in range(d_min, min(d_min + 100000, 10000001)):
                if d in used_denoms:
                    continue
                if d > 10000000:
                    break
                unit = Fraction(1, d)
                if unit <= remaining:
                    terms_left = max_terms - len(chosen) - 1
                    if terms_left > 0:
                        if remaining - unit > terms_left * Fraction(1, d + 1):
                            continue
                    
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

    def search_optimized(num, den, k, min_d, path, used_set, time_budget):
        """Optimized DFS with aggressive pruning for small k."""
        if time.time() > deadline:
            return None
        
        if k == 0:
            return path if num == 0 and den == 1 else None
            
        if k == 1:
            if den % num == 0:
                d = den // num
                if d >= min_d and d <= 10000000 and d not in used_set:
                    return path + [d]
            return None

        # Bounds for d
        d_min = max(min_d, math.ceil(den / num))
        # For k terms each >= 1/d, we need k/d >= num/den, so d <= k*den/num
        d_max = int((k * den) / num)
        
        if d_max > 10000000:
            d_max = 10000000
            
        if d_min > d_max:
            return None
        
        # Additional pruning: even with smallest possible denominators, can we reach num/den?
        # Sum of 1/(d_min), 1/(d_min+1), ..., 1/(d_min+k-1) must be >= num/den
        # Use a quick float check
        max_possible_sum = 0.0
        for i in range(k):
            d_candidate = d_min + i
            if d_candidate <= 10000000:
                max_possible_sum += 1.0 / d_candidate
        
        if max_possible_sum * den < num:
            return None
        
        # Iterate d from d_min to d_max
        # Try smaller d first (more likely to work)
        for d in range(d_min, d_max + 1):
            if d in used_set:
                continue
            
            # Check: 1/d <= num/den (i.e., d >= den/num)
            if d * num < den:
                continue
            
            # Check: remaining (num/den - 1/d) must be representable by k-1 terms with denominators > d
            # num/den - 1/d = (num*d - den) / (den*d)
            n2 = num * d - den
            d2 = den * d
            
            if n2 <= 0:
                continue
            
            # Quick float check: can k-1 terms with denominators >= d+1 sum to n2/d2?
            # Max sum: 1/(d+1) + 1/(d+2) + ... + 1/(d+k-1)
            max_rem_sum = 0.0
            for i in range(k - 1):
                d_rem = d + 1 + i
                if d_rem <= 10000000:
                    max_rem_sum += 1.0 / d_rem
            
            if max_rem_sum * d2 < n2:
                continue
            
            # Min sum check: (k-1)/(d+1) >= n2/d2 must hold
            if (k - 1) * d2 < n2 * (d + 1):
                continue
            
            g = gcd(n2, d2)
            n2_reduced = n2 // g
            d2_reduced = d2 // g
            
            used_set.add(d)
            result = search_optimized(n2_reduced, d2_reduced, k - 1, d + 1, path + [d], used_set, time_budget)
            used_set.remove(d)
            
            if result is not None:
                return result
        
        return None

    # Precomputed valid solution with 7 terms
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/21 + 1/35 = 1
    # Let's verify: LCM(5,6,7,10,14,21,35) = 210
    # 210/5 + 210/6 + 210/7 + 210/10 + 210/14 + 210/21 + 210/35 = 42 + 35 + 30 + 21 + 15 + 10 + 6 = 159
    # 159/210 = 53/70 ≠ 1
    # 
    # Let me find a correct 7-term solution:
    # Known identity: 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/15 + 1/30 = 1
    # LCM(5,6,7,10,14,15,30) = 210
    # 42 + 35 + 30 + 21 + 15 + 14 + 7 = 164 ≠ 210
    #
    # Try: 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/20 + 1/42 = 1
    # LCM = 420
    # 84 + 70 + 60 + 42 + 30 + 21 + 10 = 317 ≠ 420
    #
    # Let's use a systematic approach or find a known solution
    # 
    # Actually, let's try: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + ... 
    # This might take too many terms.
    #
    # Better approach: use the identity 1/2 = 1/3 + 1/6, so 1 = 1/2 + 1/2 = 1/3 + 1/6 + 1/2
    # But 2,3 < 5.
    #
    # Let's try to construct a valid solution programmatically and hardcode it
    # For now, use a verified 10-term solution and try to find shorter ones at runtime
    
    # Verified 10-term solution:
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/20 + 1/42 + 1/70 + 1/105 + 1/210
    # LCM(5,6,7,10,14,20,42,70,105,210) = 420
    # 84 + 70 + 60 + 42 + 30 + 21 + 10 + 6 + 4 + 2 = 329 ≠ 420
    
    # Let me compute this properly:
    # 1/5 = 84/420
    # 1/6 = 70/420
    # 1/7 = 60/420
    # 1/10 = 42/420
    # 1/14 = 30/420
    # 1/20 = 21/420
    # 1/42 = 10/420
    # 1/70 = 6/420
    # 1/105 = 4/420
    # 1/210 = 2/420
    # Sum = 84+70+60+42+30+21+10+6+4+2 = 329
    # 329/420 ≠ 1
    
    # This doesn't work. Let me find a correct solution.
    # 
    # Known Egyptian fraction for 1 with min denominator 5:
    # After research, a valid 7-term solution is:
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/15 + 1/30 = 1? 
    # Let's check with LCM = 210:
    # 42 + 35 + 30 + 21 + 15 + 14 + 7 = 164 ≠ 210
    #
    # Try: 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/21 + 1/35
    # LCM = 210
    # 42 + 35 + 30 + 21 + 15 + 10 + 6 = 159 ≠ 210
    #
    # I need to find a correct solution. Let me use the greedy algorithm result or a known valid one.
    # 
    # Actually, let's just use a verified solution from the literature or compute it.
    # For now, let's keep the search but improve it.
    
    # Hardcoded candidates to try (will be verified)
    hardcoded_candidates = [
        # These need to be verified
        [5, 6, 7, 10, 14, 20, 42, 70, 105, 210],
        [5, 6, 7, 10, 14, 21, 35, 42, 70, 105],
        [5, 6, 7, 10, 15, 20, 42, 70, 105, 210],
        [5, 6, 7, 10, 14, 35, 42, 70, 105, 210],
        # Try some 7-term candidates
        [5, 6, 7, 10, 14, 15, 30],
        [5, 6, 7, 10, 14, 21, 35],
        [5, 6, 7, 10, 14, 20, 42],
        [5, 6, 7, 10, 15, 20, 30],
        [5, 6, 7, 10, 15, 21, 35],
    ]
    
    best_solution = None
    best_k = 257
    
    for candidate in hardcoded_candidates:
        if verify_solution(candidate):
            if len(candidate) < best_k:
                best_solution = candidate
                best_k = len(candidate)
    
    # If no hardcoded solution works, use greedy
    if best_solution is None:
        best_solution = greedy_egyptian(Fraction(1, 1), 5, 256)
        if best_solution is not None:
            best_k = len(best_solution)
    
    # Try to find shorter solutions with optimized DFS
    # Start from k=5 and go up (k=4 is likely impossible with min denom 5)
    for k in range(5, min(best_k, 12)):
        if time.time() > deadline:
            break
            
        # Use a time budget for each k
        t_start = time.time()
        solution = search_optimized(1, 1, k, 5, [], set(), 10.0)
        t_elapsed = time.time() - t_start
        
        if solution is not None:
            if verify_solution(solution):
                best_solution = solution
                best_k = k
                break
        else:
            # If we spent too much time, stop
            if t_elapsed > 15.0:
                break
    
    # Final fallback if still no solution
    if best_solution is None:
        # Use greedy as final fallback
        best_solution = greedy_egyptian(Fraction(1, 1), 5, 256)
    
    return best_solution
# EVOLVE-BLOCK-END
