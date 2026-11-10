"""Few-term Egyptian fractions with denominators greater than four."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    def verify(denoms):
        if not denoms or len(denoms) > 256:
            return False
        if any(d < 5 or d > 10000000 for d in denoms):
            return False
        if len(set(denoms)) != len(denoms):
            return False
        total = sum(Fraction(1, d) for d in denoms)
        return total == 1
    
    def find_solution(max_terms, time_budget):
        """DFS to find Egyptian fraction with exactly max_terms denominators >= 5."""
        def dfs(remaining, min_denom, used, depth, max_depth, result):
            if time.time() > time_budget:
                return False
            if remaining == 0:
                if len(used) == max_depth:
                    result.extend(used)
                    return True
                return False
            if depth == max_depth:
                return False
            if remaining < 0:
                return False
            
            k = max_depth - depth
            
            # Pruning: max possible sum with k terms starting from min_denom
            max_possible = sum(Fraction(1, min_denom + i) for i in range(k))
            if remaining > max_possible:
                return False
            
            min_d = max(min_denom, math.ceil(1 / float(remaining)))
            
            if remaining > 0:
                max_d = min(200, int(1 / float(remaining)) + k + 10)
            else:
                max_d = 200
            
            for d in range(min_d, max_d + 1):
                if d < 5 or d in used:
                    continue
                new_remaining = remaining - Fraction(1, d)
                if new_remaining < 0:
                    break
                used.add(d)
                if dfs(new_remaining, d + 1, used, depth + 1, max_depth, result):
                    return True
                used.remove(d)
            return False
        
        result = []
        if dfs(Fraction(1, 1), 5, set(), 0, max_terms, result):
            return sorted(result)
        return None
    
    # Try increasing number of terms starting from 8
    best = None
    for n_terms in range(8, 17):
        if time.time() > deadline - 5:
            break
        remaining_time = deadline - 5 - time.time()
        if remaining_time <= 0:
            break
        time_budget = time.time() + remaining_time / (17 - n_terms)
        sol = find_solution(n_terms, time_budget)
        if sol is not None and verify(sol):
            best = sol
            break
    
    if best is not None:
        return best
    
    # Fallback: generate a valid solution by splitting
    # 1 = 1/5 + 4/5, 4/5 = 1/6 + 19/30, 19/30 = 1/7 + 73/210, ...
    # Simpler: use the greedy algorithm to get a valid (if not minimal) solution
    def greedy_egyptian(target, min_denom, max_terms):
        denoms = []
        remaining = target
        for i in range(max_terms):
            if time.time() > deadline - 2:
                break
            if remaining == 0:
                break
            d = max(min_denom, math.ceil(1 / float(remaining)))
            if d < 5:
                d = 5
            if d in denoms:
                d += 1
            denoms.append(d)
            remaining = remaining - Fraction(1, d)
            min_denom = d + 1
        return denoms
    
    # Try greedy with various starting points
    for max_terms in range(10, 20):
        sol = greedy_egyptian(Fraction(1, 1), 5, max_terms)
        if verify(sol):
            return sol
    
    # Last resort: a very large but valid solution
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + ...
    # This won't sum to exactly 1, so let me use a different approach
    # Just return a hardcoded valid solution found by the DFS or similar
    
    # Generate a valid solution using the identity 1 = 1/n + 1/n + ... (n times) but with distinct
    # Use: 1 = 1/5 + 1/5 + 1/5 + 1/5 + 1/5 is invalid (not distinct)
    # Use: 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50 + 1/51 + 1/52 + 1/53 + 1/54 + 1/55 + 1/56 + 1/57 + 1/58 + 1/59 + 1/60 + 1/61 + 1/62 + 1/63 + 1/64 + 1/65 + 1/66 + 1/67 + 1/68 + 1/69 + 1/70 + 1/71 + 1/72 + 1/73 + 1/74 + 1/75 + 1/76 + 1/77 + 1/78 + 1/79 + 1/80 + 1/81 + 1/82 + 1/83 + 1/84 + 1/85 + 1/86 + 1/87 + 1/88 + 1/89 + 1/90 + 1/91 + 1/92 + 1/93 + 1/94 + 1/95 + 1/96 + 1/97 + 1/98 + 1/99 + 1/100
    # This sums to more than 1, so it's not valid either.
    
    # I need a guaranteed valid solution. Let me use the DFS with a larger budget.
    sol = find_solution(16, deadline - 1)
    if sol is not None and verify(sol):
        return sol
    
    # Absolute last resort: return an empty list (will score 0 but avoid crash)
    return []

# EVOLVE-BLOCK-END
