"""Optimized program for the suite task; uses search to minimize number of divisors."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    divisors.sort(reverse=True)
    
    # Precompute for fast access
    divisor_set = set(divisors)
    
    def valid_solution(chosen):
        if not chosen:
            return False
        if len(chosen) > 256:
            return False
        if len(set(chosen)) != len(chosen):
            return False
        if any(d not in divisor_set for d in chosen):
            return False
        return sum(chosen) == target
    
    best_solution = None
    best_len = float('inf')
    
    # Strategy 1: Greedy from largest, then try to improve by swapping
    def greedy_solve():
        rem = target
        chosen = []
        for d in divisors:
            if d <= rem:
                chosen.append(d)
                rem -= d
                if rem == 0:
                    break
        if rem == 0:
            return chosen
        return None
    
    g = greedy_solve()
    if g:
        best_solution = g
        best_len = len(g)
    
    # Check if target is a divisor
    if target in divisor_set:
        return [target]
    
    # Try pairs
    for i in range(len(divisors)):
        if time.time() > deadline:
            break
        d1 = divisors[i]
        if d1 > target:
            continue
        rem = target - d1
        if rem in divisor_set and rem != d1:
            candidate = [d1, rem]
            if valid_solution(candidate):
                if len(candidate) < best_len:
                    best_solution = candidate
                    best_len = len(candidate)
                return candidate
    
    # Try triples using a hash set for the third
    # For efficiency, iterate over pairs of larger divisors and check if remainder is a divisor
    if time.time() < deadline:
        div_set = set(divisors)
        for i in range(len(divisors)):
            if time.time() > deadline:
                break
            d1 = divisors[i]
            if d1 > target:
                continue
            rem1 = target - d1
            if rem1 <= 0:
                continue
            # Need two distinct divisors d2, d3 such that d2 + d3 = rem1, d2 != d3, d2 != d1, d3 != d1
            for j in range(i + 1, len(divisors)):
                if time.time() > deadline:
                    break
                d2 = divisors[j]
                if d2 >= rem1:
                    break  # since sorted desc, further j will be smaller but d2 already too big? No, d2 < d1 but still could be >= rem1
                if d2 >= rem1:
                    continue
                rem2 = rem1 - d2
                if rem2 <= 0:
                    continue
                if rem2 in div_set and rem2 != d1 and rem2 != d2:
                    candidate = [d1, d2, rem2]
                    if valid_solution(candidate):
                        if len(candidate) < best_len:
                            best_solution = candidate
                            best_len = len(candidate)
                        return candidate
    
    # Strategy 3: Randomized local search / simulated annealing-like approach
    if time.time() < deadline:
        # Start from greedy solution or random feasible solution
        current = greedy_solve()
        if current is None:
            # Construct a feasible solution
            current = []
            rem = target
            for d in divisors:
                if d <= rem:
                    current.append(d)
                    rem -= d
                    if rem == 0:
                        break
            if rem != 0:
                # Fallback: just use all divisors that sum to target via simple greedy
                # This should work for practical numbers
                current = []
                rem = target
                for d in divisors:
                    if d <= rem:
                        current.append(d)
                        rem -= d
                        if rem == 0:
                            break
                if rem != 0:
                    # Last resort
                    return best_solution if best_solution else divisors[:1]
        
        if current is not None and valid_solution(current):
            if len(current) < best_len:
                best_solution = current
                best_len = len(current)
        
        # Local search: try to reduce number of terms
        # Pick a random subset of current, replace with smaller count of divisors that sum to same value
        rng = random.Random(42)
        for iteration in range(200000):
            if time.time() > deadline:
                break
            if best_len <= 3:
                break
            
            cur = best_solution if best_solution else current
            if cur is None:
                break
            
            cur_set = set(cur)
            cur_sum = sum(cur)
            cur_len = len(cur)
            
            # Try to replace k elements with m < k elements
            # Pick a random subset S of cur to potentially remove
            for k in range(2, min(6, cur_len)):
                if time.time() > deadline:
                    break
                subset = rng.sample(cur, k)
                subset_sum = sum(subset)
                subset_set = set(subset)
                remaining = [x for x in cur if x not in subset_set]
                
                # Try to find m < k divisors (not in remaining, not in subset) that sum to subset_sum
                available = [d for d in divisors if d not in remaining and d not in subset_set]
                
                # Try m = 1
                if subset_sum in available:
                    new_cur = remaining + [subset_sum]
                    if valid_solution(new_cur) and len(new_cur) < best_len:
                        best_solution = new_cur
                        best_len = len(new_cur)
                        break
                # Try m = 2
                if k > 2:
                    for a in available:
                        if a > subset_sum:
                            continue
                        b = subset_sum - a
                        if b in available and b != a:
                            new_cur = remaining + [a, b]
                            if valid_solution(new_cur) and len(new_cur) < best_len:
                                best_solution = new_cur
                                best_len = len(new_cur)
                                break
                    if best_len < cur_len:
                        break
    
    # Final validation
    if best_solution is None:
        # Fallback to greedy
        rem = target
        chosen = []
        for d in divisors:
            if d <= rem:
                chosen.append(d)
                rem -= d
                if rem == 0:
                    break
        best_solution = chosen
    
    assert valid_solution(best_solution), f"Invalid solution: sum={sum(best_solution)}, len={len(best_solution)}"
    return best_solution
# EVOLVE-BLOCK-END
