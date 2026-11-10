import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    target = Fraction(5, 31)
    
    # Fallback solution (Greedy)
    def get_greedy():
        r = target
        values = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
    
    fallback = get_greedy()
    best_solution = fallback
    best_max_denom = max(fallback)
    
    # Search for better solutions with smaller max denominator M
    # We iterate M upwards. The first solution found for a given M has max denom <= M.
    # Actually, the solution found will have max denom exactly M or less.
    # Since we iterate M from low to high, the first solution found has the minimal max denom 
    # among all solutions with max denom <= current M.
    
    # Precompute suffix sums for efficiency? 
    # We can just compute inside DFS or precompute for each M.
    
    def solve_with_max_m(M):
        # Precompute suffix sums of reciprocals for 1..M
        # S[i] = sum(1/k for k in i..M)
        # We need S[min_d+1]
        S = [Fraction(0, 1)] * (M + 2)
        curr_sum = Fraction(0, 1)
        for i in range(M, 0, -1):
            curr_sum += Fraction(1, i)
            S[i] = curr_sum
        # S[M+1] = 0
        
        def dfs(rem, min_d, current_list):
            nonlocal best_solution, best_max_denom
            if time.time() > deadline:
                return None
            
            if rem == 0:
                # Found a solution
                # Check if it's better (smaller max denom)
                # The max denom is max(current_list)
                # Since we are searching for a specific M, and we iterate M upwards,
                # the first solution found for the smallest M is the best.
                # However, within a single M, we might find a solution with max < M.
                # But we return immediately, so it's the first one.
                # Is it guaranteed to be the one with smallest max?
                # No. But we iterate M upwards.
                # If we find a solution for M=10, we return it. Max denom is <= 10.
                # If we didn't find one for M=10, we try M=11.
                # So the first success is the global minimum max denom (among those < deadline).
                return current_list[:]
            
            # Pruning: Not enough sum
            if rem > S[min_d + 1]:
                return None
                
            # Determine start_d
            # 1/d <= rem => d >= 1/rem
            # d >= min_d + 1
            if rem.numerator == 0:
                return None # Should not happen if rem > 0 check passed?
                
            # d_min_val = ceil(1/rem)
            # 1/rem = rem.denominator / rem.numerator
            d_min_val = (rem.denominator + rem.numerator - 1) // rem.numerator
            
            start_d = max(min_d + 1, d_min_val)
            
            if start_d > M:
                return None
                
            for d in range(start_d, M + 1):
                new_rem = rem - Fraction(1, d)
                if new_rem < 0:
                    continue
                
                # Pruning: Can new_rem be formed by denominators > d and <= M?
                if new_rem > S[d + 1]:
                    continue
                    
                # Optional: If new_rem is very small, check if we have enough terms?
                # We don't have a hard limit on terms other than 256.
                # With M small, terms are limited.
                
                res = dfs(new_rem, d, current_list + [d])
                if res is not None:
                    return res
            return None
            
        return dfs(target, 1, [])

    # Try M from 10 to 500
    # Start from 10 because 1/6 is too big, so min denom is 7, but max denom must be at least min denom.
    # Actually, if min denom is 7, max denom is at least 7.
    for M in range(10, 501):
        if time.time() > deadline:
            break
            
        sol = solve_with_max_m(M)
        if sol is not None:
            # Valid solution found
            # Verify distinctness and sum (optional but good for safety)
            if len(set(sol)) == len(sol) and sum(Fraction(1, d) for d in sol) == target:
                return sol
            else:
                # If validation fails (shouldn't), continue
                continue
        
    # If no better solution found, return fallback
    return fallback

