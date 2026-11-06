import math
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s/90s kill
    
    # Known 10-term solution
    best_sol = [5, 6, 7, 8, 9, 10, 14, 15, 63, 2520]
    best_k = 10
    
    # Verify known solution
    total = sum(Fraction(1, d) for d in best_sol)
    if total != 1:
        # Fallback hardcoded
        best_sol = [5, 6, 7, 8, 9, 10, 14, 15, 63, 2520]
        best_k = 10
    
    # Try to find a solution with k < best_k using LCM-based DP
    # For a given L (lcm of denominators), each denominator d divides L,
    # weight w = L//d, and we need subset of weights summing to L.
    
    # Candidate L values: LCMs of small sets, or products that give many divisors
    candidate_Ls = []
    
    # LCMs of subsets of small integers
    small_nums = list(range(5, 21))
    # Add some structured LCMs
    for r in range(4, 12):
        l = 1
        for m in range(5, 5 + r):
            l = math.lcm(l, m)
            if l > 200000:
                break
        if l >= 5 and l <= 200000:
            candidate_Ls.append(l)
    
    # Also try specific values
    for v in [2520, 5040, 10080, 25200, 50400, 100800, 201600]:
        if v <= 200000:
            candidate_Ls.append(v)
    
    candidate_Ls = sorted(set(candidate_Ls))
    
    for L in candidate_Ls:
        if time.time() > deadline:
            break
        
        # Get all divisors of L that are >= 5
        divs = []
        for i in range(1, int(math.isqrt(L)) + 1):
            if L % i == 0:
                if i >= 5:
                    divs.append(i)
                other = L // i
                if other != i and other >= 5:
                    divs.append(other)
        
        if not divs:
            continue
        
        # Convert to weights
        weights = [L // d for d in divs]
        
        # DP: dp[s] = min count to reach sum s
        INF = 10**9
        dp = [INF] * (L + 1)
        dp[0] = 0
        parent = [None] * (L + 1)
        
        # Sort weights to help find solutions faster
        # Process each weight (each divisor can be used at most once)
        for idx, w in enumerate(weights):
            if w > L:
                continue
            # Reverse update for 0/1 knapsack
            for s in range(L - w, -1, -1):
                if dp[s] + 1 < dp[s + w]:
                    dp[s + w] = dp[s] + 1
                    parent[s + w] = (s, idx)
            
            if time.time() > deadline:
                break
        
        if dp[L] < INF and dp[L] < best_k:
            # Reconstruct
            sol_indices = []
            s = L
            while s > 0:
                if parent[s] is None:
                    break
                prev_s, idx = parent[s]
                sol_indices.append(idx)
                s = prev_s
            
            if len(sol_indices) == dp[L]:
                sol = [divs[i] for i in sol_indices]
                # Verify
                total = sum(Fraction(1, d) for d in sol)
                if total == 1 and len(set(sol)) == len(sol) and all(5 <= d <= 10000000 for d in sol):
                    best_k = dp[L]
                    best_sol = sol
                    break  # Can't do better than this L's minimum
    
    # Final verification
    total = sum(Fraction(1, d) for d in best_sol)
    if total != 1 or len(set(best_sol)) != len(best_sol) or not all(5 <= d <= 10000000 for d in best_sol):
        best_sol = [5, 6, 7, 8, 9, 10, 14, 15, 63, 2520]
        best_k = 10
    
    return best_sol
