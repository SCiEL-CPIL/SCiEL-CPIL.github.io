import math
import time

def get_divisors(n):
    divs = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return divs

def find_solution_for_L(L, deadline):
    # Get divisors >= 5
    divs = [d for d in get_divisors(L) if d >= 5]
    if not divs:
        return None, 100
    
    # Weights are L // d
    # DP: min count to reach sum s
    # Initialize with infinity
    INF = 10**9
    dp = [INF] * (L + 1)
    dp[0] = 0
    
    # To reconstruct, we can store the choice, but for speed, 
    # we might just want the count first. 
    # If we find a count < best, we can reconstruct or just return the count 
    # and then do a second pass or store parents.
    # Given the small size of L (hopefully), storing parents is feasible.
    # parent[s] = (prev_sum, divisor_used)
    parent = [None] * (L + 1)
    
    for d in divs:
        w = L // d
        # Update dp in reverse to avoid using same divisor twice
        # We only care about sums <= L
        for s in range(L - w, -1, -1):
            if dp[s] + 1 < dp[s + w]:
                dp[s + w] = dp[s] + 1
                parent[s + w] = (s, d)
        
        # Check time
        if time.time() > deadline:
            return None, 100

    if dp[L] < INF:
        # Reconstruct solution
        sol = []
        s = L
        while s > 0:
            if parent[s] is None:
                break
            prev_s, d = parent[s]
            sol.append(d)
            s = prev_s
        if sum(1/x for x in sol) == 1.0: # Floating point check, risky but good for sanity
            # Actually, better to check exact fraction or just trust DP
            pass
        return sol, dp[L]
    return None, INF

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # Known solution
    best_sol = [5, 6, 7, 8, 9, 10, 14, 15, 63, 2520]
    best_k = len(best_sol)
    
    # Candidate L values
    # LCMs of 5..m
    lcms = []
    current_lcm = 1
    for m in range(5, 30):
        current_lcm = math.lcm(current_lcm, m)
        lcms.append(current_lcm)
        if current_lcm > 200000:
            break
            
    # Also add some other candidates
    candidates = sorted(set(lcms))
    
    # Add some specific values
    candidates.extend([2520, 5040, 10080, 25200, 50400, 100800, 252000, 504000])
    
    # Remove duplicates and sort
    candidates = sorted(set(candidates))
    
    for L in candidates:
        if time.time() > deadline:
            break
            
        # Skip if L is too large for DP array in reasonable time
        # O(N*L). If L=100000, N~100, 10^7 ops. ~1s in Python? Maybe.
        # Let's cap L at 100000 for safety, or check time strictly inside.
        if L > 100000:
            # We can still try, but it's risky. 
            # The deadline check inside the loop handles it.
            pass

        sol, k = find_solution_for_L(L, deadline)
        
        if sol is not None and k < best_k:
            best_k = k
            best_sol = sol
            # If we found k=5, it's likely optimal or near optimal. 
            # We can continue to see if k=4 is possible (unlikely with distinct >=5).
            # 1/5+1/6+1/7+1/8 = 0.75... need 0.25 from 2 terms? 
            # 1/10+1/20 = 0.15. No.
            # Let's just keep searching until deadline.
            
    return best_sol

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
