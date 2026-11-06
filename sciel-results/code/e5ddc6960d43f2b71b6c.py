import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_set = []
    
    # Strategy: Multiple restarts with randomized greedy
    # We use a precomputed dp_minus structure that is updated only when S changes.
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        S = []
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        # dp_S[k] is bitset of sums of k elements from S
        # Max k is 48. Max sum is 512*48 = 24576.
        dp_S = [0] * 50
        dp_S[0] = 1
        
        # dp_minus[a][k] is bitset of sums of k elements from S \ {a}
        dp_minus = {}
        
        # Helper to compute dp for a list
        def compute_dp(lst):
            n = len(lst)
            dp = [0] * (n + 1)
            dp[0] = 1
            for elem in lst:
                for k in range(n - 1, -1, -1):
                    if dp[k]:
                        dp[k+1] |= (dp[k] << elem)
            return dp

        # Initialize dp_minus is empty since S is empty
        
        while len(S) < 48:
            if time.time() > deadline:
                break
            
            n = len(S)
            found = False
            
            # Try to find a valid candidate
            for x in candidates:
                if time.time() > deadline:
                    break
                
                if x in S:
                    continue
                
                valid = True
                
                # Check 1: Is x the mean of a subset of S?
                # Sum = k * x for k >= 2
                for k in range(2, n + 1):
                    target = k * x
                    if target > 512 * 48:
                        break
                    if (dp_S[k] >> target) & 1:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check 2: Is any a in S the mean of a subset of S U {x} \ {a}?
                # Subset T contains x. T = {x} U U, U subset of S \ {a}.
                # |T| = |U| + 1.
                # Sum(T) = x + Sum(U) = (|U| + 1) * a
                # Sum(U) = (|U| + 1) * a - x
                for a in S:
                    # Get dp for S \ {a}
                    if a not in dp_minus:
                         # This should not happen if we maintain dp_minus correctly
                         # But for the first few steps, we might need to compute
                         # Actually, we recompute dp_minus after every addition.
                         # So dp_minus should contain all a in S.
                         temp_S = [y for y in S if y != a]
                         dp_minus[a] = compute_dp(temp_S)
                    
                    dp_a = dp_minus[a]
                    # j = |U|, ranges from 1 to n-1
                    for j in range(1, n):
                        target = (j + 1) * a - x
                        if target < 0:
                            continue
                        if target > 512 * 48:
                            break
                        if (dp_a[j] >> target) & 1:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Add x to S
                    S.append(x)
                    
                    # Update dp_S
                    for k in range(n, -1, -1):
                        if dp_S[k]:
                            dp_S[k+1] |= (dp_S[k] << x)
                    
                    # Recompute dp_minus for all elements in S
                    # This is O(N^2 * DP) but only happens N times total
                    new_dp_minus = {}
                    for a in S:
                        temp_S = [y for y in S if y != a]
                        new_dp_minus[a] = compute_dp(temp_S)
                    dp_minus = new_dp_minus
                    
                    found = True
                    break
            
            if not found:
                break
        
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) == 48:
                break
    
    return best_set
