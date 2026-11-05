import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds to be safe
    
    # We want to minimize the count. We try k = 3, 4, ...
    # Max denominator is 10,000,000
    
    def dfs(rem_sum, k, min_d, current_denoms):
        if time.time() > deadline:
            return None
        
        if k == 0:
            if rem_sum == 0:
                return current_denoms
            return None
        
        if rem_sum == 0:
            return None # Should not happen if k>0 and we only add positive terms
            
        # Lower bound for next denominator d
        # 1/d <= rem_sum => d >= 1/rem_sum
        # Also d >= min_d
        d_min = max(min_d, (1 // rem_sum) + 1 if rem_sum < 1 else 1)
        # Careful with Fraction division
        if rem_sum > 0:
            d_min_candidate = (1 / rem_sum)
            d_min = max(min_d, int(d_min_candidate) + 1)
        else:
            return None
            
        # Upper bound for next denominator d
        # We need to be able to form rem_sum - 1/d with k-1 terms, all > d.
        # Max sum of k-1 terms with denominators > d is (k-1)/(d+1).
        # So rem_sum - 1/d <= (k-1)/(d+1)
        # This implies d cannot be too large.
        # Approximation: rem_sum <= (k-1)/d => d <= (k-1)/rem_sum
        if rem_sum > 0:
            d_max = int((k - 1) / rem_sum) + 1
            # Refine upper bound with the exact inequality if needed, 
            # but the approximation is usually tight enough for pruning.
            # Let's ensure d_max is at least d_min
            if d_max < d_min:
                return None
        else:
            return None

        for d in range(d_min, d_max + 1):
            if d > 10000000:
                break
            if d in current_denoms:
                continue
                
            next_rem = rem_sum - Fraction(1, d)
            if next_rem < 0:
                break # Since d is increasing, 1/d is decreasing, so next_rem will increase? 
                # Wait. If d increases, 1/d decreases. 
                # rem_sum - 1/d increases.
                # So if next_rem < 0, it means 1/d > rem_sum.
                # But we set d_min such that 1/d <= rem_sum.
                # So next_rem should be >= 0.
                # If next_rem == 0 and k > 1, we can't fill remaining terms with 0.
                if next_rem == 0:
                    if k == 1:
                        return current_denoms + [d]
                    else:
                        continue # Can't represent 0 with positive terms
                
                # Recursive step
                res = dfs(next_rem, k - 1, d + 1, current_denoms + [d])
                if res:
                    return res
        
        return None

    # Try increasing k
    for k in range(3, 257):
        result = dfs(Fraction(1, 1), k, 5, [])
        if result:
            return result
        
    # Fallback to a known valid solution if search fails or times out
    # A simple solution: 1/5 + 1/6 + 1/30 is not 1.
    # Let's use a pre-computed valid set or the DP result from the seed.
    # The seed found a solution with target 2520.
    # Let's just return a valid set if we find one quickly, else raise or return best effort.
    # For safety, we can return a known valid set of, say, 10 terms if DFS fails.
    # Example: 1/5 + 1/6 + 1/7 + 1/42 + ...
    # Actually, the seed program returns a valid list. We can reuse that logic as a fallback.
    
    # Fallback DP from seed
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
            
    raise RuntimeError("no representation found")
