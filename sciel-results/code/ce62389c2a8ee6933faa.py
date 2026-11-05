import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Baseline solution from the seed program (11 terms)
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/21 + 1/30 + 1/35 + 1/42 + 1/70 + 1/105 ? No, seed used DP.
    # Let's just generate a known valid 11-term solution or keep the seed logic as fallback.
    # Seed logic:
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
            # This is the seed's result. We assume it's valid.
            # We will try to beat it.
            pass

    # We want to find a solution with fewer terms.
    # Min possible terms is 4 (since 1,2,3 are impossible with distinct den >= 5).
    # We will use DFS to find the shortest sequence.
    
    def dfs(rem_num, rem_den, min_den, depth, max_depth, path):
        # Check time
        if time.time() > deadline:
            return None
        
        if depth == max_depth:
            if rem_num == 0:
                return path[:]
            return None
        
        if rem_num == 0:
            return path[:]
        
        # Pruning: if we can't fill the remainder with 'depth' terms, return None
        # Max sum of 'depth' distinct unit fractions starting from min_den
        # This is an upper bound check.
        # 1/min_den + 1/(min_den+1) + ... + 1/(min_den+depth-1)
        # If this sum < rem_num/rem_den, it's impossible.
        max_possible_sum = 0.0
        for i in range(depth):
            max_possible_sum += 1.0 / (min_den + i)
        if max_possible_sum < rem_num / rem_den:
            return None
            
        # Lower bound for next denominator d
        # 1/d <= rem_num/rem_den  => d >= rem_den/rem_num
        # Also d >= min_den
        # Also to allow remaining terms, d cannot be too large.
        # If we pick d, remaining rem' = rem - 1/d.
        # We need rem' to be sum of depth-1 terms >= d+1.
        # A simple bound: d must be small enough such that 1/d + 1/(d+1) + ... >= rem.
        # But the loop below handles the upper bound implicitly by checking if rem becomes negative or if we run out of terms.
        
        lower_d = max(min_den, (rem_den + rem_num - 1) // rem_num)
        
        # Upper bound for d:
        # We need to leave enough for the remaining 'depth - 1' terms.
        # The largest possible sum of the remaining 'depth - 1' terms is when they are as small as possible (d+1, d+2, ...).
        # So we need: rem - 1/d <= 1/(d+1) + 1/(d+2) + ... + 1/(d+depth-1)
        # This is a tight bound. Let's use a looser one for speed:
        # 1/d < rem  => d > 1/rem
        # Also, the sum of the remaining depth-1 terms must be at least rem - 1/d.
        # If rem - 1/d is very small, d can be large.
        # However, if d is too large, 1/d is small, and we might not be able to sum up to rem with the remaining terms if they are forced to be distinct and larger?
        # Actually, if d is large, 1/d is small. The remaining sum is close to rem.
        # The max sum of remaining terms is ~ (depth-1)/d.
        # So we need rem - 1/d <= (depth-1)/d  => rem <= depth/d  => d <= depth/rem.
        # So d is bounded by [lower_d, floor(depth / (rem_num/rem_den))] = [lower_d, floor(depth * rem_den / rem_num)]
        
        upper_d = int((depth * rem_den) / rem_num)
        
        if lower_d > upper_d:
            return None
            
        for d in range(lower_d, upper_d + 1):
            # Calculate new remainder
            # rem' = rem_num/rem_den - 1/d = (rem_num*d - rem_den) / (rem_den*d)
            new_num = rem_num * d - rem_den
            if new_num < 0:
                break # d is too small? No, d >= lower_d ensures new_num >= 0.
            # If new_num == 0, we have a solution if depth == 1? 
            # No, if new_num == 0, we used 'd' and the remainder is 0.
            # This is valid if this was the last term (depth was 1 in the call stack? No, depth is remaining terms including this one).
            # If new_num == 0, we have successfully formed the sum.
            # But we must ensure we don't have more terms to add (which would make sum > 1).
            # So if new_num == 0, we return path + [d] ONLY IF depth == 1?
            # No, if depth > 1 and new_num == 0, we have extra terms to place, which would make sum > 1.
            # So new_num must be 0 only if this is the last term.
            # Wait, the loop is for the NEXT term.
            # If new_num == 0, it means 1/d exactly equals the remainder.
            # This is a valid completion IF we don't need more terms.
            # But the function `dfs` is called with `depth` remaining terms.
            # If we pick `d`, we have `depth - 1` terms left.
            # If `new_num == 0`, we have 0 remainder. We can only accept this if `depth - 1 == 0`.
            # If `depth - 1 > 0`, we cannot add any more positive terms.
            
            if new_num == 0:
                if depth == 1:
                    return path + [d]
                else:
                    # We have remainder 0 but need to place more terms? Impossible.
                    continue
                    
            # Simplify fraction?
            # For exact arithmetic, we should keep it as integers.
            # new_den = rem_den * d
            
            # Recurse
            res = dfs(new_num, rem_den * d, d + 1, depth - 1, max_depth, path + [d])
            if res:
                return res
                
        return None

    # Try to find solution with k=4
    # rem = 1/1
    best_sol = None
    for k in range(4, 12): # Try 4 to 11
        if time.time() > deadline:
            break
        sol = dfs(1, 1, 5, k, k, [])
        if sol:
            best_sol = sol
            break
            
    if best_sol:
        return best_sol
    else:
        # Fallback to seed solution if DFS fails (should not happen)
        # Re-run seed logic to get the 11-term solution
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
        raise RuntimeError("no solution found")
