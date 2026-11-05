import time

def construct():
    deadline = time.time() + 80
    target = 200
    universe = list(range(1, 101))
    
    # Pre-compute weights w_i: number of subsets of [1,100]\{i} summing to 200-i
    weights = [0] * 101
    for i in range(1, 101):
        # Count subsets of [1,100] excluding i that sum to target - i
        # If target - i < 0, weight is 0
        if target - i < 0:
            weights[i] = 0
            continue
        
        # DP to count subsets
        # dp[k] = number of subsets of current processed elements summing to k
        # We process elements 1..100 except i
        dp = [0] * (target + 1)
        dp[0] = 1
        for j in range(1, 101):
            if j == i:
                continue
            # Update dp in reverse to avoid using j multiple times
            # We only care about sums up to target - i
            max_sum_needed = target - i
            # Optimization: only update up to max_sum_needed
            for k in range(max_sum_needed, j - 1, -1):
                dp[k] += dp[k - j]
        weights[i] = dp[target - i]
        
    # Greedy Hitting Set
    S = set(universe)
    
    def find_bad_subset(S_set):
        """
        Returns a list of elements in S_set that sum to target, or None if no such subset exists.
        """
        mask = (1 << (target + 1)) - 1
        reachable = 1  # bit 0 set
        # choice[s] = (prev_sum, element)
        # We only need to store choices for sums that are newly reached
        choice = {}
        
        # Sort S for deterministic behavior? Or just iterate.
        # Iterating in a specific order might affect which subset is found, but any valid subset works.
        # However, for the backtracking to work correctly with the 'choice' map, 
        # we must ensure that if a sum is reachable via multiple paths, we pick one.
        # The bitset OR operation merges paths. 
        # To reconstruct, we need to know which element added to which previous sum.
        # If sum s is reached by x from s-x, and also by y from s-y, we need to pick one.
        # The standard way is to record the choice only if the sum was NOT reachable before.
        
        for x in S_set:
            new_sums = (reachable << x) & mask
            # Only consider sums that are newly added
            # new_sums & ~reachable
            to_add = new_sums & ~reachable
            if to_add:
                # Iterate through bits in to_add
                # We need to map each new sum s to (s-x, x)
                # Since x is fixed for this iteration, for any s in to_add, s-x must have been in reachable.
                # We can just record the choice for all s in to_add.
                # But wait, if multiple x's could lead to the same s in different iterations,
                # the first one that reaches s will be recorded.
                # This is fine.
                
                # Efficiently iterate bits in to_add
                # Python ints don't have a direct "iterate bits" that is very fast for large ints, 
                # but target is 200, so int is small.
                # We can loop k from 0 to target
                for k in range(target + 1):
                    if (to_add >> k) & 1:
                        choice[k] = (k - x, x)
            
            reachable |= new_sums
            if (reachable >> target) & 1:
                # Found a subset summing to target
                # Backtrack
                subset = []
                current_sum = target
                while current_sum > 0:
                    if current_sum not in choice:
                        # This should not happen if we recorded correctly
                        break
                    prev_sum, elem = choice[current_sum]
                    subset.append(elem)
                    current_sum = prev_sum
                return subset
        return None

    while time.time() < deadline:
        bad_set = find_bad_subset(S)
        if bad_set is None:
            break
        
        # Select element with highest weight from bad_set
        # If weights are tied, pick arbitrarily (e.g., largest value)
        # We want to remove the element that hits the most "potential" conflicts.
        # Highest weight is a good proxy.
        
        # Sort bad_set by weight descending
        # If weights are equal, we might want to remove the one that is "least useful" for other sums?
        # Or just remove the one with max weight.
        
        # Let's pick the element with max weight.
        # If there's a tie, pick the one that appears in the most subsets of S summing to target?
        # We don't have that info.
        # Let's just pick max weight.
        
        # To be deterministic, sort by (-weight, -value)
        candidates = sorted(bad_set, key=lambda x: (-weights[x], -x))
        x_to_remove = candidates[0]
        
        S.remove(x_to_remove)
        
    # Convert to list and return
    # The problem asks for a subset. A list is fine.
    return list(S)
