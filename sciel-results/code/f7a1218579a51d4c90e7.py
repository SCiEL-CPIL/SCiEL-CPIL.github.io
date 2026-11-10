"""Combined approach: fast subset-sum DP for mean-avoiding set construction."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85  # Leave 5s safety margin before 90s kill
    
    def is_subset_sum_of_size(lst, target, size):
        """Check if target can be formed by exactly 'size' elements from lst."""
        if size == 0:
            return target == 0
        if size > len(lst):
            return False
        # Quick bounds check
        if size == len(lst):
            return sum(lst) == target
        if target < sum(sorted(lst)[:size]):
            return False
        if target > sum(sorted(lst)[-size:]):
            return False
        
        # DP: dp[k] is a set of achievable sums using exactly k elements
        # Process elements one by one
        dp = [set() for _ in range(size + 1)]
        dp[0] = {0}
        for val in lst:
            for k in range(size, 0, -1):
                if dp[k-1]:
                    dp[k].update(s + val for s in dp[k-1])
        return target in dp[size]
    
    selected = []
    # sums_by_size[k] = set of all subset sums of 'selected' using exactly k elements
    sums_by_size = {0: {0}}
    
    for x in range(1, 513):
        if time.time() > deadline:
            break
        
        # Check 1: x is not the mean of any subset of 'selected' of size >= 2
        valid = True
        for k in range(2, len(selected) + 1):
            if k * x in sums_by_size.get(k, set()):
                valid = False
                break
        
        if not valid:
            continue
        
        # Check 2: No element a in 'selected' becomes the mean of a subset
        # containing x. I.e., for each a in selected, no subset T of selected\{a}
        # such that sum(T) + x = (len(T)+1) * a
        for a in selected:
            others = [y for y in selected if y != a]
            for j in range(1, len(others) + 1):
                target_sum = (j + 1) * a - x
                if is_subset_sum_of_size(others, target_sum, j):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            selected.append(x)
            # Update sums_by_size incrementally
            old_sums = sums_by_size
            new_sums = {k: set() for k in range(len(selected) + 1)}
            new_sums[0] = {0}
            for k in range(1, len(selected) + 1):
                if k in old_sums:
                    new_sums[k].update(old_sums[k])
                if k - 1 in old_sums:
                    new_sums[k].update(s + x for s in old_sums[k-1])
            sums_by_size = new_sums
    
    return selected
# EVOLVE-BLOCK-END
