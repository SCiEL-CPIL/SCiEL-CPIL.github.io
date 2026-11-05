"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # Generate candidates
    # Prioritize smooth numbers (2^i * 3^j) as they have fewer additive relations
    # Then add other numbers, potentially sorted by value or some heuristic
    candidates = list(range(1, 513))
    
    # Heuristic: sort by "smoothness" score
    # Lower score = smoother = better candidate earlier
    def smooth_score(x):
        # Count factors of 2 and 3
        score = 0
        temp = x
        while temp % 2 == 0:
            temp //= 2
            score += 1
        while temp % 3 == 0:
            temp //= 3
            score += 1
        # If temp > 1, it has other factors. Penalize based on size of remaining part
        if temp > 1:
            score += temp
        return score
        
    candidates.sort(key=smooth_score)

    selected = []
    selected_set = set()

    def is_valid_addition(x, current_set):
        """
        Check if adding x to current_set maintains the property.
        Property: No element is the mean of a subset of >= 2 OTHER elements.
        """
        # 1. Check if x becomes the mean of a subset of current_set
        #    sum(S) = k * x for some S subset of current_set, |S| >= 2
        if len(current_set) >= 2:
            # Compute subset sums of current_set with cardinality
            # dp[k] is a bitset of sums achievable with exactly k elements
            n = len(current_set)
            max_sum = sum(current_set)
            # We only care about sums up to max_sum
            # dp[k] is an integer where bit s is 1 if sum s is possible with k elements
            dp = [0] * (n + 1)
            dp[0] = 1  # sum 0 with 0 elements
            
            for elem in current_set:
                # Iterate backwards to avoid using an element multiple times
                for k in range(n, 0, -1):
                    dp[k] |= (dp[k-1] << elem)
            
            for k in range(2, n + 1):
                target = k * x
                if target <= max_sum:
                    if (dp[k] >> target) & 1:
                        return False
                else:
                    break # Sums only increase with k, so if k*x > max_sum, larger k will too? 
                          # Not necessarily, but max_sum is fixed. If k*x > max_sum, no subset of size k can sum to it.
                          # However, larger k means larger min sum? No, min sum is sum of smallest k.
                          # But max_sum is the upper bound. So if k*x > max_sum, no subset of size k sums to k*x.
                          # We can break if k*x > max_sum.

        # 2. Check if any a in current_set becomes the mean of a subset involving x
        #    For each a in current_set:
        #    Does there exist S subset of (current_set \ {a}) such that sum(S) + x = k * a?
        #    => sum(S) = k * a - x
        #    Here |S| >= 1 (so total subset size |S|+1 >= 2).
        #    k = |S| + 1. So k >= 2.
        
        for a in current_set:
            others = [y for y in current_set if y != a]
            if len(others) == 0:
                continue
            
            m = len(others)
            max_others_sum = sum(others)
            
            # Compute subset sums of 'others'
            dp2 = [0] * (m + 1)
            dp2[0] = 1
            
            for elem in others:
                for k in range(m, 0, -1):
                    dp2[k] |= (dp2[k-1] << elem)
            
            # We need to check if k*a - x is a subset sum of 'others' for some k >= 2
            # where the subset size is k-1.
            # Let j = k-1. j ranges from 1 to m.
            # Target sum = (j+1)*a - x
            for j in range(1, m + 1):
                target = (j + 1) * a - x
                if 0 <= target <= max_others_sum:
                    if (dp2[j] >> target) & 1:
                        return False
                elif target > max_others_sum:
                    # Since a > 0, (j+1)*a increases with j. 
                    # If target > max_others_sum, larger j will yield larger targets.
                    break

        return True

    # Greedy selection
    for x in candidates:
        if time.time() > deadline:
            break
        
        if x in selected_set:
            continue
        
        if is_valid_addition(x, selected):
            selected.append(x)
            selected_set.add(x)
            
            if len(selected) >= 48:
                break

    return selected

# EVOLVE-BLOCK-END
