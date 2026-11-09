"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80
    
    def can_form_mean_dp(target, elements, k):
        """Check if any subset of exactly k elements from 'elements' sums to target. Returns bool."""
        if k == 0:
            return target == 0
        if k > len(elements):
            return False
        if target < 0 or target > sum(elements[-k:]):  # max possible sum of k largest
            return False
        if target < sum(elements[:k]):  # min possible sum of k smallest
            return False
        # Bitset DP
        dp = [0] * (k + 1)
        dp[0] = 1
        mask = (1 << (target + 1)) - 1
        for x in elements:
            if x > target:
                continue
            for j in range(min(k, len(elements)), 0, -1):
                dp[j] |= (dp[j-1] << x) & mask
        return (dp[k] >> target) & 1
    
    def check_new_element(x, S, S_set):
        """Check if adding x to S violates the condition. S is sorted list, S_set is set."""
        n = len(S)
        # Part 1: Is x the mean of any subset of S (size >= 2)?
        for k in range(2, n + 1):
            target = k * x
            # Quick bounds: min sum of k elements, max sum of k elements
            min_sum = sum(S[:k]) if k <= n else float('inf')
            max_sum = sum(S[-k:]) if k <= n else 0
            if target < min_sum or target > max_sum:
                continue
            if k == 2:
                # Check if any pair in S sums to 2*x
                for i in range(n):
                    complement = 2 * x - S[i]
                    if complement != S[i] and complement in S_set:
                        return False
            else:
                if can_form_mean_dp(target, S, k):
                    return False
        # Part 2: For each a in S, is a the mean of a subset of S∪{x} that includes x?
        for a in S:
            others_no_a = [y for y in S if y != a]
            others_set = S_set - {a}
            m = len(others_no_a)
            for u in range(1, m + 1):
                target_sum = (1 + u) * a - x
                if target_sum < 0:
                    continue
                # U is subset of others_no_a of size u, sum = target_sum
                min_sum = sum(others_no_a[:u]) if u <= m else float('inf')
                max_sum = sum(others_no_a[-u:]) if u <= m else 0
                if target_sum < min_sum or target_sum > max_sum:
                    continue
                if u == 1:
                    # Just check if target_sum is in others_no_a
                    if target_sum in others_set:
                        return False
                else:
                    if can_form_mean_dp(target_sum, others_no_a, u):
                        return False
        return True
    
    selected = []
    selected_set = set()
    for x in range(1, 513):
        if time.time() > deadline:
            break
        if check_new_element(x, selected, selected_set):
            selected.append(x)
            selected_set.add(x)
        if len(selected) >= 48:
            break
    
    return selected
# EVOLVE-BLOCK-END
