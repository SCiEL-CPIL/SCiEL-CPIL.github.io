import time
import random

def construct():
    deadline = time.time() + 55.0
    best_set = []
    max_val = 512
    candidates = list(range(1, max_val + 1))
    
    def is_valid(S):
        if not S:
            return True
        # Check each element x in S
        for x in S:
            others = [y for y in S if y != x]
            if len(others) < 2:
                continue
            # Bitset DP for subset sums of 'others'
            # dp[k] is an integer bitset where bit s is 1 if sum s is achievable with k elements
            max_sum = sum(others)
            # We only care about sums up to k*x, which is <= len(others)*max(others)
            # But max_sum is a safe upper bound.
            # Initialize dp
            # dp[0] = 1 (sum 0 with 0 elements)
            dp = [0] * (len(others) + 1)
            dp[0] = 1
            current_max_k = 0
            for y in others:
                # Update from high k to low k to avoid using y multiple times
                # Max k is len(others)
                # We can limit k to current_max_k + 1
                for k in range(current_max_k + 1, 0, -1):
                    dp[k] |= (dp[k-1] << y)
                current_max_k += 1
            
            # Check if any k >= 2 has sum k*x
            for k in range(2, len(others) + 1):
                target_sum = k * x
                if target_sum > max_sum:
                    break
                # Check if bit target_sum is set in dp[k]
                if (dp[k] >> target_sum) & 1:
                    return False
        return True

    # Initial greedy with sorted order (small to large)
    # This often gives a decent baseline
    current_set = []
    for c in candidates:
        if is_valid(current_set + [c]):
            current_set.append(c)
    best_set = current_set[:]

    # Randomized restarts
    while time.time() < deadline:
        # Shuffle candidates
        random.shuffle(candidates)
        current_set = []
        for c in candidates:
            if is_valid(current_set + [c]):
                current_set.append(c)
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            # If we reach a very high size, we might stop early or continue
            if len(best_set) >= 40: # Example threshold
                # Keep searching for a bit more to be sure
                pass

    return best_set
