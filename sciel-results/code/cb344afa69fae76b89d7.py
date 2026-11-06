import time
import random

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin under 60s kill
    n = 512
    target = 48

    def build_dp(arr):
        """Build subset-sum DP: dp[k] is bitset of sums of k-element subsets of arr."""
        m = len(arr)
        dp = [0] * (m + 1)
        dp[0] = 1
        for val in arr:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k - 1] << val)
        return dp

    def is_valid_set(S):
        """Check if S is valid: no element is mean of any k-subset of others, k>=2."""
        m = len(S)
        if m <= 2:
            return True
        S_set = set(S)
        for i, x in enumerate(S):
            others = [v for j, v in enumerate(S) if j != i]
            m_others = len(others)
            dp = build_dp(others)
            for k in range(2, m_others + 1):
                target_sum = k * x
                if target_sum > m_others * n:
                    continue
                if (dp[k] >> target_sum) & 1:
                    return False
        return True

    def greedy_build(order, current_best_len):
        """Greedily build a valid set following the given order."""
        C = []
        for x in order:
            if len(C) >= target:
                break
            # Tentatively add x and check validity
            C.append(x)
            if is_valid_set(C):
                pass  # keep x
            else:
                C.pop()
            if len(C) > current_best_len:
                current_best_len = len(C)
        return C

    # Strategy: focus on odd numbers since they have better parity properties
    odd_numbers = list(range(1, n + 1, 2))  # 256 odd numbers from 1 to 511
    all_numbers = list(range(1, n + 1))

    best = []
    rng = random.Random(42)

    # First try: sorted odd numbers (increasing)
    trial_order = sorted(odd_numbers)
    C = greedy_build(trial_order, len(best))
    if len(C) > len(best):
        best = C

    # Try: sorted odd numbers (decreasing)
    trial_order = sorted(odd_numbers, reverse=True)
    C = greedy_build(trial_order, len(best))
    if len(C) > len(best):
        best = C

    # Multiple random trials with odd numbers
    max_trials = 500
    for trial in range(max_trials):
        if time.time() > deadline - 2.0:
            break
        if len(best) >= target:
            break

        # Mix: mostly odd numbers with some random selection
        # Try different strategies
        if trial % 3 == 0:
            order = list(odd_numbers)
            rng.shuffle(order)
        elif trial % 3 == 1:
            # Weighted toward larger odd numbers
            order = list(odd_numbers)
            rng.shuffle(order)
            # Put larger numbers first
            order.sort(reverse=True)
            rng.shuffle(order[:len(order)//2])
        else:
            # Mix odd and even
            order = list(all_numbers)
            rng.shuffle(order)

        C = greedy_build(order, len(best))
        if len(C) > len(best):
            best = C
            if len(best) >= target:
                break

    # Try local improvement on best found
    if len(best) < target and time.time() < deadline - 3.0:
        best_set = set(best)
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            # Try adding each unused number
            candidates = [x for x in range(1, n + 1) if x not in best_set]
            rng.shuffle(candidates)
            for x in candidates:
                if len(best) >= target:
                    break
                test_set = best + [x]
                if is_valid_set(test_set):
                    best = test_set
                    best_set.add(x)
                    improved = True
                    break

    # Final validation
    if best and not is_valid_set(best):
        # Trim to valid
        S = list(best)
        while S and not is_valid_set(S):
            S.pop()
        best = S

    return best
