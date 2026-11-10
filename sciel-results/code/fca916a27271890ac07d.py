    # Precompute suffix sums of values for quick loose bound
    suffix_sums = [0] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + values[i]

    def bound(idx, used_mask):
        # Loose bound first
        if curr_sum + suffix_sums[idx] <= best_sum:
            return False
        # Tight bound
        b = 0
        for j in range(idx, N):
            if item_masks[j] & used_mask == 0:
                b += values[j]
        return curr_sum + b > best_sum
