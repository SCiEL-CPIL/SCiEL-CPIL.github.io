for combo in itertools.combinations(indices, k):
    total_pos = sum(weights[idx] for idx in combo)
    for mask in range(1 << k):
        subset_sum = 0
        for j in range(k):
            if (mask >> j) & 1:
                subset_sum += weights[combo[j]]
        num = total_pos - 2 * subset_sum
        ...
