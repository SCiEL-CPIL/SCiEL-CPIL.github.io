    # Precompute comparables
    comparables = [[] for _ in range(max_mask)]
    for m in range(max_mask):
        for k in range(max_mask):
            if is_comparable(m, k):
                comparables[m].append(k)
