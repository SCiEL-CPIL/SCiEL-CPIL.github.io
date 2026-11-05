def _find_bad_subset_sampled(adj_mask, n, num_samples=1000):
    rng = random.Random()
    for _ in range(num_samples):
        k = rng.randint(6, n)
        subset = rng.sample(range(n), k)
        mask = 0
        for v in subset:
            mask |= (1 << v)
        if _is_regular_subset(adj_mask, mask, n):
            return mask, subset
    return 0, []
