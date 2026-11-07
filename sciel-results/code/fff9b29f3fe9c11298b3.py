    while time.time() < deadline - 5:
        # Find the spec with the largest count
        max_idx = max(range(3), key=lambda i: best_counts[i])
        n, k, p, seed = specs[max_idx]
        rng_seed = random.randint(0, 2**31 - 1)
        cliques = get_cliques(n, k, p, seed, rng_seed)
        count = len(cliques)
        if count < best_counts[max_idx]:
            best_counts[max_idx] = count
            best_results[max_idx] = cliques
