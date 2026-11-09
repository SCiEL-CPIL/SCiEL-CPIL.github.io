<<<<<<< SEARCH
    # Multi-start construction + local search for reference tour
    best_tour = None
    best_len = float('inf')
    rng = np.random.default_rng(42)

    def refine(tour):
        nonlocal best_tour, best_len
        t = two_opt(tour)
        t = or_opt(t)
        t = two_opt(t)
        L = tour_len(t)
        if L < best_len:
            best_len = L
            best_tour = list(t)
        return t

    # Start from node 0
    refine(nearest_neighbor(0))

    # Randomized starts
    num_starts = 200
    for _ in range(num_starts):
        if time.time() > deadline - 15.0:
            break
        s = int(rng.integers(n))
        refine(nearest_neighbor(s))

    # Random perturbation + local search on best tour
    for _ in range(60):
        if time.time() > deadline - 5.0:
            break
        t = list(best_tour)
        for _ in range(2):
            i = int(rng.integers(n))
            j = int(rng.integers(n))
            if i > j:
                i, j = j, i
            if j - i >= 1:
                t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
        refine(t)

    if best_tour is None:
        best_tour = two_opt(nearest_neighbor(0))
=======
    # Multi-start construction + local search for reference tour
    best_tour = None
    best_len = float('inf')
    rng = np.random.default_rng(42)

    def refine(tour):
        nonlocal best_tour, best_len
        t = two_opt_vectorized(tour)
        t = or_opt(t)
        t = two_opt_vectorized(t)
        t = three_opt(t)
        t = two_opt_vectorized(t)
        L = tour_len(t)
        if L < best_len:
            best_len = L
            best_tour = list(t)
        return t

    # Start from node 0
    refine(nearest_neighbor(0))

    # Randomized starts
    num_starts = 500
    for _ in range(num_starts):
        if time.time() > deadline - 30.0:
            break
        s = int(rng.integers(n))
        refine(nearest_neighbor(s))

    # Random perturbation + local search on best tour
    for _ in range(200):
        if time.time() > deadline - 5.0:
            break
        t = list(best_tour)
        # Apply 2 random 2-opt reversals
        for _ in range(2):
            i = int(rng.integers(n - 1))
            j = int(rng.integers(i + 2, n)) if i + 2 < n else i + 1
            if j - i >= 2:
                t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
        refine(t)

    if best_tour is None:
        best_tour = two_opt_vectorized(nearest_neighbor(0))
>>>>>>> REPLACE

