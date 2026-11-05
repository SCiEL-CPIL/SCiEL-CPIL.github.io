def construct():
    deadline = time.time() + (90.0 - DEADLINE_MARGIN)
    best_size = 0
    best_sets = ([], [])

    # Strategy 1: Deterministic greedy with different orderings
    orderings = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        [x for x in range(1, N + 1) if x % 2 == 1] + [x for x in range(1, N + 1) if x % 2 == 0],
        [x for x in range(N, 0, -1) if x % 2 == 0] + [x for x in range(N, 0, -1) if x % 2 == 1],
        list(range(1, 65)) + list(range(65, 129)) + list(range(129, 193)) + list(range(193, 257)),
        list(range(193, 257)) + list(range(129, 193)) + list(range(65, 129)) + list(range(1, 65)),
        [x for x in range(1, N + 1) if x % 3 != 0] + [x for x in range(1, N + 1) if x % 3 == 0],
        [x for x in range(N, 0, -1) if x % 3 != 0] + [x for x in range(N, 0, -1) if x % 3 == 0],
        [x for x in range(1, N + 1) if x % 4 in (1, 2)] + [x for x in range(1, N + 1) if x % 4 in (0, 3)],
        [x for x in range(N, 0, -1) if x % 5 in (1, 2, 3)] + [x for x in range(N, 0, -1) if x % 5 in (0, 4)],
    ]

    for order in orderings:
        if time.time() > deadline:
            break
        A, B = _greedy_build(order, deadline=deadline)
        if _valid_pair(A, B):
            s = len(A) + len(B)
            if s > best_size:
                best_size = s
                best_sets = (list(A), list(B))
        A2, B2 = _greedy_build(order, prefer_A_first=False, deadline=deadline)
        if _valid_pair(A2, B2):
            s = len(A2) + len(B2)
            if s > best_size:
                best_size = s
                best_sets = (list(A2), list(B2))

    # Strategy 2: Randomized greedy with multiple restarts
    rng = random.Random(42)
    
    # Run as many iterations as possible
    while time.time() < deadline - 5.0:
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        A, B = _greedy_build(candidates, deadline=deadline)
        if _valid_pair(A, B):
            s = len(A) + len(B)
            if s > best_size:
                best_size = s
                best_sets = (list(A), list(B))
            
            # Local search on this candidate
            # Run local search if we have time and it's promising
            if s >= best_size - 5 and time.time() < deadline - 1.0:
                A_ls, B_ls = _local_search(A, B, max_iters=2000, deadline=deadline)
                if _valid_pair(A_ls, B_ls):
                    s2 = len(A_ls) + len(B_ls)
                    if s2 > best_size:
                        best_size = s2
                        best_sets = (list(A_ls), list(B_ls))

    # Strategy 3: Final polish on the best found
    if best_sets[0] and best_sets[1] and time.time() < deadline - 0.5:
        A_ls, B_ls = _local_search(best_sets[0], best_sets[1], max_iters=50000, deadline=deadline)
        if _valid_pair(A_ls, B_ls):
            s = len(A_ls) + len(B_ls)
            if s > best_size:
                best_size = s
                best_sets = (list(A_ls), list(B_ls))

    # Fallback
    if not best_sets[0] or not best_sets[1]:
        best_sets = ([1, 10], [2, 20])
        best_size = 4

    # Final validation
    if not _valid_pair(best_sets[0], best_sets[1]):
        best_sets = ([1, 10], [2, 20])

    return [list(best_sets[0]), list(best_sets[1])]
