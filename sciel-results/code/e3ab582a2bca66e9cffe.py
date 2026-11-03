<<<<<<< SEARCH
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    # 1) Deterministic greedy baseline
    best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))
    best_max = _max_entry(best_rows)

    # 2) Many randomized greedy restarts
    rng = random.Random(42)
    greedy_deadline = deadline - 60
    trial = 0
    while time.time() < greedy_deadline:
        trial += 1
        skip_p = rng.random() * 0.15
        jump_p = rng.random() * 0.08
        shuffle = rng.random() < 0.6
        bias_p = rng.random() * 0.4
        bias_range = rng.randint(3, 8)
        rows = _greedy_variant(n_rows, k, rng, skip_p=skip_p, jump_p=jump_p,
                               shuffle_rows=shuffle, bias_p=bias_p, bias_range=bias_range)
        if rows is not None and _check_valid(rows, k):
            rmax = _max_entry(rows)
            if rmax < best_max:
                best_rows = rows
                best_max = rmax

    # 3) Local compaction
    compact_deadline = deadline - 40
    if time.time() < compact_deadline:
        compacted = _local_compact(best_rows, n_rows, k, compact_deadline)
        if _check_valid(compacted, k):
            cmax = _max_entry(compacted)
            if cmax < best_max:
                best_rows = compacted
                best_max = cmax

    # 4) Bitset-accelerated backtracking scope reduction
    backtrack_deadline = deadline - 20
    if time.time() < backtrack_deadline:
        improved = _improve_scope_bitset(best_rows, n_rows, k, backtrack_deadline)
        if _check_valid(improved, k):
            imax = _max_entry(improved)
            if imax < best_max:
                best_rows = improved
                best_max = imax

    # 5) Simulated annealing
    sa_deadline = deadline - 2
    if time.time() < sa_deadline:
        sa_result = _simulated_annealing(best_rows, n_rows, k, rng, sa_deadline)
        if _check_valid(sa_result, k):
            smax = _max_entry(sa_result)
            if smax < best_max:
                best_rows = sa_result
                best_max = smax

    # 6) Final validation
    if not _check_valid(best_rows, k):
        best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))

    return best_rows
=======
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    # 1) Deterministic greedy baseline
    best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))
    best_max = _max_entry(best_rows)

    # 2) Many randomized greedy restarts
    rng = random.Random(42)
    greedy_deadline = deadline - 55
    trial = 0
    while time.time() < greedy_deadline:
        trial += 1
        skip_p = rng.random() * 0.15
        jump_p = rng.random() * 0.08
        shuffle = rng.random() < 0.6
        bias_p = rng.random() * 0.4
        bias_range = rng.randint(3, 8)
        rows = _greedy_variant(n_rows, k, rng, skip_p=skip_p, jump_p=jump_p,
                               shuffle_rows=shuffle, bias_p=bias_p, bias_range=bias_range)
        if rows is not None and _check_valid(rows, k):
            rmax = _max_entry(rows)
            if rmax < best_max:
                best_rows = rows
                best_max = rmax

    # 3) Local compaction with bitset
    compact_deadline = deadline - 35
    if time.time() < compact_deadline:
        compacted = _local_compact_bitset(best_rows, n_rows, k, compact_deadline)
        if _check_valid(compacted, k):
            cmax = _max_entry(compacted)
            if cmax < best_max:
                best_rows = compacted
                best_max = cmax

    # 4) Bitset-accelerated backtracking scope reduction
    backtrack_deadline = deadline - 15
    if time.time() < backtrack_deadline:
        improved = _improve_scope_bitset(best_rows, n_rows, k, backtrack_deadline)
        if _check_valid(improved, k):
            imax = _max_entry(improved)
            if imax < best_max:
                best_rows = improved
                best_max = imax

    # 5) Simulated annealing
    sa_deadline = deadline - 2
    if time.time() < sa_deadline:
        sa_result = _simulated_annealing(best_rows, n_rows, k, rng, sa_deadline)
        if _check_valid(sa_result, k):
            smax = _max_entry(sa_result)
            if smax < best_max:
                best_rows = sa_result
                best_max = smax

    # 6) Final validation
    if not _check_valid(best_rows, k):
        best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))

    return best_rows
>>>>>>> REPLACE
