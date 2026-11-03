<<<<<<< SEARCH
def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0

    rng = random.Random(12345)

    # Precompute triples
    triples, triple_to_idx = _precompute_triples(v, t)
    n_triples = len(triples)

    # Precompute all blocks and their bitmask coverage
    all_blocks = []
    block_masks = []
    for block in itertools.combinations(range(v), k):
        all_blocks.append(block)
        mask = 0
        for combo in itertools.combinations(block, t):
            idx = triple_to_idx.get(combo, -1)
            if idx >= 0:
                mask |= (1 << idx)
        block_masks.append(mask)

    # Step 1: Multiple greedy restarts (point-by-point)
    best_greedy = None
    best_greedy_n = float('inf')

    greedy_time = 15.0
    greedy_deadline = start_time + greedy_time

    for restart in range(10):
        if time.time() > greedy_deadline:
            break
        r = random.Random(12345 + restart * 7919)
        blocks = _greedy_cover_bitmask(v, k, t, r, triples, triple_to_idx)
        if len(blocks) < best_greedy_n:
            best_greedy_n = len(blocks)
            best_greedy = blocks

    if best_greedy is None:
        best_greedy = _greedy_cover_bitmask(v, k, t, rng, triples, triple_to_idx)

    # Step 2: Iterative greedy set cover with precomputed masks
    remaining = time_limit - (time.time() - start_time)
    if remaining > 10.0:
        igc_time = min(remaining * 0.4, 30.0)
        igc_blocks, igc_n = _iterative_greedy_set_cover(
            v, k, t, rng, triples, triple_to_idx,
            block_masks, all_blocks, time_limit=igc_time
        )
        if igc_blocks is not None and igc_n < best_greedy_n:
            best_greedy = igc_blocks
            best_greedy_n = igc_n

    # Step 3: Simulated annealing local search
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5.0:
        sa_time = remaining - 5.0
        best_blocks, best_n = _local_search_sa(
            best_greedy, v, k, t, triples, triple_to_idx,
            time_limit=sa_time
        )
    else:
        best_blocks = best_greedy
        best_n = len(best_greedy)

    return [list(b) for b in best_blocks]
=======
def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0

    rng = random.Random(12345)

    # Precompute triples
    triples, triple_to_idx = _precompute_triples(v, t)
    n_triples = len(triples)

    # Precompute all blocks and their bitmask coverage (fast incremental)
    all_blocks = []
    block_masks = []
    for block in itertools.combinations(range(v), k):
        all_blocks.append(block)
        mask = 0
        for combo in itertools.combinations(block, t):
            idx = triple_to_idx.get(combo, -1)
            if idx >= 0:
                mask |= (1 << idx)
        block_masks.append(mask)

    # Step 1: Multiple greedy restarts (point-by-point)
    best_greedy = None
    best_greedy_n = float('inf')

    greedy_time = 12.0
    greedy_deadline = start_time + greedy_time

    for restart in range(12):
        if time.time() > greedy_deadline:
            break
        r = random.Random(12345 + restart * 7919)
        blocks = _greedy_cover_bitmask(v, k, t, r, triples, triple_to_idx)
        if len(blocks) < best_greedy_n:
            best_greedy_n = len(blocks)
            best_greedy = blocks

    if best_greedy is None:
        best_greedy = _greedy_cover_bitmask(v, k, t, rng, triples, triple_to_idx)

    # Step 2: Iterative greedy set cover with precomputed masks
    remaining = time_limit - (time.time() - start_time)
    if remaining > 15.0:
        igc_time = min(remaining * 0.35, 25.0)
        igc_blocks, igc_n = _iterative_greedy_set_cover(
            v, k, t, rng, triples, triple_to_idx,
            block_masks, all_blocks, time_limit=igc_time
        )
        if igc_blocks is not None and igc_n < best_greedy_n:
            best_greedy = igc_blocks
            best_greedy_n = igc_n

    # Step 3: Post-hoc removal pass
    remaining = time_limit - (time.time() - start_time)
    if remaining > 20.0:
        best_greedy = _post_hoc_removal(best_greedy, v, k, t, triples, triple_to_idx)
        best_greedy_n = len(best_greedy)

    # Step 4: Simulated annealing local search
    remaining = time_limit - (time.time() - start_time)
    if remaining > 8.0:
        sa_time = remaining - 5.0
        best_blocks, best_n = _local_search_sa(
            best_greedy, v, k, t, triples, triple_to_idx,
            time_limit=sa_time
        )
    else:
        best_blocks = best_greedy
        best_n = len(best_greedy)

    # Final removal pass after SA
    remaining = time_limit - (time.time() - start_time)
    if remaining > 3.0:
        best_blocks = _post_hoc_removal(best_blocks, v, k, t, triples, triple_to_idx)
        best_n = len(best_blocks)

    return [list(b) for b in best_blocks]
>>>>>>> REPLACE
