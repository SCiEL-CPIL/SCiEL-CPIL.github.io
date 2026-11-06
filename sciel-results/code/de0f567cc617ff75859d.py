<<<<<<< SEARCH
    # --- Phase 2: local search from the best seed ---
    counts = np.zeros(MAX_SUM + 1, dtype=np.int64)
    # rebuild counts from best_chosen
    a = best_chosen
    if a.size:
        iu, ju = np.triu_indices(a.size)
        np.add.at(counts, a[iu] + a[ju], 1)
    ch = _local_search(best_chosen, counts, rng, deadline, max_iter=3000000)
    if len(ch) > best_size:
        best_size = len(ch)
        best_chosen = ch
=======
    # --- Phase 2: local search from the best seed ---
    counts = np.zeros(MAX_SUM + 1, dtype=np.int64)
    # rebuild counts from best_chosen
    a = best_chosen
    if a.size:
        iu, ju = np.triu_indices(a.size)
        np.add.at(counts, a[iu] + a[ju], 1)
    ch = _local_search(best_chosen, counts, rng, deadline, max_iter=3000000)
    if len(ch) > best_size:
        best_size = len(ch)
        best_chosen = ch
        # Rebuild counts for next phase
        counts[:] = 0
        if ch.size:
            iu, ju = np.triu_indices(ch.size)
            np.add.at(counts, ch[iu] + ch[ju], 1)
>>>>>>> REPLACE

