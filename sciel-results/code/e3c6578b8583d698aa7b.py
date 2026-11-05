<<<<<<< SEARCH
    n = len(sequence)
    # Phase 1: generate a pool of diverse starting conformations via greedy growth
    starts = []
    for _ in range(60):
        if _time_left() < 2.0:
            break
        c = _grow(sequence, rng)
        if c is None or len(c) != n:
            continue
        if not _is_valid(c):
            continue
        starts.append(c)
        if len(starts) >= 25:
            break
=======
    n = len(sequence)
    # Phase 1: generate a pool of diverse starting conformations via greedy growth
    starts = []
    # Reduced restarts to ensure time safety
    for _ in range(20):
        if _time_left() < 2.0:
            break
        c = _grow(sequence, rng)
        if c is None or len(c) != n:
            continue
        if not _is_valid(c):
            continue
        starts.append(c)
        if len(starts) >= 10:
            break
>>>>>>> REPLACE

And adjust the per-start budget calculation in `fold`.

