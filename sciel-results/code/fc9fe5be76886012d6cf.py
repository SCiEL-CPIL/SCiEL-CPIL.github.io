<<<<<<< SEARCH
    # ---------- driver: restarts -> best -> local search ----------
    best_code = None
    best_len = float('inf')

    # greedy restarts (cheap, vectorized)
    while time.time() < deadline - 30.0:
        seed = int((time.time() - start_time) * 100000) % (10 ** 9)
        code = greedy_cover(seed)
        refined = refine(code)
        if len(refined) < best_len:
            best_len = len(refined)
            best_code = refined

    if best_code is not None and time.time() < deadline - 2.0:
        ls_code = local_search(best_code, 12345, deadline - 2.0)
        if len(ls_code) < best_len:
            best_len = len(ls_code)
            best_code = ls_code
        # Second local search pass with different seed for diversity
        if time.time() < deadline - 1.0:
            ls_code2 = local_search(best_code, 99999, deadline - 1.0)
            if len(ls_code2) < best_len:
                best_len = len(ls_code2)
                best_code = ls_code2
        # Third pass if time remains, with yet another seed
        if time.time() < deadline - 0.5:
            ls_code3 = local_search(best_code, 77777, deadline - 0.5)
            if len(ls_code3) < best_len:
                best_len = len(ls_code3)
                best_code = ls_code3

    if best_code is not None:
        final_code = refine(best_code)
    else:
        final_code = []

    return [int_to_base3(c, n) for c in final_code]
=======
    # ---------- driver: restarts -> best -> local search ----------
    best_code = None
    best_len = float('inf')

    # greedy restarts (cheap, vectorized)
    while time.time() < deadline - 35.0:
        seed = int((time.time() - start_time) * 100000) % (10 ** 9)
        code = greedy_cover(seed)
        refined = refine(code)
        if len(refined) < best_len:
            best_len = len(refined)
            best_code = refined

    if best_code is not None:
        # Run multiple local search passes with different seeds
        ls_seeds = [12345, 99999, 77777, 31415, 27182, 16180]
        for ls_seed in ls_seeds:
            if time.time() >= deadline - 1.0:
                break
            remaining_time = deadline - 1.0 - time.time()
            if remaining_time < 0.5:
                break
            ls_code = local_search(best_code, ls_seed, time.time() + remaining_time)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code
            # Quick refine after each LS pass
            refined = refine(best_code)
            if len(refined) < best_len:
                best_len = len(refined)
                best_code = refined

    if best_code is not None:
        final_code = refine(best_code)
    else:
        final_code = []

    return [int_to_base3(c, n) for c in final_code]
>>>>>>> REPLACE

