<<<<<<< SEARCH
    # --- Phase 2: Benders-style surrogate-guided master problem ---
    if time.time() < deadline - 15.0:
        surrogate = _build_surrogate(a, b, best_p)
        row_ind, col_ind = linear_sum_assignment(surrogate)
        p_surr = np.empty(n, dtype=np.int64)
        p_surr[row_ind] = col_ind
        p_surr, c_surr = _local_search(a, b, p_surr, deadline - 10.0)
        if c_surr < best_cost:
            best_cost = c_surr
            best_p = p_surr.copy()
            surrogate = _build_surrogate(a, b, best_p)
=======
    # --- Phase 2: Surrogate-guided master problem with soft-min ---
    if time.time() < deadline - 15.0:
        # Try both standard surrogate and soft-min surrogate
        for temp in [50.0, 200.0]:
            if time.time() > deadline - 15.0:
                break
            surrogate = _soft_min_surrogate(a, b, best_p, temperature=temp)
            row_ind, col_ind = linear_sum_assignment(surrogate)
            p_surr = np.empty(n, dtype=np.int64)
            p_surr[row_ind] = col_ind
            p_surr, c_surr = _local_search(a, b, p_surr, min(deadline - 10.0, time.time() + 15.0))
            if c_surr < best_cost:
                best_cost = c_surr
                best_p = p_surr.copy()

        # Also try standard surrogate
        if time.time() < deadline - 10.0:
            surrogate = _build_surrogate(a, b, best_p)
            row_ind, col_ind = linear_sum_assignment(surrogate)
            p_surr = np.empty(n, dtype=np.int64)
            p_surr[row_ind] = col_ind
            p_surr, c_surr = _local_search(a, b, p_surr, deadline - 10.0)
            if c_surr < best_cost:
                best_cost = c_surr
                best_p = p_surr.copy()
>>>>>>> REPLACE

