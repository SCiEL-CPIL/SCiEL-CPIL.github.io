<<<<<<< SEARCH
    # Fallback
    if best_arr is None:
        w = h = 1.0
        seed_x = _grid_seed(w, h)
        slsqp_x = _slsqp_optimize(seed_x, w, h, time_budget=20.0)
        if not _check_validity(slsqp_x[N:2 * N], w, h, slsqp_x[:N]):
            # trivial valid
            xs = np.linspace(1e-3, w - 1e-3, N)
            ys = np.full(N, h / 2.0)
            r = np.full(N, 1e-3)
            best_arr = np.column_stack([xs, ys, r])
        else:
            best_arr = np.column_stack([slsqp_x[N:2 * N], slsqp_x[2 * N:3 * N], slsqp_x[:N]])
=======
    # Fallback
    if best_arr is None:
        w = h = 1.0
        seed_x = _grid_seed(w, h)
        opt_x = _penalty_optimize(seed_x, w, h, time_budget=10.0)
        if not _check_validity(opt_x[N:2 * N], w, h, opt_x[:N]):
            # trivial valid
            xs = np.linspace(1e-3, w - 1e-3, N)
            ys = np.full(N, h / 2.0)
            r = np.full(N, 1e-3)
            best_arr = np.column_stack([xs, ys, r])
        else:
            best_arr = np.column_stack([opt_x[N:2 * N], opt_x[2 * N:3 * N], opt_x[:N]])
>>>>>>> REPLACE

