<<<<<<< SEARCH
    # --- GLOBAL SEARCH (Differential Evolution) ---
    if time_left() > 60:
        def de_obj_0(p):
            pos = p.reshape(N, 2)
            orient = np.zeros(N, dtype=int)
            side = calc_outer_side(np.concatenate([pos.flatten(), orient]))
            penalty = calc_overlap_penalty(pos, orient)
            return side + penalty
            
        bounds = [(-5, 5)] * (2*N)
        try:
            res_de = differential_evolution(de_obj_0, bounds, maxiter=50, popsize=10, tol=1e-4, seed=42, polish=False)
            if res_de.fun < best_side:
                pos_de = res_de.x.reshape(N, 2)
                pen = calc_overlap_penalty(pos_de, np.zeros(N, dtype=int))
                if pen < 1e-4:
                    side_de = calc_outer_side(np.concatenate([res_de.x, np.zeros(N, dtype=int)]))
                    if side_de < best_side:
                        best_side = side_de
                        best_params = np.concatenate([res_de.x, np.zeros(N, dtype=int)])
        except:
            pass
=======
    # --- GLOBAL SEARCH (Differential Evolution) with better bounds ---
    if time_left() > 50:
        def de_obj_0(p):
            pos = p.reshape(N, 2)
            orient = np.zeros(N, dtype=int)
            side = calc_outer_side(np.concatenate([pos.flatten(), orient]))
            penalty = calc_overlap_penalty(pos, orient, penalty_weight=50.0)
            return side + penalty
            
        # Tighter bounds based on expected optimal configuration
        bounds = [(-4, 4)] * (2*N)
        try:
            res_de = differential_evolution(de_obj_0, bounds, maxiter=80, popsize=15, tol=1e-5, seed=42, polish=False, mutation=(0.5, 1.0))
            if res_de.fun < best_side:
                pos_de = res_de.x.reshape(N, 2)
                pen = calc_overlap_penalty(pos_de, np.zeros(N, dtype=int), penalty_weight=50.0)
                if pen < 1e-6:
                    side_de = calc_outer_side(np.concatenate([res_de.x, np.zeros(N, dtype=int)]))
                    if side_de < best_side:
                        best_side = side_de
                        best_params = np.concatenate([res_de.x, np.zeros(N, dtype=int)])
        except:
            pass
>>>>>>> REPLACE

