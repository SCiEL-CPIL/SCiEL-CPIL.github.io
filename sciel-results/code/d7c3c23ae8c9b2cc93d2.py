<<<<<<< SEARCH
        # SA
        x_sa, y_sa, r_sa, val_sa = sa_optimize(x0, y0, seed=i, max_iters=2000)

        # Best-fit
        x_bf, y_bf, r_bf, val_bf = best_fit_local_search(x_sa, y_sa, max_rounds=20)

        # Gradient polish
        if time.time() - start_time < DEADLINE - 10:
            polished = gradient_polish(x_bf, y_bf, beta_start=15.0, beta_end=500.0, num_steps=4)
            if polished is not None:
                xp, yp, rp, vp = polished
                if vp > val_bf:
                    x_sa, y_sa, r_sa, val_sa = xp, yp, rp, vp

        if val_sa > best_val:
            best_val = val_sa
            best_x = x_sa.copy()
            best_y = y_sa.copy()
            best_r = r_sa.copy()
=======
        # SA
        x_sa, y_sa, r_sa, val_sa = sa_optimize(x0, y0, seed=i, max_iters=2500)

        # Best-fit
        x_bf, y_bf, r_bf, val_bf = best_fit_local_search(x_sa, y_sa, max_rounds=20)

        # Grow radii
        if time.time() - start_time < DEADLINE - 12:
            x_gr, y_gr, r_gr, val_gr = grow_radii(x_bf, y_bf, r_bf, max_rounds=150)
            if val_gr > val_bf:
                x_bf, y_bf, r_bf, val_bf = x_gr, y_gr, r_gr, val_gr

        # Gradient polish
        if time.time() - start_time < DEADLINE - 10:
            polished = gradient_polish(x_bf, y_bf, beta_start=15.0, beta_end=500.0, num_steps=4)
            if polished is not None:
                xp, yp, rp, vp = polished
                if vp > val_bf:
                    x_bf, y_bf, r_bf, val_bf = xp, yp, rp, vp

        # Grow radii again after polish
        if time.time() - start_time < DEADLINE - 8:
            x_gr2, y_gr2, r_gr2, val_gr2 = grow_radii(x_bf, y_bf, r_bf, max_rounds=100)
            if val_gr2 > val_bf:
                x_bf, y_bf, r_bf, val_bf = x_gr2, y_gr2, r_gr2, val_gr2

        if val_bf > best_val:
            best_val = val_bf
            best_x = x_bf.copy()
            best_y = y_bf.copy()
            best_r = r_bf.copy()
>>>>>>> REPLACE

