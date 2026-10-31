    # Final polish with basinhopping if time permits
    if best_points is not None and time.time() - start_time < time_limit - 15:
        try:
            # Use a high beta for the final objective to be close to true min
            final_beta = 1000
            bh_result = basinhopping(
                lambda x: objective_smooth(x, final_beta),
                best_points.flatten(),
                minimizer_kwargs={'method': 'L-BFGS-B', 'maxiter': 100, 'maxfev': 1000},
                niter=5,
                seed=123
            )
            bh_points = bh_result.x.reshape(N, 2)
            areas = calc_areas(bh_points)
            min_area = np.min(areas)
            pen = penalty(bh_points)
            score = min_area - 1000 * pen if pen > 1e-6 else min_area
            if score > best_score:
                best_score = score
                best_points = bh_points.copy()
        except Exception:
            pass
