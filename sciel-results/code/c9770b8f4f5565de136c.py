<<<<<<< SEARCH
    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        # High-resolution grid search over beta (0 to pi/6)
        n_grid = 720
        for i in range(n_grid + 1):
            beta = i * (math.pi / 6) / n_grid
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        # Refine beta with bounded scalar optimization using smooth surrogate
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                # Multiple starting points for the refinement
                for offset in [0.0, 0.005, -0.005, 0.01, -0.01, 0.015, -0.015]:
                    lo = max(0.0, best_beta + offset - 0.03)
                    hi = min(math.pi / 6, best_beta + offset + 0.03)
                    if lo >= hi:
                        continue
                    res = minimize_scalar(lambda b: calc_S_smooth(pts, b)[0],
                                         bounds=(lo, hi), method='bounded',
                                         options={'maxiter': 1000, 'xatol': 1e-16})
                    if res.fun < best_S:
                        # Verify with exact LP
                        S_exact, center_exact = calc_S_for_beta(pts, res.x)
                        if S_exact < best_S:
                            best_S = S_exact
                            best_beta = res.x
                            best_center = center_exact
        except Exception:
            pass

        # Final polish with high-resolution points around best_beta
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            for i in range(-50, 51):
                beta = best_beta + i * 1e-6
                if beta < 0:
                    beta += math.pi / 6
                if beta > math.pi / 6:
                    beta -= math.pi / 6
                S, center, D = calc_S_for_beta(pts, beta)
                if S < best_S:
                    best_S = S
                    best_beta = beta
                    best_center = center

        # Additional local refinement with tighter tolerance
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            try:
                for offset in [0.0, 1e-4, -1e-4, 2e-4, -2e-4]:
                    lo = max(0.0, best_beta + offset - 1e-5)
                    hi = min(math.pi / 6, best_beta + offset + 1e-5)
                    if lo >= hi:
                        continue
                    res = minimize_scalar(lambda b: calc_S_smooth(pts, b)[0],
                                         bounds=(lo, hi), method='bounded',
                                         options={'maxiter': 2000, 'xatol': 1e-18})
                    if res.fun < best_S:
                        S_exact, center_exact = calc_S_for_beta(pts, res.x)
                        if S_exact < best_S:
                            best_S = S_exact
                            best_beta = res.x
                            best_center = center_exact
            except Exception:
                pass

        # High-precision final polish: exact LP at very fine grid around best_beta
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            for i in range(-200, 201):
                beta = best_beta + i * 1e-7
                if beta < 0:
                    beta += math.pi / 6
                if beta > math.pi / 6:
                    beta -= math.pi / 6
                S, center, D = calc_S_for_beta(pts, beta)
                if S < best_S:
                    best_S = S
                    best_beta = beta
                    best_center = center

        return best_S, best_beta, best_center
=======
    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        # Coarse grid search over beta (0 to pi/6)
        n_grid = 360
        for i in range(n_grid + 1):
            beta = i * (math.pi / 6) / n_grid
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        # Medium-resolution refinement around best
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            for i in range(-36, 37):
                beta = best_beta + i * (math.pi / 6) / (n_grid * 36)
                if 0 <= beta <= math.pi / 6:
                    S, center, D = calc_S_for_beta(pts, beta)
                    if S < best_S:
                        best_S = S
                        best_beta = beta
                        best_center = center

        # Bounded scalar optimization from multiple starting points
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                for offset in [0.0, 0.003, -0.003, 0.006, -0.006]:
                    lo = max(0.0, best_beta + offset - 0.02)
                    hi = min(math.pi / 6, best_beta + offset + 0.02)
                    if lo >= hi:
                        continue
                    res = minimize_scalar(lambda b: calc_S_smooth(pts, b)[0],
                                         bounds=(lo, hi), method='bounded',
                                         options={'maxiter': 500, 'xatol': 1e-15})
                    if res.fun < best_S:
                        S_exact, center_exact = calc_S_for_beta(pts, res.x)
                        if S_exact < best_S:
                            best_S = S_exact
                            best_beta = res.x
                            best_center = center_exact
        except Exception:
            pass

        # Fine grid polish around best_beta
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            for i in range(-100, 101):
                beta = best_beta + i * 1e-6
                if 0 <= beta <= math.pi / 6:
                    S, center, D = calc_S_for_beta(pts, beta)
                    if S < best_S:
                        best_S = S
                        best_beta = beta
                        best_center = center

        # Very fine polish
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            for i in range(-500, 501):
                beta = best_beta + i * 1e-8
                if 0 <= beta <= math.pi / 6:
                    S, center, D = calc_S_for_beta(pts, beta)
                    if S < best_S:
                        best_S = S
                        best_beta = beta
                        best_center = center

        return best_S, best_beta, best_center
>>>>>>> REPLACE

