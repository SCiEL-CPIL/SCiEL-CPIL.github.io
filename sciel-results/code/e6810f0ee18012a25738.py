<<<<<<< SEARCH
    def optimize_full(pts0):
        """Optimize center (2D) + rotation (1D) for fixed hex positions."""
        best_S = float('inf')
        best_beta = 0.0
        best_c = np.mean(pts0, axis=0)

        def calc_S_beta(beta):
            """LP over center for fixed beta -> (S, center)."""
            normals = get_normals(beta)
            n_pts = len(pts0)
            proj = pts0 @ normals.T  # (n,6)
            A1 = np.zeros((n_pts, 6, 3)); A2 = np.zeros((n_pts, 6, 3))
            A1[:, :, 0] = -normals[0]; A1[:, :, 1] = -normals[1]; A1[:, :, 2] = -1.0
            A2[:, :, 0] = normals[0]; A2[:, :, 1] = normals[1]; A2[:, :, 2] = -1.0
            A_ub = np.concatenate([A1.reshape(-1, 3), A2.reshape(-1, 3)], axis=0)
            b_ub = np.concatenate([-proj.ravel(), proj.ravel()])
            try:
                res = linprog(np.array([0.0, 0.0, 1.0]), A_ub=A_ub, b_ub=b_ub,
                              bounds=[(None, None), (None, None), (0, None)],
                              method='highs')
                if res.status == 0:
                    return S_from_D(float(res.x[2])), (float(res.x[0]), float(res.x[1]))
            except Exception:
                pass
            # fallback
            c0 = np.mean(pts0, axis=0)

            def D_of_c(c):
                return D_exact(pts0, c, normals)
            r = minimize(D_of_c, c0, method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-12})
            return S_from_D(float(r.fun)), (float(r.x[0]), float(r.x[1]))

        # phase 1: coarse grid over beta
        for i in range(41):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (PI / 6.0) / 40.0
            S, c = calc_S_beta(beta)
            if S < best_S:
                best_S, best_beta, best_c = S, beta, np.asarray(c)
        # phase 2: scalar refine
        try:
            lo = max(0.0, best_beta - PI / 60.0)
            hi = min(PI / 6.0, best_beta + PI / 60.0)
            r = minimize_scalar(lambda b: calc_S_beta(b)[0], bounds=(lo, hi),
                                method='bounded', options={'maxiter': 300, 'xatol': 1e-13})
            if r.fun < best_S:
                best_S, best_beta = float(r.fun), float(r.x)
                _, best_c = calc_S_beta(best_beta)
                best_c = np.asarray(best_c)
        except Exception:
            pass
        return best_S, best_beta, best_c
=======
    def optimize_full(pts0):
        """Optimize center (2D) + rotation (1D) for fixed hex positions."""
        best_S = float('inf')
        best_beta = 0.0
        best_c = np.mean(pts0, axis=0)

        def calc_S_beta(beta):
            """LP over center for fixed beta -> (S, center)."""
            normals = get_normals(beta)
            n_pts = len(pts0)
            proj = pts0 @ normals.T  # (n,6)
            
            # Corrected LP setup
            # Variables: cx, cy, D
            # Constraints: n_i . (p - c) <= D  =>  -n_i.x cx - n_i.y cy + D >= -n_i.p
            #             n_i . (p - c) >= -D  =>   n_i.x cx + n_i.y cy + D >=  n_i.p
            # Standard form A_ub x <= b_ub:
            # (-n_i.x, -n_i.y, 1) . (cx, cy, D) <= -n_i.p
            # ( n_i.x,  n_i.y, 1) . (cx, cy, D) <=  n_i.p
            
            A1 = np.zeros((n_pts, 3))
            A2 = np.zeros((n_pts, 3))
            A1[:, 0] = -normals[:, 0]
            A1[:, 1] = -normals[:, 1]
            A1[:, 2] = 1.0
            
            A2[:, 0] = normals[:, 0]
            A2[:, 1] = normals[:, 1]
            A2[:, 2] = 1.0
            
            A_ub = np.vstack([A1, A2])
            b_ub = np.concatenate([-proj.ravel(), proj.ravel()])
            
            try:
                res = linprog(np.array([0.0, 0.0, 1.0]), A_ub=A_ub, b_ub=b_ub,
                              bounds=[(None, None), (None, None), (0, None)],
                              method='highs')
                if res.status == 0:
                    return S_from_D(float(res.x[2])), (float(res.x[0]), float(res.x[1]))
            except Exception:
                pass
            # fallback
            c0 = np.mean(pts0, axis=0)

            def D_of_c(c):
                return D_exact(pts0, c, normals)
            r = minimize(D_of_c, c0, method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-12})
            return S_from_D(float(r.fun)), (float(r.x[0]), float(r.x[1]))

        # phase 1: coarse grid over beta
        for i in range(41):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (PI / 6.0) / 40.0
            S, c = calc_S_beta(beta)
            if S < best_S:
                best_S, best_beta, best_c = S, beta, np.asarray(c)
        # phase 2: scalar refine
        try:
            lo = max(0.0, best_beta - PI / 60.0)
            hi = min(PI / 6.0, best_beta + PI / 60.0)
            r = minimize_scalar(lambda b: calc_S_beta(b)[0], bounds=(lo, hi),
                                method='bounded', options={'maxiter': 300, 'xatol': 1e-13})
            if r.fun < best_S:
                best_S, best_beta = float(r.fun), float(r.x)
                _, best_c = calc_S_beta(best_beta)
                best_c = np.asarray(best_c)
        except Exception:
            pass
        return best_S, best_beta, best_c
>>>>>>> REPLACE

