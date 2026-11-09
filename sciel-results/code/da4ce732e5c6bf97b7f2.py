<<<<<<< SEARCH
def _optimize_from(c0, deadline):
    c = np.asarray(c0, dtype=np.float64).copy()
    c = _project_to_sphere(c)
    if c is None:
        return None

    def fun(x):
        return _objective(x)

    def jac(x):
        return _gradient(x)

    best_c = c.copy()
    best_val = fun(c)

    # 1. L-BFGS-B
    try:
        res = minimize(fun, c, jac=jac, method="L-BFGS-B",
                       options={"maxiter": 300, "maxfun": 2000,
                                "ftol": 1e-16, "gtol": 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x.copy()
    except Exception:
        pass

    if time.time() > deadline - 10.0:
        proj = _project_to_sphere(best_c)
        if proj is not None:
            return proj, float(_objective(proj))
        return None

    # 2. Conjugate Gradient on Sphere (from L-BFGS-B result)
    # Project to sphere first
    c_cg_init = _project_to_sphere(best_c)
    if c_cg_init is not None:
        try:
            cg_c, cg_val = _conjugate_gradient_sphere(fun, c_cg_init, maxiter=200, tol=1e-12, deadline=deadline)
            if cg_val < best_val:
                best_val = cg_val
                best_c = cg_c.copy()
        except Exception:
            pass

    if time.time() > deadline - 10.0:
        proj = _project_to_sphere(best_c)
        if proj is not None:
            return proj, float(_objective(proj))
        return None

    # 3. Nelder-Mead polish
    try:
        res2 = minimize(fun, best_c, method="Nelder-Mead",
                        options={"maxiter": 1500, "xatol": 1e-12,
                                 "fatol": 1e-14, "adaptive": True})
        if res2.fun < best_val:
            best_val = res2.fun
            best_c = res2.x.copy()
    except Exception:
        pass

    # 4. Second L-BFGS-B pass
    if time.time() < deadline - 5.0:
        try:
            res3 = minimize(fun, best_c, jac=jac, method="L-BFGS-B",
                            options={"maxiter": 200, "maxfun": 1000,
                                     "ftol": 1e-16, "gtol": 1e-12})
            if res3.fun < best_val:
                best_val = res3.fun
                best_c = res3.x.copy()
        except Exception:
            pass

    # 5. Final CG polish
    if time.time() < deadline - 2.0:
        c_cg_final_init = _project_to_sphere(best_c)
        if c_cg_final_init is not None:
            try:
                cg_c_final, cg_val_final = _conjugate_gradient_sphere(fun, c_cg_final_init, maxiter=100, tol=1e-14, deadline=deadline)
                if cg_val_final < best_val:
                    best_val = cg_val_final
                    best_c = cg_c_final.copy()
            except Exception:
                pass

    final_c = _project_to_sphere(best_c)
    if final_c is None:
        return None
    return final_c, float(_objective(final_c))
=======
def _optimize_from(c0, deadline):
    c = np.asarray(c0, dtype=np.float64).copy()
    c = _project_to_sphere(c)
    if c is None:
        return None

    def fun(x):
        return _objective(x)

    def jac(x):
        return _gradient(x)

    best_c = c.copy()
    best_val = fun(c)

    # 1. SLSQP with norm constraint (handles scale invariance well)
    try:
        cons = [{'type': 'eq', 'fun': _norm_constraint, 'jac': _norm_jac}]
        res = minimize(fun, c, jac=jac, method='SLSQP', constraints=cons,
                       options={'maxiter': 200, 'ftol': 1e-14, 'maxls': 50})
        proj = _project_to_sphere(res.x)
        if proj is not None:
            v = float(_objective(proj))
            if v < best_val:
                best_val = v
                best_c = proj.copy()
    except Exception:
        pass

    if time.time() > deadline - 10.0:
        proj = _project_to_sphere(best_c)
        if proj is not None:
            return proj, float(_objective(proj))
        return None

    # 2. Conjugate Gradient on Sphere
    c_cg_init = _project_to_sphere(best_c)
    if c_cg_init is not None:
        try:
            cg_c, cg_val = _conjugate_gradient_sphere(fun, c_cg_init, maxiter=200, tol=1e-12, deadline=deadline)
            if cg_val < best_val:
                best_val = cg_val
                best_c = cg_c.copy()
        except Exception:
            pass

    if time.time() > deadline - 10.0:
        proj = _project_to_sphere(best_c)
        if proj is not None:
            return proj, float(_objective(proj))
        return None

    # 3. L-BFGS-B pass
    try:
        res2 = minimize(fun, best_c, jac=jac, method="L-BFGS-B",
                        options={"maxiter": 200, "maxfun": 1000,
                                 "ftol": 1e-16, "gtol": 1e-12})
        proj = _project_to_sphere(res2.x)
        if proj is not None:
            v = float(_objective(proj))
            if v < best_val:
                best_val = v
                best_c = proj.copy()
    except Exception:
        pass

    # 4. Nelder-Mead polish
    if time.time() < deadline - 8.0:
        try:
            res3 = minimize(fun, best_c, method="Nelder-Mead",
                            options={"maxiter": 1000, "xatol": 1e-12,
                                     "fatol": 1e-14, "adaptive": True})
            proj = _project_to_sphere(res3.x)
            if proj is not None:
                v = float(_objective(proj))
                if v < best_val:
                    best_val = v
                    best_c = proj.copy()
        except Exception:
            pass

    # 5. Final SLSQP polish with tighter tolerance
    if time.time() < deadline - 5.0:
        try:
            cons = [{'type': 'eq', 'fun': _norm_constraint, 'jac': _norm_jac}]
            res4 = minimize(fun, best_c, jac=jac, method='SLSQP', constraints=cons,
                            options={'maxiter': 300, 'ftol': 1e-16, 'maxls': 50})
            proj = _project_to_sphere(res4.x)
            if proj is not None:
                v = float(_objective(proj))
                if v < best_val:
                    best_val = v
                    best_c = proj.copy()
        except Exception:
            pass

    # 6. Final CG polish with tight tolerance
    if time.time() < deadline - 2.0:
        c_cg_final_init = _project_to_sphere(best_c)
        if c_cg_final_init is not None:
            try:
                cg_c_final, cg_val_final = _conjugate_gradient_sphere(fun, c_cg_final_init, maxiter=100, tol=1e-14, deadline=deadline)
                if cg_val_final < best_val:
                    best_val = cg_val_final
                    best_c = cg_c_final.copy()
            except Exception:
                pass

    final_c = _project_to_sphere(best_c)
    if final_c is None:
        return None
    return final_c, float(_objective(final_c))
>>>>>>> REPLACE

