            # Phase 1: L-BFGS-B for local refinement
            if time.time() < deadline - 10:
                try:
                    res = sco.minimize(f, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})
                    if res.success:
                        x_nm = res.x
                        f_nm = res.fun
                except Exception:
                    x_nm = x0
                    f_nm = f0
