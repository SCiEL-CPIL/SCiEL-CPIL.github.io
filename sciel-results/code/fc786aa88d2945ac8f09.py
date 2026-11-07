<<<<<<< SEARCH
        # --- Phase 3: Final Polish with Nelder-Mead ---
        # If there is still budget left, run a few Nelder-Mead iterations from the best point
        if best_x is not None and calls_made < self.budget and time.time() < deadline - 5:
            x0 = best_x.copy()
            
            def nm_func(x):
                nonlocal calls_made, best_f, best_x
                if calls_made >= self.budget or time.time() > deadline - 2:
                    return best_f
                x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
                val = func(x)
                calls_made += 1
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val

            try:
                res = minimize(
                    nm_func, x0, method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 50, 'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        return best_x, best_f
=======
        # --- Phase 3: Final Polish with Nelder-Mead ---
        # If there is still budget left, run a few Nelder-Mead iterations from the best point
        if best_x is not None and calls_made < self.budget and time.time() < deadline - 5:
            x0 = best_x.copy()
            
            def nm_func(x):
                nonlocal calls_made, best_f, best_x
                if calls_made >= self.budget or time.time() > deadline - 2:
                    return best_f
                x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
                val = func(x)
                calls_made += 1
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val

            try:
                res = minimize(
                    nm_func, x0, method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 50, 'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        # --- Phase 4: High-precision polish if budget remains ---
        if best_x is not None and calls_made < self.budget - 10 and time.time() < deadline - 5:
            x0 = best_x.copy()
            
            def polish_func(x):
                nonlocal calls_made, best_f, best_x
                if calls_made >= self.budget or time.time() > deadline - 2:
                    return best_f
                x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
                val = func(x)
                calls_made += 1
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val

            try:
                # Use L-BFGS-B with tighter tolerances for final polish
                res = minimize(
                    polish_func, x0, method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * n,
                    options={'gtol': 1e-10, 'maxiter': 100, 'ftol': 1e-12}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        return best_x, best_f
>>>>>>> REPLACE
