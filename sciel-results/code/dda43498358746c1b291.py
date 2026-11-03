<<<<<<< SEARCH
    t_end = _time_left()
    if t_end > 5.0:
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 4000, 'xatol': 1e-7, 'fatol': 1e-7,
                                    'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best = list(res.x)
        except Exception:
            pass

    t_end = _time_left()
    if t_end > 4.0:
        try:
            res = minimize(_objective, np.array(best), method='Nelder-Mead',
                           options={'maxiter': 6000, 'xatol': 1e-9, 'fatol': 1e-9,
                                    'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best = list(res.x)
        except Exception:
            pass
=======
    t_end = _time_left()
    if t_end > 5.0:
        try:
            res = minimize(_objective, x0, method='Powell',
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5,
                                    'disp': False})
            if res.fun < best_val:
                best_val = res.fun
                best = list(res.x)
        except Exception:
            pass

    t_end = _time_left()
    if t_end > 4.0:
        try:
            res = minimize(_objective, np.array(best), method='Powell',
                           options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7,
                                    'disp': False})
            if res.fun < best_val:
                best_val = res.fun
                best = list(res.x)
        except Exception:
            pass
>>>>>>> REPLACE
