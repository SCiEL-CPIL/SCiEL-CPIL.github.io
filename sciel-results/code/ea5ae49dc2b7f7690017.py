<<<<<<< SEARCH
            except Exception:
                break

    # Final safety check
=======
            except Exception:
                break

    # Final Nelder-Mead polish
    if time_ok() and best_roots is not None:
        try:
            nm_res = minimize(
                lambda r: _compute_c4(r),
                best_roots.copy(),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            if np.all(np.isfinite(nm_res.x)) and np.all(nm_res.x > 0):
                candidate = np.sort(nm_res.x)
                c4 = _compute_c4(candidate)
                if np.isfinite(c4) and c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Final safety check
>>>>>>> REPLACE
