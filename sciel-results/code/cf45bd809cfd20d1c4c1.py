<<<<<<< SEARCH
    # Final polish with tight tolerance
    if time.time() < deadline - 5:
        res = minimize(
            _objective, best_pts, args=(500,),
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        new_pts = _normalize(res.x)
        new_val = _compute_covering_radius(new_pts)
        if new_val < best_val - 1e-10:
            best_val = new_val
            best_pts = new_pts
=======
    # Final polish with tight tolerance
    if time.time() < deadline - 5:
        res = minimize(
            _objective, best_pts.flatten(), args=(500,),
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        new_pts = _normalize(res.x.reshape(N, 3))
        new_val = _compute_covering_radius(new_pts)
        if new_val < best_val - 1e-10:
            best_val = new_val
            best_pts = new_pts
>>>>>>> REPLACE
