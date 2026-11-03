<<<<<<< SEARCH
    # Phase 3: L-BFGS-B polish for high precision
    if time.time() < deadline - 25.0:
        best_pts = _lbfgs_polish(best_pts, deadline - 25.0, maxiter=600)
        best_energy = _energy_only(best_pts)

    # Phase 3b: Second, tighter L-BFGS-B pass for final precision
    if time.time() < deadline - 12.0:
        best_pts = _lbfgs_tight_polish(best_pts, deadline - 12.0, maxiter=2000)
        best_energy = _energy_only(best_pts)
=======
    # Phase 3: Riemannian CG polish
    if time.time() < deadline - 30.0:
        best_pts, cg_energy = _riemannian_cg(best_pts, max_iter=150,
                                             deadline=deadline - 30.0, check_interval=5)
        if cg_energy < best_energy:
            best_energy = cg_energy
        else:
            best_pts = _project_to_sphere(best_pts)

    # Phase 3b: L-BFGS-B polish for high precision
    if time.time() < deadline - 15.0:
        best_pts = _lbfgs_polish(best_pts, deadline - 15.0, maxiter=600)
        best_energy = _energy_only(best_pts)

    # Phase 3c: Second, tighter L-BFGS-B pass for final precision
    if time.time() < deadline - 8.0:
        best_pts = _lbfgs_tight_polish(best_pts, deadline - 8.0, maxiter=2000)
        best_energy = _energy_only(best_pts)
>>>>>>> REPLACE

