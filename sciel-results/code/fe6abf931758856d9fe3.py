<<<<<<< SEARCH
    # Local Perturbation Restart Phase
    # Try perturbing the best solution to escape local minima
    if best_pts_global is not None and time.time() < deadline - 10.0:
        perturb_deadline = deadline - 10.0
        rng_perturb = np.random.RandomState(42)
        
        # Try several perturbations with varying scales and point counts
        for scale, n_pert in [(0.1, 12), (0.06, 8), (0.03, 5), (0.015, 3)]:
            if time.time() > perturb_deadline - 5.0:
                break
            
            perturbations = _local_perturb_restart(best_pts_global, rng_perturb, 
                                                   perturb_scale=scale, n_perturb=n_pert)
            for pert_pts in perturbations:
                if time.time() > perturb_deadline - 2.0:
                    break
                
                # Quick Adam refinement
                if time.time() < perturb_deadline - 3.0:
                    pts_ref, e_ref = _adam_optimize(pert_pts, max_iter=400, 
                                                     lr_start=0.005, lr_end=0.0005,
                                                     deadline=perturb_deadline - 3.0)
                    # Quick L-BFGS polish
                    if time.time() < perturb_deadline - 1.0:
                        pts_pol = _lbfgs_polish(pts_ref, perturb_deadline - 1.0, maxiter=150)
                        e_pol = _energy_only(pts_pol)
                    else:
                        pts_pol, e_pol = pts_ref, e_ref
                    
                    if e_pol < best_energy_global:
                        best_energy_global = e_pol
                        best_pts_global = pts_pol.copy()
=======
    # Basin Hopping Phase
    # Continuously perturb the best solution and refine to escape local minima
    if best_pts_global is not None and time.time() < deadline - 10.0:
        bh_deadline = deadline - 10.0
        rng_bh = np.random.RandomState(42)
        
        while time.time() < bh_deadline - 5.0:
            # Choose a perturbation mode
            mode = rng_bh.choice(['random_subset', 'rotation', 'noise'])
            pert_pts = _basin_hopping_perturb(best_pts_global, rng_bh, mode=mode)
            
            # Refine the perturbed configuration
            # Use Adam for faster convergence from a distant start
            if time.time() < bh_deadline - 2.0:
                pts_ref, e_ref = _adam_optimize(pert_pts, max_iter=500, 
                                                 lr_start=0.01, lr_end=0.001,
                                                 deadline=bh_deadline - 2.0)
                # Polish with L-BFGS
                if time.time() < bh_deadline - 1.0:
                    pts_pol = _lbfgs_polish(pts_ref, bh_deadline - 1.0, maxiter=200)
                    e_pol = _energy_only(pts_pol)
                else:
                    pts_pol, e_pol = pts_ref, e_ref
            else:
                break
            
            if e_pol < best_energy_global:
                best_energy_global = e_pol
                best_pts_global = pts_pol.copy()
>>>>>>> REPLACE
