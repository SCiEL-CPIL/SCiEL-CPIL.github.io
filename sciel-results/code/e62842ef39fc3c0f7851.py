<<<<<<< SEARCH
        # Phase 3: L-BFGS-B (polish)
        if time.time() < current_deadline - 1.0:
            pts_lbfgs = _lbfgs_polish(pts_adam, current_deadline - 1.0, maxiter=300)
            e_lbfgs = _energy_only(pts_lbfgs)
        else:
            pts_lbfgs, e_lbfgs = pts_adam, e_adam
=======
        # Phase 3: BB gradient descent (fast local convergence)
        if time.time() < current_deadline - 3.0:
            pts_bb, e_bb = _bb_gradient_descent_sphere(pts_adam, current_deadline - 3.0, max_iter=1500)
        
        # Phase 4: L-BFGS-B (polish)
        if time.time() < current_deadline - 0.5:
            pts_lbfgs = _lbfgs_polish(pts_bb, current_deadline - 0.5, maxiter=200)
            e_lbfgs = _energy_only(pts_lbfgs)
        else:
            pts_lbfgs, e_lbfgs = pts_bb, e_bb
>>>>>>> REPLACE
