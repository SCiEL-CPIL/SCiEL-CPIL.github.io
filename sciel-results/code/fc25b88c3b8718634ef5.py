<<<<<<< SEARCH
    # Main Search Loop
    best_params = None
    best_score = np.inf
    
    inits = [honeycomb_init(), flower_init()]
    # Add a few random inits
    for _ in range(3):
        inits.append(random_init())
        
    for init_params in inits:
        if time.time() > deadline:
            break
            
        # Optimize
        try:
            # Use L-BFGS-B for speed and handling of bounds (though we don't have strict bounds, it's good)
            # Or Nelder-Mead for robustness
            result = minimize(
                objective,
                init_params,
                method='L-BFGS-B',
                options={
                    'maxiter': 2000,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                }
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
                
        except Exception as e:
            pass

        # If L-BFGS-B fails or gets stuck, try Nelder-Mead from the result
        if time.time() > deadline - 10:
            continue
            
        try:
            start_nm = result.x if 'result' in locals() else init_params
            result_nm = minimize(
                objective,
                start_nm,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if result_nm.fun < best_score:
                best_score = result_nm.fun
                best_params = result_nm.x.copy()
        except Exception:
            pass

    # Final Polish
    if best_params is not None and time.time() < deadline - 5.0:
        # Local search around the best solution
        for _ in range(500):
            if time.time() > deadline - 2.0:
                break
            new_params = best_params.copy()
            # Perturb a random subset of variables
            for _ in range(3):
                idx = random.randint(0, 3*N + 3)
                new_params[idx] += random.gauss(0, 0.01)
            new_score = objective(new_params)
            if new_score < best_score:
                best_score = new_score
                best_params = new_params.copy()

    if best_params is None:
        # Fallback
        best_params = honeycomb_init()
=======
    # Main Search Loop
    best_params = None
    best_score = np.inf
    
    # Use the lattice init as the primary, robust starting point
    lattice_params = lattice_init()
    
    # We will try to optimize from the lattice configuration using Nelder-Mead
    # which is robust to the non-smoothness of the penalty terms.
    try:
        result = minimize(
            objective,
            lattice_params,
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-6,
                'fatol': 1e-8,
                'adaptive': True
            }
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x.copy()
    except Exception:
        pass

    # If the optimizer fails or doesn't improve, use the lattice config
    if best_params is None or objective(best_params) > objective(lattice_params):
        best_params = lattice_params
        best_score = objective(best_params)
        
    # Try L-BFGS-B for polishing if time permits
    if time.time() < deadline - 20.0:
        try:
            result_lbf = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                options={
                    'maxiter': 1000,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                }
            )
            if result_lbf.fun < best_score:
                best_score = result_lbf.fun
                best_params = result_lbf.x.copy()
        except Exception:
            pass
>>>>>>> REPLACE

