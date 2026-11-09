<<<<<<< SEARCH
def _multi_restart_optimize(n, cut_sizes, p, num_restarts=20):
    """Multi-restart Nelder-Mead with diverse initial guesses."""
    best_angles = None
    best_score = -np.inf
    
    # Generate diverse initial guesses
    np.random.seed(123)
    initial_guesses = []
    
    # Known good QAOA angles for p=2 (from literature and empirical testing)
    initial_guesses.append(np.array([0.35, 0.25, 0.35, 0.25]))
    initial_guesses.append(np.array([0.5, 0.3, 0.5, 0.3]))
    initial_guesses.append(np.array([0.2, 0.4, 0.2, 0.4]))
    initial_guesses.append(np.array([0.7, 0.2, 0.7, 0.2]))
    initial_guesses.append(np.array([0.4, 0.35, 0.4, 0.35]))
    initial_guesses.append(np.array([0.6, 0.4, 0.6, 0.4]))
    initial_guesses.append(np.array([0.3, 0.3, 0.3, 0.3]))
    initial_guesses.append(np.array([0.45, 0.2, 0.45, 0.2]))
    initial_guesses.append(np.array([0.55, 0.25, 0.55, 0.25]))
    initial_guesses.append(np.array([0.25, 0.35, 0.25, 0.35]))
    
    # Random guesses with better coverage
    for _ in range(num_restarts - len(initial_guesses)):
        gammas = np.random.uniform(0.1, np.pi * 0.8, p)
        betas = np.random.uniform(0.1, np.pi * 0.7, p)
        initial_guesses.append(np.column_stack((gammas, betas)).flatten())
    
    # Optimize from each initial guess
    for x0 in initial_guesses:
        if _time_left() < 10.0:
            break
        
        x_opt, score = _nelder_mead_optimize(x0, n, cut_sizes, p, maxiter=150, tol=1e-8)
        
        if score > best_score:
            best_score = score
            best_angles = x_opt.reshape(p, 2)
    
    # Polish the best result with tighter tolerance
    if best_angles is not None and _time_left() > 15.0:
        x_polished, score_polished = _polish(
            best_angles.flatten(), n, cut_sizes, p, maxiter=500, tol=1e-12
        )
        if score_polished > best_score:
            best_score = score_polished
            best_angles = x_polished.reshape(p, 2)
    
    return best_angles, best_score
=======
def _multi_restart_optimize(n, cut_sizes, p, num_restarts=20):
    """Multi-restart Nelder-Mead with diverse initial guesses."""
    best_angles = None
    best_score = -np.inf
    
    # Generate diverse initial guesses
    np.random.seed(123)
    initial_guesses = []
    
    # Known good QAOA angles for p=2 (from literature and empirical testing)
    initial_guesses.append(np.array([0.35, 0.25, 0.35, 0.25]))
    initial_guesses.append(np.array([0.5, 0.3, 0.5, 0.3]))
    initial_guesses.append(np.array([0.2, 0.4, 0.2, 0.4]))
    initial_guesses.append(np.array([0.7, 0.2, 0.7, 0.2]))
    initial_guesses.append(np.array([0.4, 0.35, 0.4, 0.35]))
    initial_guesses.append(np.array([0.6, 0.4, 0.6, 0.4]))
    initial_guesses.append(np.array([0.3, 0.3, 0.3, 0.3]))
    initial_guesses.append(np.array([0.45, 0.2, 0.45, 0.2]))
    initial_guesses.append(np.array([0.55, 0.25, 0.55, 0.25]))
    initial_guesses.append(np.array([0.25, 0.35, 0.25, 0.35]))
    
    # Add more diverse structured guesses
    initial_guesses.append(np.array([0.4, 0.15, 0.3, 0.2]))
    initial_guesses.append(np.array([0.3, 0.2, 0.5, 0.15]))
    initial_guesses.append(np.array([0.6, 0.1, 0.4, 0.25]))
    initial_guesses.append(np.array([0.25, 0.3, 0.35, 0.2]))
    
    # Random guesses with better coverage
    for _ in range(max(0, num_restarts - len(initial_guesses))):
        gammas = np.random.uniform(0.1, np.pi * 0.8, p)
        betas = np.random.uniform(0.1, np.pi * 0.7, p)
        initial_guesses.append(np.column_stack((gammas, betas)).flatten())
    
    # Optimize from each initial guess
    for x0 in initial_guesses:
        if _time_left() < 12.0:
            break
        
        x_opt, score = _nelder_mead_optimize(x0, n, cut_sizes, p, maxiter=200, tol=1e-9)
        
        if score > best_score:
            best_score = score
            best_angles = x_opt.reshape(p, 2)
    
    # Polish the best result with tighter tolerance
    if best_angles is not None and _time_left() > 18.0:
        x_polished, score_polished = _polish(
            best_angles.flatten(), n, cut_sizes, p, maxiter=600, tol=1e-13
        )
        if score_polished > best_score:
            best_score = score_polished
            best_angles = x_polished.reshape(p, 2)
    
    return best_angles, best_score
>>>>>>> REPLACE

