<<<<<<< SEARCH
def construct():
    n_basis = 6
    # Parameter bounds:
    # Centers c_i in [-0.25, 0.25]
    # Widths w_i in [0.01, 0.5]
    # Amplitudes a_i in [0, 5]
    
    bounds = []
    for _ in range(n_basis):
        bounds.append((-0.25, 0.25))  # center
        bounds.append((0.01, 0.5))     # width
        bounds.append((0.0, 5.0))      # amplitude
    
    def objective(params):
        heights = _gaussian_mixture_heights(params, n_basis)
        return _compute_score(heights)
    
    # Use differential evolution with tuned parameters
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=100,  # Reduced iterations to fit in time with finer grid
        tol=1e-6,
        popsize=10,   # Reduced population
        init='sobol',
        polish=True,
        updating='immediate',
        workers=1
    )
=======
def construct():
    n_basis = 10
    # Parameter bounds:
    # Centers c_i in [-0.25, 0.25]
    # Widths w_i in [0.01, 0.5]
    # Amplitudes a_i in [0, 5]
    
    bounds = []
    for _ in range(n_basis):
        bounds.append((-0.25, 0.25))  # center
        bounds.append((0.01, 0.5))     # width
        bounds.append((0.0, 5.0))      # amplitude
    
    def objective(params):
        heights = _gaussian_mixture_heights(params, n_basis)
        return _compute_score(heights)
    
    # Use differential evolution with tuned parameters
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=200,  # Increased iterations for better convergence
        tol=1e-8,
        popsize=20,   # Increased population for better exploration
        init='sobol',
        polish=True,
        updating='immediate',
        workers=1
    )
>>>>>>> REPLACE
