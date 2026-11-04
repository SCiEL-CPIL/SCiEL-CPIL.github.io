<<<<<<< SEARCH
def _polish(circles, W, H, max_iter=8000):
    """Four-pass polish: coarse, medium, fine, ultra-fine with adaptive eps."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    # Pass 1: coarse (larger eps for smooth landscape)
    result = minimize(
        _penalty_objective, params, args=(W, H, 5e-3, 300.0, 300.0),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 4, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 100}
    )
    params = result.x

    # Pass 2: medium
    result = minimize(
        _penalty_objective, params, args=(W, H, 1e-3, 500.0, 500.0),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 4, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 150}
    )
    params = result.x

    # Pass 3: fine
    result = minimize(
        _penalty_objective, params, args=(W, H, 1e-4, 800.0, 800.0),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 4, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200}
    )
    params = result.x

    # Pass 4: ultra-fine (tightest tolerance, highest penalty)
    result = minimize(
        _penalty_objective, params, args=(W, H, 1e-5, 1500.0, 1500.0),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 4, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 300}
    )
    return result.x.reshape(N, 3)
=======
def _polish(circles, W, H, max_iter=8000):
    """Five-pass polish: coarse, medium, fine, ultra-fine, and final tight pass."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    # Pass 1: coarse (larger eps for smooth landscape)
    result = minimize(
        _penalty_objective, params, args=(W, H, 5e-3, 300.0, 300.0),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 5, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 100}
    )
    params = result.x

    # Pass 2: medium
    result = minimize(
        _penalty_objective, params, args=(W, H, 1e-3, 500.0, 500.0),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 5, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 150}
    )
    params = result.x

    # Pass 3: fine
    result = minimize(
        _penalty_objective, params, args=(W, H, 1e-4, 800.0, 800.0),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 5, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200}
    )
    params = result.x

    # Pass 4: ultra-fine (tightest tolerance, highest penalty)
    result = minimize(
        _penalty_objective, params, args=(W, H, 1e-5, 1500.0, 1500.0),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 5, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 300}
    )
    params = result.x

    # Pass 5: final tight pass with very small eps and high weights
    result = minimize(
        _penalty_objective, params, args=(W, H, 1e-6, 3000.0, 3000.0),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 5, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 400}
    )
    return result.x.reshape(N, 3)
>>>>>>> REPLACE

