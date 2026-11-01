def _riemannian_step_direction(x, direction, alpha):
    """Take a step in a given direction and project back to sphere."""
    x_new = x + alpha * direction
    norms = np.linalg.norm(x_new, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return x_new / norms

def _line_search(x, grad, alpha, f0):
    """Backtracking line search for Riemannian GD."""
    c1 = 1e-4
    rho = 0.5
    dot = np.sum(x * grad, axis=1, keepdims=True)
    g_tangent = grad - dot * x
    direction = -g_tangent
    slope = np.sum(grad * direction)
    if slope >= 0:
        return x, f0, alpha
    for _ in range(30):
        x_trial = _riemannian_step_direction(x, direction, alpha)
        f_trial, _ = _energy_and_grad(x_trial)
        if f_trial <= f0 + c1 * alpha * slope:
            return x_trial, f_trial, alpha
        alpha *= rho
    return x, f0, alpha

def _optimize_riemannian(x0, max_iter=2000, tol=1e-12, deadline=None):
    x = x0.copy()
    f, g = _energy_and_grad(x)
    alpha = 1e-2
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        dot = np.sum(x * g, axis=1, keepdims=True)
        g_tangent = g - dot * x
        norm_g = np.linalg.norm(g_tangent)
        if norm_g < tol:
            break
        x, f, alpha = _line_search(x, g, alpha, f)
    return x

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_x = None
    best_energy = float('inf')
    
    initializations = [
        ("halton", _halton_init(N)),
        ("fibonacci", _fibonacci_init(N)),
    ]
    
    np.random.seed(42)
    for _ in range(4):
        x_rand = np.random.randn(N, 3)
        x_rand = x_rand / np.linalg.norm(x_rand, axis=1, keepdims=True)
        initializations.append(("random", x_rand))
        
    for name, x0 in initializations:
        if time.time() > deadline:
            break
        
        x0 = _project_to_sphere(x0)
        
        try:
            x_opt = _optimize_riemannian(x0, max_iter=2000, tol=1e-12, deadline=deadline)
            
            energy, _ = _energy_and_grad(x_opt)
            
            if not np.all(np.isfinite(x_opt)):
                continue
            if np.any(np.linalg.norm(x_opt, axis=1) < 1e-15):
                continue
            
            diff = x_opt[:, np.newaxis, :] - x_opt[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            min_dist_sq = np.min(dist_sq)
            if min_dist_sq < (1e-9) ** 2:
                continue
            
            if energy < best_energy:
                best_energy = energy
                best_x = x_opt.copy()
                
        except Exception:
            continue
            
    if best_x is None:
        best_x = _fibonacci_init(N)
        
    return _project_to_sphere(best_x)
