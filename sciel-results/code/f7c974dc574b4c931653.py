def _newton_polish(pts, deadline, max_iter=50, tol=1e-12):
    n = pts.shape[0]
    current = pts.copy()
    
    for it in range(max_iter):
        if time.time() > deadline - 1.0:
            break
            
        # Get ambient gradient
        g_amb = _energy_ambient_grad(current)
        g_norm = np.linalg.norm(g_amb)
        if g_norm < tol:
            break
            
        # Build Hessian (Ambient)
        H = np.zeros((3*n, 3*n))
        # ... (same as before) ...
        
        # Solve H delta = g_amb  (since we want to move against gradient, delta = H^-1 g)
        # Wait. Newton method for minimization: x_new = x - H^-1 grad.
        # So delta = - H^-1 g_amb.
        try:
            H_reg = H + 1e-12 * np.eye(3*n)
            delta = -np.linalg.solve(H_reg, g_amb)
        except np.linalg.LinAlgError:
            delta = -0.01 * g_amb
            
        # Project delta to tangent space
        dot = np.sum(delta * current, axis=1, keepdims=True)
        delta_tan = delta - dot * current
        
        # Line search
        e_curr = _energy(current)
        alpha = 1.0
        for _ in range(20):
            new_pts = current + alpha * delta_tan
            new_pts = _project(new_pts)
            e_new = _energy(new_pts)
            if e_new < e_curr:
                current = new_pts
                break
            alpha *= 0.5
            
    return current
