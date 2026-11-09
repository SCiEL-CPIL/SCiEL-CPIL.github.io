"""Root finder for 3 fixed 2D systems using derivative-free quadratic model optimization."""
# EVOLVE-BLOCK-START
import numpy as np


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Newton's method with finite-difference Jacobian for fast local convergence.
    2. If Newton fails, use a derivative-free quadratic model method that exploits 2D structure.
    3. The quadratic model is updated with 3 evaluations per iteration (one at current point,
       two along coordinate directions) to estimate gradient and Hessian.
    4. Steepest descent direction from the quadratic model with aggressive line search.
    """
    x = np.array(x0, dtype=float)
    
    # Phase 1: Newton's method with finite-difference Jacobian
    max_newton = 30
    for i in range(max_newton):
        Fx = F(x)
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-12:
            return x
        
        # Compute Jacobian via central differences
        eps = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = eps
            J[:, j] = (F(x + e) - F(x - e)) / (2 * eps)
        
        try:
            dx = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            break
        
        if not np.all(np.isfinite(dx)):
            break
        
        # Damped Newton with backtracking
        alpha = 1.0
        for _ in range(25):
            x_new = x - alpha * dx
            F_new = F(x_new)
            if np.linalg.norm(F_new) < norm_Fx:
                x = x_new
                break
            alpha *= 0.5
        else:
            break
    
    Fx = F(x)
    if np.linalg.norm(Fx) <= 1e-8:
        return x
    
    # Phase 2: Derivative-free quadratic model method
    # g(x) = ||F(x)||^2
    # We maintain a local quadratic model q(x) = c + grad^T (x - x0) + 0.5 (x-x0)^T H (x-x0)
    # In 2D, we can estimate grad and H from evaluations at x0, x0+e1*h, x0+e2*h
    
    def g(x_val):
        Fv = F(x_val)
        return float(np.dot(Fv, Fv))
    
    # Start from current best
    x0_local = x.copy()
    g0 = g(x0_local)
    best_x = x0_local
    best_g = g0
    
    # Adaptive step size for finite differences
    h = max(1e-4, 0.05 * max(1.0, np.linalg.norm(x0_local)))
    
    max_iter = 300
    for it in range(max_iter):
        if best_g <= 1e-16:
            return best_x
        
        # Estimate gradient and Hessian using 3 points: x0, x0+h*e1, x0+h*e2
        # g(x0) = g0
        # g(x0 + h*e1) = g0 + h*grad[0] + 0.5*h^2*H[0,0]
        # g(x0 + h*e2) = g0 + h*grad[1] + 0.5*h^2*H[1,1]
        # This gives us grad[0], grad[1] but only diagonal of H.
        # For off-diagonal, we need a 4th point or use a different approach.
        
        # Better: use points x0, x0+h*e1, x0+h*e2, x0+h*(e1+e2)
        # But that's 4 evaluations. Let's use 3 with a smarter model.
        
        # Alternative: Use the fact that in 2D, we can do a quadratic fit with 3 points
        # if we assume the Hessian is symmetric and use a specific parameterization.
        
        # Let's use a simpler approach: estimate gradient with 2 extra evaluations,
        # then do steepest descent with line search.
        
        # Estimate gradient at x0_local
        eps_fd = max(1e-6, 1e-4 * max(1.0, np.linalg.norm(x0_local)))
        
        # Central difference gradient
        F0 = F(x0_local)
        g0_val = float(np.dot(F0, F0))
        
        # For gradient of g = ||F||^2, grad_g = 2 * J^T F
        # So we need J. Let's use forward differences for gradient estimation.
        grad_g = np.zeros(2)
        for j in range(2):
            e = np.zeros(2)
            e[j] = eps_fd
            Fp = F(x0_local + e)
            gp = float(np.dot(Fp, Fp))
            grad_g[j] = (gp - g0_val) / eps_fd
        
        # If gradient is near zero, we're close to a minimum
        if np.linalg.norm(grad_g) < 1e-12:
            # Try to refine with Newton
            eps_j = 1e-7
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = eps_j
                J[:, j] = (F(x0_local + e) - F(x0_local - e)) / (2 * eps_j)
            try:
                dx = np.linalg.solve(J, F0)
                if np.all(np.isfinite(dx)):
                    x0_local = x0_local - dx
                    g0_val = g(x0_local)
                    if g0_val < best_g:
                        best_g = g0_val
                        best_x = x0_local.copy()
                    if best_g <= 1e-16:
                        return best_x
            except np.linalg.LinAlgError:
                pass
            continue
        
        # Steepest descent direction
        d = -grad_g
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-15:
            break
        d = d / d_norm
        
        # Line search: try several step sizes
        best_alpha = 0.0
        best_g_alpha = g0_val
        
        # Try a range of step sizes
        alphas = [1e-2, 1e-1, 0.5, 1.0, 2.0, 5.0, 10.0]
        for a in alphas:
            x_try = x0_local + a * d * d_norm  # Scale by gradient magnitude
            g_try = g(x_try)
            if g_try < best_g_alpha:
                best_g_alpha = g_try
                best_alpha = a
        
        if best_alpha == 0.0:
            # No improvement found, try smaller steps
            for a in [1e-4, 1e-5, 1e-6]:
                x_try = x0_local + a * d * d_norm
                g_try = g(x_try)
                if g_try < best_g_alpha:
                    best_g_alpha = g_try
                    best_alpha = a
        
        if best_alpha == 0.0:
            # Stuck, break
            break
        
        x0_local = x0_local + best_alpha * d * d_norm
        g0_val = best_g_alpha
        
        if g0_val < best_g:
            best_g = g0_val
            best_x = x0_local.copy()
        
        if best_g <= 1e-16:
            return best_x
        
        # Adaptive step size update
        if best_alpha < 0.1:
            h *= 0.5
        elif best_alpha > 1.0:
            h *= 1.5
        h = max(1e-6, min(h, 10.0))
    
    return best_x
# EVOLVE-BLOCK-END
