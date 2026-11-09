"""Root finder for 3 fixed two-dimensional systems using a robust hybrid approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _safe_F(F, x):
    """Call F safely, returning a finite array of length 2 or None on failure."""
    try:
        y = F(x)
        y = np.asarray(y, dtype=float)
        if y.shape != (2,) or not np.all(np.isfinite(y)):
            return None
        return y
    except Exception:
        return None


def _broyden_update(J, s, y):
    """Rank-1 Broyden update (good for root finding)."""
    denom = np.dot(y, s)
    if abs(denom) > 1e-14:
        J = J + np.outer(y - J @ s, s) / denom
    return J


def _line_search(F, x, d, Fx, max_steps=20):
    """
    Backtracking line search to find a step alpha such that
    ||F(x + alpha*d)|| < ||F(x)||.
    Returns (x_new, F_new, alpha).
    """
    alpha = 1.0
    norm_Fx = np.linalg.norm(Fx)
    
    for _ in range(max_steps):
        x_try = x + alpha * d
        F_try = _safe_F(F, x_try)
        if F_try is not None:
            norm_F_try = np.linalg.norm(F_try)
            if norm_F_try < norm_Fx:
                return x_try, F_try, alpha
        alpha *= 0.5
        if alpha < 1e-12:
            break
    
    # If no improvement found, return the best tried or original
    return x, Fx, 0.0


def _initial_jacobian(F, x, Fx, h=1e-6):
    """Compute initial Jacobian using finite differences."""
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        F_plus = _safe_F(F, x + e)
        if F_plus is not None:
            J[:, j] = (F_plus - Fx) / h
        else:
            J[:, j] = 1.0  # Fallback
    return J


def find_root_system(F, x0):
    """
    Robust root finder using Broyden's method with line search.
    Falls back to Powell's method if Broyden fails.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    x = np.array(x0, dtype=float)
    Fx = _safe_F(F, x)
    if Fx is None:
        return x
    
    norm_Fx = np.linalg.norm(Fx)
    best_x = x.copy()
    best_norm = norm_Fx
    
    # Phase 1: Broyden's method
    if norm_Fx <= 1e-8:
        return x
    
    J = _initial_jacobian(F, x, Fx)
    
    for iteration in range(80):
        if time.time() > deadline:
            break
            
        # Check convergence
        if np.linalg.norm(Fx) <= 1e-8:
            return x
        
        # Update best
        if np.linalg.norm(Fx) < best_norm:
            best_norm = np.linalg.norm(Fx)
            best_x = x.copy()
        
        # Solve J * d = -Fx
        try:
            d = -np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            # Fallback: use gradient descent on ||F||^2
            d = -J.T @ Fx
            norm_d = np.linalg.norm(d)
            if norm_d < 1e-12:
                break
            d = d / norm_d
        
        norm_d = np.linalg.norm(d)
        if norm_d < 1e-12:
            break
        
        # Line search
        x_new, F_new, alpha = _line_search(F, x, d, Fx)
        
        if alpha == 0.0:
            # No improvement, try smaller step or different direction
            # Use gradient descent step
            d_grad = -J.T @ Fx
            norm_dg = np.linalg.norm(d_grad)
            if norm_dg < 1e-12:
                break
            d_grad = d_grad / norm_dg
            x_new, F_new, alpha = _line_search(F, x, d_grad, Fx, max_steps=10)
            if alpha == 0.0:
                break
        
        # Broyden update
        s = x_new - x
        y = F_new - Fx
        J = _broyden_update(J, s, y)
        
        x = x_new
        Fx = F_new
        
        if np.linalg.norm(Fx) <= 1e-8:
            return x
    
    if np.linalg.norm(Fx) <= 1e-8:
        return x
    
    # Phase 2: Powell's method (derivative-free)
    if time.time() > deadline:
        return best_x
    
    # Initialize directions
    directions = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    
    for powell_iter in range(30):
        if time.time() > deadline:
            break
            
        x_old = x.copy()
        Fx_old = _safe_F(F, x)
        if Fx_old is None:
            break
        norm_old = np.linalg.norm(Fx_old)
        
        if norm_old <= 1e-8:
            return x
        
        if norm_old < best_norm:
            best_norm = norm_old
            best_x = x.copy()
        
        # Directional minimization
        for i in range(len(directions)):
            d = directions[i]
            # Find best alpha along direction
            best_alpha = 0.0
            best_Fx_dir = Fx_old
            best_x_dir = x
            
            # Try a range of alphas
            for alpha in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, -0.5, -1.0, -2.0]:
                x_try = x + alpha * d
                F_try = _safe_F(F, x_try)
                if F_try is not None:
                    norm_try = np.linalg.norm(F_try)
                    if norm_try < np.linalg.norm(best_Fx_dir):
                        best_Fx_dir = F_try
                        best_x_dir = x_try
                        best_alpha = alpha
            
            x = best_x_dir
            Fx = best_Fx_dir
            
            # Update direction to x - x_old
            if best_alpha != 0.0:
                directions[i] = x - x_old
        
        # Check if we made progress
        x_new = x
        Fx_new = _safe_F(F, x_new)
        if Fx_new is None:
            break
        norm_new = np.linalg.norm(Fx_new)
        
        if norm_new <= 1e-8:
            return x_new
        
        if norm_new < best_norm:
            best_norm = norm_new
            best_x = x_new.copy()
        
        # Interpolate new direction
        if norm_new < norm_old:
            # New direction is x_new - x_old
            new_dir = x_new - x_old
            if np.linalg.norm(new_dir) > 1e-12:
                directions = directions[:-1] + [new_dir]
        
        if time.time() > deadline:
            break
    
    # Return best result
    if np.linalg.norm(_safe_F(F, best_x) or np.array([np.inf, np.inf])) <= 1e-8:
        return best_x
    
    # Final check
    F_final = _safe_F(F, best_x)
    if F_final is not None and np.linalg.norm(F_final) <= 1e-8:
        return best_x
    
    return best_x
# EVOLVE-BLOCK-END
