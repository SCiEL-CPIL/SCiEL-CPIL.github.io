"""Root finder for 3 fixed two-dimensional systems using Broyden's method with safeguards."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0


def _time_ok():
    return (time.time() - _START_TIME) < _DEADLINE


def _is_finite(arr):
    try:
        arr = np.asarray(arr, dtype=float)
        return arr.shape == (2,) and np.all(np.isfinite(arr))
    except Exception:
        return False


def _fd_jacobian(F, x, Fx=None):
    """Compute finite-difference Jacobian using central differences."""
    J = np.zeros((2, 2))
    h = 1e-6
    if Fx is None:
        Fx = F(x)
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_plus = F(x + e)
        Fx_minus = F(x - e)
        J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
    return J


def _broyden(F, x0, max_iter=100):
    """Broyden's method (good-Broyden) for 2D root finding."""
    x = np.array(x0, dtype=float)
    Fx = F(x)
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx)
    
    if best_norm <= 1e-10:
        return x
    
    # Initial Jacobian via finite differences
    J = _fd_jacobian(F, x, Fx)
    
    for it in range(max_iter):
        if not _time_ok():
            break
        
        # Check convergence
        if best_norm <= 1e-10:
            break
        
        # Solve J dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If J is singular, recompute Jacobian
            J = _fd_jacobian(F, x, Fx)
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
        
        # Line search: try full step first, then damped
        x_new = x + dx
        Fx_new = F(x_new)
        norm_new = np.linalg.norm(Fx_new)
        
        if not np.all(np.isfinite(x_new)) or not np.all(np.isfinite(Fx_new)):
            # Step went bad, try smaller step
            for damp in [0.5, 0.25, 0.1]:
                x_new = x + damp * dx
                Fx_new = F(x_new)
                norm_new = np.linalg.norm(Fx_new)
                if np.all(np.isfinite(x_new)) and np.all(np.isfinite(Fx_new)):
                    break
            if not np.all(np.isfinite(x_new)):
                break
        
        if norm_new < best_norm:
            # Update Broyden approximation
            s = x_new - x
            y = Fx_new - Fx
            J = J + np.outer(y - J @ s, s) / (s @ s)
            x = x_new
            Fx = Fx_new
            best_norm = norm_new
            best_x = x.copy()
        else:
            # Try damped step
            accepted = False
            for damp in [0.5, 0.25, 0.1, 0.05]:
                x_trial = x + damp * dx
                Fx_trial = F(x_trial)
                norm_trial = np.linalg.norm(Fx_trial)
                if np.all(np.isfinite(x_trial)) and np.all(np.isfinite(Fx_trial)) and norm_trial < best_norm:
                    s = x_trial - x
                    y = Fx_trial - Fx
                    J = J + np.outer(y - J @ s, s) / (s @ s)
                    x = x_trial
                    Fx = Fx_trial
                    best_norm = norm_trial
                    best_x = x.copy()
                    accepted = True
                    break
            if not accepted:
                # Try recomputing Jacobian from scratch
                J = _fd_jacobian(F, x, Fx)
                # If still stuck, break
                if norm_new >= best_norm:
                    break
    
    return best_x


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method with line search and safeguards.
    """
    x0 = np.array(x0, dtype=float)
    
    # Try Broyden's method first
    x = _broyden(F, x0, max_iter=80)
    
    if _is_finite(x):
        Fx = F(x)
        if np.all(np.isfinite(Fx)) and np.linalg.norm(Fx) <= 1e-8:
            return x
    
    # If Broyden didn't converge tightly enough, try Newton with FD Jacobian as fallback
    x = np.array(x0, dtype=float)
    for _ in range(40):
        if not _time_ok():
            break
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            break
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-10:
            break
        J = _fd_jacobian(F, x, Fx)
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break
        x_new = x + dx
        if not np.all(np.isfinite(x_new)):
            break
        Fx_new = F(x_new)
        if not np.all(np.isfinite(Fx_new)):
            break
        if np.linalg.norm(Fx_new) < norm_Fx:
            x = x_new
        else:
            # Damped
            for damp in [0.5, 0.25]:
                x_trial = x + damp * dx
                Fx_trial = F(x_trial)
                if np.all(np.isfinite(x_trial)) and np.all(np.isfinite(Fx_trial)) and np.linalg.norm(Fx_trial) < norm_Fx:
                    x = x_trial
                    break
            else:
                break
    
    if _is_finite(x):
        Fx = F(x)
        if np.all(np.isfinite(Fx)) and np.linalg.norm(Fx) <= 1e-8:
            return x
    
    # Return best effort
    return x
# EVOLVE-BLOCK-END
