"""Root finder for 3 fixed two-dimensional systems using scipy dogleg trust-region on ||F||^2."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill


def _time_ok():
    return (time.time() - _START_TIME) < _DEADLINE


def _is_finite(arr):
    try:
        arr = np.asarray(arr)
        return arr.shape == (2,) and np.all(np.isfinite(arr))
    except Exception:
        return False


def _norm_F(F, x):
    try:
        Fx = F(x)
        if not _is_finite(Fx):
            return np.inf
        return np.linalg.norm(Fx)
    except Exception:
        return np.inf


def _dogleg_solve(F, x0):
    """Use scipy's dogleg trust-region method on f(x) = 0.5 * ||F(x)||^2."""
    x0 = np.array(x0, dtype=float)
    
    def objective(x):
        try:
            Fx = F(x)
            if not _is_finite(Fx):
                return 1e30
            return 0.5 * np.dot(Fx, Fx)
        except Exception:
            return 1e30
    
    def gradient(x):
        # Compute J^T * F using finite differences
        Fx = F(x)
        if not _is_finite(Fx):
            return np.array([1e10, 1e10])
        eps = 1e-6
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = eps
            Fx_e = F(x + e)
            if not _is_finite(Fx_e):
                Fx_e = Fx + eps * np.ones(2)
            J[:, j] = (Fx_e - Fx) / eps
        return J.T @ Fx
    
    try:
        if not _time_ok():
            return x0
        result = minimize(objective, x0, method='dogleg', jac=gradient,
                          options={'gtol': 1e-12, 'maxiter': 100})
        x_best = result.x if _is_finite(result.x) else x0
        return x_best
    except Exception:
        return x0


def _newton_fd(F, x0, max_iter=40):
    """Newton's method with finite-difference Jacobian."""
    x = np.array(x0, dtype=float)
    calls = 0
    
    for it in range(max_iter):
        if not _time_ok():
            break
        
        Fx = F(x)
        calls += 1
        if not _is_finite(Fx):
            return x
        
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-12:
            return x
        
        # Finite difference Jacobian
        J = np.zeros((2, 2))
        eps = 1e-7
        for j in range(2):
            e = np.zeros(2)
            e[j] = eps
            Fx_e = F(x + e)
            calls += 1
            if not _is_finite(Fx_e):
                Fx_e = Fx + eps * np.ones(2)
            J[:, j] = (Fx_e - Fx) / eps
        
        # Solve for Newton step
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Perturb and retry
            try:
                J[0, 0] += 1e-10
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
        
        # Damped Newton with backtracking
        alpha = 1.0
        for _ in range(20):
            x_new = x + alpha * dx
            Fx_new = F(x_new)
            calls += 1
            if _is_finite(Fx_new):
                if np.linalg.norm(Fx_new) < (1 - 1e-4 * alpha) * norm_Fx:
                    break
            alpha *= 0.5
        
        x = x + alpha * dx
    
    return x


def _nelder_mead(F, x0, max_iter=200):
    """Nelder-Mead simplex method on ||F||^2 as fallback."""
    x0 = np.array(x0, dtype=float)
    
    def objective(x):
        try:
            Fx = F(x)
            if not _is_finite(Fx):
                return 1e30
            return np.dot(Fx, Fx)
        except Exception:
            return 1e30
    
    try:
        if not _time_ok():
            return x0
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-14})
        x_best = result.x if _is_finite(result.x) else x0
        return x_best
    except Exception:
        return x0


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Primary: dogleg trust-region on ||F||^2.
    Fallback: damped Newton with FD Jacobian, then Nelder-Mead.
    """
    x0 = np.array(x0, dtype=float)
    
    # Try dogleg trust-region first
    x_dogleg = _dogleg_solve(F, x0)
    if _is_finite(x_dogleg):
        Fx_check = F(x_dogleg)
        if _is_finite(Fx_check) and np.linalg.norm(Fx_check) <= 1e-8:
            return x_dogleg
    
    # If dogleg didn't converge, try Newton FD
    x_newton = _newton_fd(F, x0)
    if _is_finite(x_newton):
        Fx_check = F(x_newton)
        if _is_finite(Fx_check) and np.linalg.norm(Fx_check) <= 1e-8:
            return x_newton
    
    # Try Nelder-Mead as last resort
    x_nm = _nelder_mead(F, x0)
    if _is_finite(x_nm):
        Fx_check = F(x_nm)
        if _is_finite(Fx_check) and np.linalg.norm(Fx_check) <= 1e-8:
            return x_nm
    
    # Return the best result found
    candidates = []
    for x_cand in [x_dogleg, x_newton, x_nm]:
        if _is_finite(x_cand):
            Fx_c = F(x_cand)
            if _is_finite(Fx_c):
                candidates.append((np.linalg.norm(Fx_c), x_cand))
    
    if candidates:
        candidates.sort(key=lambda t: t[0])
        return candidates[0][1]
    
    return x0
# EVOLVE-BLOCK-END
