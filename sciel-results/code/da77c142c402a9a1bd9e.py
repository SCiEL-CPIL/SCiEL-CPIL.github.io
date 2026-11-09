import numpy as np
from scipy.optimize import minimize, root

def find_root_system(F, x0):
    x0 = np.asarray(x0, dtype=float)
    
    # Check initial point
    Fx0 = F(x0)
    if np.linalg.norm(Fx0) <= 1e-8:
        return x0

    def objective(x):
        Fx = F(x)
        return np.sum(Fx**2)

    best_x = x0
    best_norm = np.linalg.norm(Fx0)

    # Strategy 1: Trust-region Dogleg (Derivative-free via internal FD)
    try:
        res = minimize(
            objective,
            x0,
            method='dogleg',
            options={
                'maxiter': 100,
                'gtol': 1e-8,
                'xtol': 1e-12,
                'ftol': 1e-16,
            }
        )
        x = res.x
        Fx = F(x)
        norm = np.linalg.norm(Fx)
        if norm < best_norm:
            best_x = x
            best_norm = norm
        if norm <= 1e-8:
            return x
    except Exception:
        pass

    # Strategy 2: Trust-region NCG (More robust than Dogleg for non-quadratic)
    try:
        res = minimize(
            objective,
            x0,
            method='trust-ncg',
            options={
                'maxiter': 100,
                'gtol': 1e-8,
                'xtol': 1e-12,
                'ftol': 1e-16,
            }
        )
        x = res.x
        Fx = F(x)
        norm = np.linalg.norm(Fx)
        if norm < best_norm:
            best_x = x
            best_norm = norm
        if norm <= 1e-8:
            return x
    except Exception:
        pass

    # Strategy 3: Levenberg-Marquardt (Root finding directly)
    try:
        res = root(
            F,
            x0,
            method='hybr',
            options={'maxiter': 100, 'xtol': 1e-12}
        )
        x = res.x
        Fx = F(x)
        norm = np.linalg.norm(Fx)
        if norm < best_norm:
            best_x = x
            best_norm = norm
        if norm <= 1e-8:
            return x
    except Exception:
        pass

    # Strategy 4: Broyden (Derivative-free root finding)
    try:
        res = root(
            F,
            x0,
            method='broyden1',
            options={'maxiter': 100, 'xtol': 1e-12}
        )
        x = res.x
        Fx = F(x)
        norm = np.linalg.norm(Fx)
        if norm < best_norm:
            best_x = x
            best_norm = norm
        if norm <= 1e-8:
            return x
    except Exception:
        pass

    # Fallback: Simple Nelder-Mead (Global-ish, robust but slow)
    try:
        res = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-16}
        )
        x = res.x
        Fx = F(x)
        norm = np.linalg.norm(Fx)
        if norm < best_norm:
            best_x = x
            best_norm = norm
        if norm <= 1e-8:
            return x
    except Exception:
        pass

    return best_x
