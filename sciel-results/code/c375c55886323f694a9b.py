"""Root finder for 3 fixed 2D systems using a robust multi-strategy approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses a combination of Newton's method with line search,
    and fallback strategies.
    """
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    x = np.array(x0, dtype=float)
    
    def time_up():
        return time.time() > deadline
    
    best_x = x.copy()
    best_norm = float('inf')
    
    def update_best(x_candidate):
        nonlocal best_x, best_norm
        if np.all(np.isfinite(x_candidate)):
            try:
                fc = F(x_candidate)
                if np.all(np.isfinite(fc)):
                    nc = np.linalg.norm(fc)
                    if nc < best_norm:
                        best_norm = nc
                        best_x = x_candidate.copy()
            except Exception:
                pass
    
    # Initial evaluation
    try:
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            raise ValueError("Non-finite F")
        best_norm = np.linalg.norm(Fx)
        best_x = x.copy()
    except Exception:
        # If initial eval fails, try nearby points
        for r in [0.1, 1.0, 10.0]:
            for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
                trial = x + r * np.array([np.cos(angle), np.sin(angle)])
                try:
                    ft = F(trial)
                    if np.all(np.isfinite(ft)):
                        nt = np.linalg.norm(ft)
                        if nt < best_norm:
                            best_norm = nt
                            best_x = trial.copy()
                            x = trial.copy()
                            break
                except Exception:
                    continue
            if best_norm < 1e-8:
                return best_x
        return best_x
    
    if best_norm <= 1e-8:
        return best_x
    
    # Strategy 1: Newton's method with finite-difference Jacobian and line search
    for iteration in range(80):
        if time_up():
            break
        
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            # Try to recover by stepping back
            x = best_x.copy()
            continue
        
        Fx_norm = np.linalg.norm(Fx)
        update_best(x)
        if Fx_norm <= 1e-12:
            return x
        
        # Compute Jacobian using central differences
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            try:
                Fp = F(x + e)
                Fm = F(x - e)
                if np.all(np.isfinite(Fp)) and np.all(np.isfinite(Fm)):
                    J[:, j] = (Fp - Fm) / (2 * h)
                else:
                    # Fall back to forward difference
                    Fp = F(x + e)
                    if np.all(np.isfinite(Fp)):
                        J[:, j] = (Fp - Fx) / h
                    else:
                        J[:, j] = (Fx - F(x - e)) / h
            except Exception:
                # Use forward difference as fallback
                try:
                    Fp = F(x + e)
                    J[:, j] = (Fp - Fx) / h
                except Exception:
                    J[:, j] = 0
        
        if not np.all(np.isfinite(J)):
            # Jacobian is bad, try a different step
            x = x + 0.1 * (np.random.rand(2) - 0.5)
            continue
        
        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Singular matrix, use pseudo-inverse
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                dx = -Fx * 0.01
        
        if not np.all(np.isfinite(dx)):
            dx = -Fx * 0.01
        
        # Line search to ensure sufficient decrease
        step = 1.0
        max_line_iter = 20
        for _ in range(max_line_iter):
            x_new = x + step * dx
            try:
                Fx_new = F(x_new)
                if not np.all(np.isfinite(Fx_new)):
                    step *= 0.5
                    continue
                Fx_new_norm = np.linalg.norm(Fx_new)
                if Fx_new_norm < Fx_norm * (1 - 1e-4 * step):
                    x = x_new
                    break
                else:
                    step *= 0.5
            except Exception:
                step *= 0.5
        
        # If line search failed, take a small step in the negative gradient direction
        if step < 1e-10:
            grad = J.T @ Fx
            if np.all(np.isfinite(grad)) and np.linalg.norm(grad) > 0:
                x = x - 0.01 * grad / np.linalg.norm(grad)
        
        if best_norm <= 1e-8:
            return best_x
    
    if best_norm <= 1e-8:
        return best_x
    
    # Strategy 2: Try Nelder-Mead on ||F||^2 from best point and nearby
    def objective(x_trial):
        try:
            ft = F(x_trial)
            if not np.all(np.isfinite(ft)):
                return 1e10
            return np.dot(ft, ft)
        except Exception:
            return 1e10
    
    # Use scipy's minimize if available
    try:
        from scipy.optimize import minimize
        
        # Try from best_x
        res = minimize(objective, best_x, method='Nelder-Mead',
                      options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-16,
                               'adaptive': True})
        if np.all(np.isfinite(res.x)):
            update_best(res.x)
            if best_norm <= 1e-8:
                return best_x
        
        # Try from a few nearby points
        for r in [0.1, 1.0, 10.0, 100.0]:
            if time_up():
                break
            for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
                if time_up():
                    break
                trial = best_x + r * np.array([np.cos(angle), np.sin(angle)])
                res = minimize(objective, trial, method='Nelder-Mead',
                              options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-16,
                                       'adaptive': True})
                if np.all(np.isfinite(res.x)):
                    update_best(res.x)
                    if best_norm <= 1e-8:
                        return best_x
    except Exception:
        pass
    
    # Strategy 3: Polish with Newton from best point
    x = best_x.copy()
    for iteration in range(50):
        if time_up():
            break
        
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            break
        
        Fx_norm = np.linalg.norm(Fx)
        if Fx_norm <= 1e-12:
            return x
        
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            try:
                Fp = F(x + e)
                Fm = F(x - e)
                if np.all(np.isfinite(Fp)) and np.all(np.isfinite(Fm)):
                    J[:, j] = (Fp - Fm) / (2 * h)
                else:
                    Fp = F(x + e)
                    if np.all(np.isfinite(Fp)):
                        J[:, j] = (Fp - Fx) / h
                    else:
                        J[:, j] = (Fx - F(x - e)) / h
            except Exception:
                try:
                    Fp = F(x + e)
                    J[:, j] = (Fp - Fx) / h
                except Exception:
                    J[:, j] = 0
        
        if not np.all(np.isfinite(J)):
            break
        
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break
        
        if not np.all(np.isfinite(dx)):
            break
        
        x = x + dx
        update_best(x)
        if best_norm <= 1e-8:
            return best_x
    
    return best_x
# EVOLVE-BLOCK-END
