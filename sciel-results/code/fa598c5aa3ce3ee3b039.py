"""Root finder for 3 fixed 2D systems using a multi-strategy approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import fsolve, minimize


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses a combination of strategies:
    1. Try scipy.optimize.fsolve (hybrid Powell method) from x0.
    2. If that fails or is too slow, try Nelder-Mead on ||F||^2 from x0 and nearby points.
    3. Polish with Newton's method (finite-difference Jacobian) from the best candidate.
    4. Keep track of the best result found and return it before the deadline.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill
    
    def time_up():
        return time.time() >= deadline
    
    x0 = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0)):
        return x0
    
    best_x = None
    best_norm = np.inf
    
    def update_best(x, Fx=None):
        nonlocal best_x, best_norm
        if Fx is None:
            try:
                Fx = F(x)
            except Exception:
                return
        if not np.all(np.isfinite(Fx)):
            return
        n = np.linalg.norm(Fx)
        if n < best_norm:
            best_norm = n
            best_x = x.copy()
    
    # Check initial point
    try:
        Fx0 = F(x0)
        if np.all(np.isfinite(Fx0)):
            n0 = np.linalg.norm(Fx0)
            if n0 <= 1e-8:
                return x0
            best_norm = n0
            best_x = x0.copy()
    except Exception:
        pass
    
    # Strategy 1: scipy.optimize.fsolve from x0
    try:
        if not time_up():
            sol, info, ier, msg = fsolve(F, x0, full_output=True, maxfev=200)
            if np.all(np.isfinite(sol)):
                Fsol = F(sol)
                if np.all(np.isfinite(Fsol)):
                    nsol = np.linalg.norm(Fsol)
                    if nsol < best_norm:
                        best_norm = nsol
                        best_x = sol.copy()
                    if nsol <= 1e-8:
                        return sol
    except Exception:
        pass
    
    # Strategy 2: Nelder-Mead on ||F||^2 from several starting points
    def objective(x):
        try:
            Fx = F(x)
            if not np.all(np.isfinite(Fx)):
                return 1e10
            return float(np.dot(Fx, Fx))
        except Exception:
            return 1e10
    
    if not time_up() and best_norm > 1e-8:
        # Generate candidate starting points
        candidates = [x0.copy()]
        # Add some perturbed points
        for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
            for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
                offset = np.array([scale * np.cos(angle), scale * np.sin(angle)])
                candidates.append(x0 + offset)
        
        for c in candidates:
            if time_up():
                break
            if not np.all(np.isfinite(c)):
                continue
            try:
                res = minimize(objective, c, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-16})
                if res.x is not None and np.all(np.isfinite(res.x)):
                    Fx_res = F(res.x)
                    if np.all(np.isfinite(Fx_res)):
                        n_res = np.linalg.norm(Fx_res)
                        if n_res < best_norm:
                            best_norm = n_res
                            best_x = res.x.copy()
                        if n_res <= 1e-8:
                            return res.x
            except Exception:
                pass
    
    # Strategy 3: Polish with Newton's method from the best candidate
    if best_x is not None and best_norm > 1e-8 and not time_up():
        x = best_x.copy()
        for _ in range(30):
            if time_up():
                break
            try:
                Fx = F(x)
                if not np.all(np.isfinite(Fx)):
                    break
                n = np.linalg.norm(Fx)
                if n < best_norm:
                    best_norm = n
                    best_x = x.copy()
                if n <= 1e-8:
                    return x
                
                # Finite-difference Jacobian
                h = 1e-6
                J = np.zeros((2, 2))
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = h
                    Fx_plus = F(x + e)
                    if not np.all(np.isfinite(Fx_plus)):
                        break
                    J[:, j] = (Fx_plus - Fx) / h
                else:
                    try:
                        dx = np.linalg.solve(J, Fx)
                    except np.linalg.LinAlgError:
                        break
                    # Damped step
                    step = dx
                    for alpha in [1.0, 0.5, 0.25, 0.1, 0.05]:
                        x_new = x - alpha * step
                        if not np.all(np.isfinite(x_new)):
                            continue
                        Fx_new = F(x_new)
                        if not np.all(np.isfinite(Fx_new)):
                            continue
                        n_new = np.linalg.norm(Fx_new)
                        if n_new < n:
                            x = x_new
                            break
                    else:
                        break
            except Exception:
                break
    
    # Return best result found
    if best_x is not None:
        return best_x
    return x0
# EVOLVE-BLOCK-END
