"""Root finder for 3 fixed 2D systems using a robust multi-strategy approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import fsolve, minimize


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses scipy.optimize.fsolve as the primary solver, with Newton polishing
    and Nelder-Mead as fallbacks.
    """
    x0 = np.array(x0, dtype=float)
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    # Helper to check if a solution is valid
    def is_valid(x):
        try:
            if not np.all(np.isfinite(x)):
                return False
            Fx = F(x)
            if not np.all(np.isfinite(Fx)):
                return False
            return np.linalg.norm(Fx) <= 1e-8
        except:
            return False

    # Helper for Newton step with line search
    def newton_polish(x, max_iter=10):
        for _ in range(max_iter):
            if time.time() > deadline:
                return x
            Fx = F(x)
            if np.linalg.norm(Fx) <= 1e-8:
                return x
            
            # Finite difference Jacobian
            h = 1e-6
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                # Central difference for better accuracy
                J[:, j] = (F(x + e) - F(x - e)) / (2 * h)
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # If singular, try a small random step or break
                break
            
            # Line search to ensure decrease in ||F(x)||
            norm_current = np.linalg.norm(Fx)
            t = 1.0
            success = False
            for _ in range(10):
                x_new = x + t * dx
                try:
                    Fx_new = F(x_new)
                    norm_new = np.linalg.norm(Fx_new)
                except:
                    break
                if norm_new < norm_current:
                    x = x_new
                    success = True
                    break
                t *= 0.5
            
            if not success:
                # If line search failed, try a smaller fixed step or break
                break
        
        return x

    # 1. Try scipy.optimize.fsolve
    try:
        sol, info, ier, msg = fsolve(F, x0, xtol=1e-12, maxfev=100, full_output=True)
        if ier == 1:
            if is_valid(sol):
                return sol
        # If fsolve didn't converge fully, polish it
        sol = newton_polish(sol, max_iter=5)
        if is_valid(sol):
            return sol
        best_x = sol
        best_norm = np.linalg.norm(F(sol))
    except:
        best_x = x0
        best_norm = np.linalg.norm(F(x0))

    # 2. Try Nelder-Mead on ||F(x)||^2
    def objective(x):
        try:
            Fx = F(x)
            return 0.5 * np.dot(Fx, Fx)
        except:
            return 1e100

    # Try from x0 and from the fsolve result
    for start in [x0, best_x]:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, start, method='Nelder-Mead', 
                           options={'xatol': 1e-10, 'fatol': 1e-16, 'maxiter': 200, 'adaptive': True})
            if res.success or res.fun < best_norm:
                x_nm = res.x
                if is_valid(x_nm):
                    return x_nm
                # Polish the Nelder-Mead result
                x_polished = newton_polish(x_nm, max_iter=10)
                if is_valid(x_polished):
                    return x_polished
                if np.linalg.norm(F(x_polished)) < best_norm:
                    best_x = x_polished
                    best_norm = np.linalg.norm(F(x_polished))
        except:
            continue

    # 3. Final attempt: Newton from best_x
    x_final = newton_polish(best_x, max_iter=20)
    if is_valid(x_final):
        return x_final

    # Return the best result found, even if it doesn't meet the tolerance
    # (This will result in a failure for that system, but avoids an exception)
    return best_x

# EVOLVE-BLOCK-END
