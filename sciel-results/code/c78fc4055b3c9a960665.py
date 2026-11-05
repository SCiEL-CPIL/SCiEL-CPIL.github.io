<<<<<<< SEARCH
import time
import math
import numpy as np
import random
=======
import time
import math
import numpy as np
import random
import scipy.optimize as so
>>>>>>> REPLACE
<<<<<<< SEARCH
def _optimize_nelder_mead(objective, x0, lo, hi, max_total, deadline, max_iter=300, tol=1e-7):
=======
def _optimize_lbfgs(objective, x0, lo, hi, max_total, deadline, max_iter=200):
    """
    L-BFGS-B optimization with bound and total-thickness constraints.
    """
    x0 = np.array(x0, dtype=float)
    x0 = _project_to_feasible(x0, lo, hi, max_total)
    
    n = len(x0)
    
    # Define constraints: total thickness <= max_total
    def constraint_total(x):
        return max_total - np.sum(x)
    
    constraints = [{'type': 'ineq', 'fun': constraint_total}]
    
    bounds = list(zip(lo, hi))
    
    try:
        result = so.minimize(
            objective, x0, method='L-BFGS-B',
            bounds=bounds, constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        x_opt = _project_to_feasible(result.x, lo, hi, max_total)
        f_opt = objective(x_opt)
        return x_opt, f_opt
    except Exception:
        # Fallback to Nelder-Mead if L-BFGS fails
        return _optimize_nelder_mead(objective, x0, lo, hi, max_total, deadline, max_iter=max_iter)

def _optimize_nelder_mead(objective, x0, lo, hi, max_total, deadline, max_iter=300, tol=1e-7):
>>>>>>> REPLACE
<<<<<<< SEARCH
        # First do coordinate descent for quick improvement
        x_cd, f_cd = _coordinate_descent(objective, x0, lo, hi, max_total, deadline, max_rounds=30)
        
        # Then Nelder-Mead for refinement
        x_nm, f_nm = _optimize_nelder_mead(objective, x_cd, lo, hi, max_total, deadline, max_iter=200)
        
        # Use the better of the two
        x_best = x_nm if f_nm <= f_cd else x_cd
        f_best = min(f_nm, f_cd)
        
        update_best(mats, x_best)
        
        # Also try the CD result separately in case NM got stuck
        if f_cd < f_nm:
            update_best(mats, x_cd)
=======
        # First do coordinate descent for quick improvement
        x_cd, f_cd = _coordinate_descent(objective, x0, lo, hi, max_total, deadline, max_rounds=30)
        
        # Then L-BFGS for faster convergence
        x_lbfgs, f_lbfgs = _optimize_lbfgs(objective, x_cd, lo, hi, max_total, deadline, max_iter=200)
        
        # Then Nelder-Mead for final refinement
        x_nm, f_nm = _optimize_nelder_mead(objective, x_lbfgs, lo, hi, max_total, deadline, max_iter=100)
        
        # Use the best of all three
        candidates_opt = [(x_cd, f_cd), (x_lbfgs, f_lbfgs), (x_nm, f_nm)]
        x_best, f_best = min(candidates_opt, key=lambda x: x[1])
        
        update_best(mats, x_best)
        
        # Also try the other results in case of local minima
        for x_cand, f_cand in candidates_opt:
            if f_cand < f_best:
                update_best(mats, x_cand)
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Additional targeted search: refine the best found so far
    if best_mats is not None and time.time() < deadline - 10:
        n_layers = len(best_mats)
        lo = np.zeros(n_layers)
        hi = np.full(n_layers, max_layer)
        
        def objective(th):
            return _mean_reflectance_percent(best_mats, list(th), problem, wls, n_mat)
        
        # Multiple restarts from perturbed best
        for _ in range(5):
            if time.time() > deadline:
                break
            x0 = np.array(best_ths, dtype=float).copy()
            # Perturb
            for i in range(n_layers):
                if random.random() < 0.5:
                    x0[i] += random.uniform(-0.1, 0.1) * (hi[i] - lo[i])
                    x0[i] = np.clip(x0[i], lo[i], hi[i])
            
            x_opt, f_opt = _optimize_nelder_mead(objective, x0, lo, hi, max_total, deadline, max_iter=150)
            update_best(best_mats, x_opt)
=======
    # Additional targeted search: refine the best found so far
    if best_mats is not None and time.time() < deadline - 10:
        n_layers = len(best_mats)
        lo = np.zeros(n_layers)
        hi = np.full(n_layers, max_layer)
        
        def objective(th):
            return _mean_reflectance_percent(best_mats, list(th), problem, wls, n_mat)
        
        # Multiple restarts from perturbed best
        for _ in range(8):
            if time.time() > deadline:
                break
            x0 = np.array(best_ths, dtype=float).copy()
            # Perturb
            for i in range(n_layers):
                if random.random() < 0.5:
                    x0[i] += random.uniform(-0.1, 0.1) * (hi[i] - lo[i])
                    x0[i] = np.clip(x0[i], lo[i], hi[i])
            
            # Use L-BFGS for faster convergence
            x_opt, f_opt = _optimize_lbfgs(objective, x0, lo, hi, max_total, deadline, max_iter=150)
            update_best(best_mats, x_opt)
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Fallback
    if best_mats is None:
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        update_best(best_mats, best_ths)
    
    # Final constraint check
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    s = sum(best_ths)
    if s > max_total + 1e-6:
        f = max_total / s
        best_ths = [t * f for t in best_ths]
        best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    
    return list(best_mats), list(best_ths)
=======
    # Fallback
    if best_mats is None:
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        update_best(best_mats, best_ths)
    
    # Final high-precision polish: run L-BFGS with tighter tolerance on the best solution
    if best_mats is not None and time.time() < deadline - 5:
        n_layers = len(best_mats)
        lo = np.zeros(n_layers)
        hi = np.full(n_layers, max_layer)
        
        def objective(th):
            return _mean_reflectance_percent(best_mats, list(th), problem, wls, n_mat)
        
        x_polish, f_polish = _optimize_lbfgs(objective, np.array(best_ths), lo, hi, max_total, deadline, max_iter=500)
        if f_polish < best_val:
            best_mats = list(best_mats)
            best_ths = [min(max(t, 0.0), max_layer) for t in x_polish]
            s = sum(best_ths)
            if s > max_total + 1e-6:
                f = max_total / s
                best_ths = [t * f for t in best_ths]
                best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    
    # Final constraint check
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    s = sum(best_ths)
    if s > max_total + 1e-6:
        f = max_total / s
        best_ths = [t * f for t in best_ths]
        best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    
    return list(best_mats), list(best_ths)
>>>>>>> REPLACE
BET: {mechanism: "L-BFGS converges faster than Nelder-Mead for smooth objectives like reflectance, allowing more diverse initial designs to be optimized within the time budget and enabling a final high-precision polish pass with tighter tolerance", q: 0.7, delta: [1.0, 3.0]}
