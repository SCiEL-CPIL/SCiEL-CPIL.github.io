import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 100
DEADLINE = 150.0
SAFETY = 5.0

def _correlation_value(h):
    """Compute the evaluator's value: max(corr) / N * 2."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _project_to_feasible(h):
    """Project h onto feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    for _ in range(100):
        diff = np.sum(h) - N / 2.0
        if abs(diff) < 1e-13:
            break
        if diff > 0:
            mask = h > 1e-14
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
        else:
            mask = h < 1.0 - 1e-14
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    return h

def _low_dim_to_high_dim(z):
    """
    Map low-dimensional control points z (length 10) to high-dimensional h (length 100)
    via piecewise linear interpolation, then project to feasible set.
    """
    # z represents control points at indices 0, 10, 20, ..., 90
    # We interpolate to get 100 points.
    # Control points are at x = [0, 10, 20, ..., 90]
    # We want h at x = [0, 1, ..., 99]
    
    x_ctrl = np.linspace(0, 90, 10)
    x_full = np.arange(100)
    
    # Linear interpolation
    h = np.interp(x_full, x_ctrl, z)
    
    # Project to feasible
    return _project_to_feasible(h)

def _soft_max_objective(h, T):
    """Soft-max approximation of max correlation, scaled for stability."""
    corr = np.correlate(h, 1 - h, mode="full")
    x = corr / T
    x_max = np.max(x)
    shifted = x - x_max
    sum_exp = np.sum(np.exp(shifted))
    lse = x_max + np.log(sum_exp)
    return T * lse

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    rng = np.random.default_rng(100)
    
    # Phase 1: Low-dimensional global search using Differential Evolution
    # Optimize 10 control points that define a piecewise linear curve
    if time.time() < deadline - 40:
        def de_objective_low_dim(z):
            h = _low_dim_to_high_dim(z)
            return _correlation_value(h)
        
        bounds = [(0.0, 1.0)] * 10
        try:
            de_result = differential_evolution(
                de_objective_low_dim,
                bounds=bounds,
                seed=100,
                maxiter=50,
                popsize=20,
                tol=1e-8,
                polish=False,
                init='sobol'
            )
            if de_result.fun < best_val:
                h_de = _low_dim_to_high_dim(de_result.x)
                val_de = _correlation_value(h_de)
                if val_de < best_val:
                    best_val = val_de
                    best_h = h_de.copy()
        except Exception:
            pass
    
    # Phase 2: High-dimensional refinement using L-BFGS-B with soft-max
    if time.time() < deadline - 20:
        for T in [50.0, 100.0, 200.0, 500.0]:
            if time.time() > deadline - 5:
                break
            
            def lbfgs_obj(x):
                x = _project_to_feasible(x)
                return _soft_max_objective(x, T)
            
            try:
                res = minimize(
                    lbfgs_obj,
                    best_h,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-8, 'disp': False}
                )
                h_opt = _project_to_feasible(res.x)
                val_opt = _correlation_value(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Phase 3: Simulated Annealing
    if time.time() < deadline - 10:
        def _anneal_step(h, T, rng):
            n_trials = 5
            best_h_local = h.copy()
            best_val_local = _correlation_value(h)
            
            for _ in range(n_trials):
                n_perturb = rng.integers(1, 5)
                indices = rng.choice(N, size=n_perturb, replace=False)
                
                h_new = h.copy()
                for idx in indices:
                    delta = rng.uniform(-0.1, 0.1)
                    h_new[idx] = np.clip(h_new[idx] + delta, 0, 1)
                
                h_new = _project_to_feasible(h_new)
                val_new = _correlation_value(h_new)
                
                if val_new < best_val_local or rng.random() < np.exp(-(val_new - best_val_local) / T):
                    best_h_local = h_new
                    best_val_local = val_new
            
            return best_h_local, best_val_local
        
        h_sa = best_h.copy()
        T_start = 0.5
        T_end = 0.01
        n_sa_steps = 200
        
        for step in range(n_sa_steps):
            if time.time() > deadline - 5:
                break
            T = T_start * (T_end / T_start) ** (step / n_sa_steps)
            h_sa, val_sa = _anneal_step(h_sa, T, rng)
            if val_sa < best_val:
                best_val = val_sa
                best_h = h_sa.copy()
    
    # Phase 4: Coordinate Descent Polish
    if time.time() < deadline - 3:
        improved = True
        max_rounds = 50
        for _ in range(max_rounds):
            if time.time() > deadline - 2:
                break
            if not improved:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 2:
                    break
                h_trial = best_h.copy()
                for trial_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    h_trial[i] = trial_val
                    h_trial = _project_to_feasible(h_trial)
                    val_trial = _correlation_value(h_trial)
                    if val_trial < best_val - 1e-12:
                        best_val = val_trial
                        best_h = h_trial.copy()
                        improved = True
    
    # Final projection and validation
    best_h = _project_to_feasible(best_h)
    diff = np.sum(best_h) - N / 2.0
    if abs(diff) > 1e-9:
        mask = (best_h > 1e-8) & (best_h < 1.0 - 1e-8)
        if np.any(mask):
            best_h[mask] -= diff / np.sum(mask)
        else:
            best_h -= diff / N
        best_h = np.clip(best_h, 0, 1)
    
    if not np.all(np.isfinite(best_h)):
        best_h = np.full(N, 0.5)
    
    return best_h
