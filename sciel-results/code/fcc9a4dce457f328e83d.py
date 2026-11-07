"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100
DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time as _time

def _compute_value(h):
    """Compute the evaluator's value."""
    c = np.correlate(h, 1 - h, mode="full")
    return np.max(c) / len(h) * 2

def _objective(x):
    """Objective: maximize t such that all correlations <= t, then we minimize t.
    We'll use a smooth approximation or direct minimization of max correlation."""
    h = x
    c = np.correlate(h, 1 - h, mode="full")
    return np.max(c)

def construct():
    start_time = _time.time()
    best_h = np.full(N, 0.5)
    best_val = _compute_value(best_h)
    
    # Try several initial patterns
    patterns = [
        np.full(N, 0.5),
        # Binary: alternating blocks
        np.concatenate([np.ones(50), np.zeros(50)]),
        np.concatenate([np.zeros(50), np.ones(50)]),
        # Periodic patterns
        np.tile([1, 1, 0, 1, 1, 0], N // 6 + 1)[:N],
        np.tile([1, 0, 1, 0], N // 4 + 1)[:N],
        # Gradient
        np.linspace(1, 0, N),
        np.linspace(0, 1, N),
        # Symmetric
        np.concatenate([np.linspace(1, 0, 50), np.linspace(0, 1, 50)]),
    ]
    
    # Normalize each pattern to sum = 50
    for p in patterns:
        p = np.clip(p, 0, 1)
        if p.sum() > 0:
            p = p * (50.0 / p.sum())
            p = np.clip(p, 0, 1)
            # Re-normalize after clipping
            for _ in range(10):
                if p.sum() > 0:
                    p = p * (50.0 / p.sum())
                    p = np.clip(p, 0, 1)
                else:
                    break
            if abs(p.sum() - 50) < 1e-9 * N:
                val = _compute_value(p)
                if val < best_val:
                    best_val = val
                    best_h = p.copy()
    
    # Now try to improve with scipy optimization
    elapsed = _time.time() - start_time
    if elapsed < DEADLINE - 30:  # leave 30s buffer
        try:
            # Use a smooth max approximation: log-sum-exp
            def smooth_obj(x, beta=50):
                h = x
                c = np.correlate(h, 1 - h, mode="full")
                # Smooth max
                c_max = np.max(c)
                if beta <= 0:
                    return c_max
                return (1.0 / beta) * np.log(np.sum(np.exp(beta * c / c_max if c_max > 0 else c)))
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 50.0},
            ]
            bounds = [(0.0, 1.0)] * N
            
            best_x = best_h.copy()
            best_obj = _compute_value(best_x)
            
            # Try SLSQP
            result = minimize(
                lambda x: np.max(np.correlate(x, 1 - x, mode="full")),
                best_x,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
            )
            
            elapsed = _time.time() - start_time
            if result.x is not None and result.x.shape == (N,):
                h_opt = np.clip(result.x, 0, 1)
                # Renormalize
                for _ in range(50):
                    s = h_opt.sum()
                    if s > 0 and abs(s - 50) > 1e-12:
                        h_opt = h_opt * (50.0 / s)
                        h_opt = np.clip(h_opt, 0, 1)
                    else:
                        break
                if abs(h_opt.sum() - 50) < 1e-9 * N:
                    val = _compute_value(h_opt)
                    if val < best_obj:
                        best_obj = val
                        best_x = h_opt.copy()
            
            # Try a few more local optimizations from different starts
            elapsed = _time.time() - start_time
            if elapsed < DEADLINE - 15:
                for i in range(3):
                    if _time.time() - start_time > DEADLINE - 15:
                        break
                    # Perturb best solution
                    rng = np.random.default_rng(42 + i)
                    x0 = best_x + rng.normal(0, 0.1, N)
                    x0 = np.clip(x0, 0, 1)
                    for _ in range(50):
                        s = x0.sum()
                        if s > 0:
                            x0 = x0 * (50.0 / s)
                            x0 = np.clip(x0, 0, 1)
                        else:
                            break
                    if abs(x0.sum() - 50) >= 1e-9 * N:
                        continue
                    
                    res = minimize(
                        lambda x: np.max(np.correlate(x, 1 - x, mode="full")),
                        x0,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 100, 'ftol': 1e-10, 'disp': False}
                    )
                    if res.x is not None and res.x.shape == (N,):
                        h_new = np.clip(res.x, 0, 1)
                        for _ in range(50):
                            s = h_new.sum()
                            if s > 0 and abs(s - 50) > 1e-12:
                                h_new = h_new * (50.0 / s)
                                h_new = np.clip(h_new, 0, 1)
                            else:
                                break
                        if abs(h_new.sum() - 50) < 1e-9 * N:
                            val = _compute_value(h_new)
                            if val < best_obj:
                                best_obj = val
                                best_x = h_new.copy()
        except Exception:
            pass
    
    # Final validation
    best_h = np.clip(best_h, 0, 1)
    for _ in range(100):
        s = best_h.sum()
        if s > 0 and abs(s - 50) > 1e-12:
            best_h = best_h * (50.0 / s)
            best_h = np.clip(best_h, 0, 1)
        else:
            break
    
    if abs(best_h.sum() - 50) >= 1e-9 * N:
        # Fallback to seed
        best_h = np.full(N, 0.5)
    
    if not np.all(np.isfinite(best_h)):
        best_h = np.full(N, 0.5)
    
    return best_h
# EVOLVE-BLOCK-END
