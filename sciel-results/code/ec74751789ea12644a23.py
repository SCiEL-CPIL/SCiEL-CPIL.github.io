import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_value(h):
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return 0.0
    h = np.maximum(h, 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate([[0.0], g, [0.0]])
    w = 1.0 / (m + 1)
    l2 = (w / 3.0) * np.sum(y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf < 1e-15:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def negative_value(params):
    h = np.exp(params)  # ensures nonnegativity
    return -compute_value(h)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit
    
    best_h = None
    best_val = -np.inf
    
    # Initial guesses
    initial_guesses = []
    
    # 1. Flat
    initial_guesses.append(np.ones(N))
    
    # 2. Hann window
    t = np.linspace(0, 1, N)
    initial_guesses.append(0.5 * (1 - np.cos(2 * np.pi * t)))
    
    # 3. Gaussian (narrow)
    x = np.arange(N)
    initial_guesses.append(np.exp(-((x - N/2) ** 2) / (2 * 8.0 ** 2)))
    
    # 4. Gaussian (medium)
    initial_guesses.append(np.exp(-((x - N/2) ** 2) / (2 * 12.0 ** 2)))
    
    # 5. Gaussian (wide)
    initial_guesses.append(np.exp(-((x - N/2) ** 2) / (2 * 18.0 ** 2)))
    
    # 6. Triangular
    initial_guesses.append(1.0 - np.abs(x - (N-1)/2) / ((N-1)/2))
    
    # 7. Cosine bell
    initial_guesses.append(np.maximum(0.0, np.cos(np.pi * (x - (N-1)/2) / (N-1))))
    
    # 8. Sine raised
    initial_guesses.append(np.sin(np.pi * (x + 0.5) / N))
    
    # Evaluate initial guesses
    for h0 in initial_guesses:
        h0 = np.maximum(h0, 0.0)
        if not np.all(np.isfinite(h0)):
            continue
        val = compute_value(h0)
        if val > best_val:
            best_val = val
            best_h = h0.copy()
    
    # Log-transform for optimization: optimize over log(h) to ensure h >= 0
    # We'll use L-BFGS-B or SLSQP with bounds on the log variables
    # Actually, let's just optimize h directly with bounds h >= 0 using SLSQP
    
    # Use the best initial guess as starting point for local optimization
    if best_h is not None:
        h_start = best_h.copy()
        # Clip to positive for safety
        h_start = np.maximum(h_start, 1e-6)
        
        # Define objective and gradient
        def obj(params):
            h = np.maximum(params, 0.0)
            return -compute_value(h)
        
        # Try SLSQP
        try:
            bounds = [(0.0, None)] * N
            result = minimize(
                obj, h_start, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.success or result.fun < -best_val:
                h_opt = np.maximum(result.x, 0.0)
                val_opt = compute_value(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
        
        # Try SLSQP from the best found so far
        try:
            result2 = minimize(
                obj, best_h.copy(), method='SLSQP',
                bounds=bounds, options={'maxiter': 300, 'ftol': 1e-12}
            )
            if result2.success or result2.fun < -best_val:
                h_opt2 = np.maximum(result2.x, 0.0)
                val_opt2 = compute_value(h_opt2)
                if val_opt2 > best_val:
                    best_val = val_opt2
                    best_h = h_opt2.copy()
        except Exception:
            pass
    
    # Try a few more random restarts if time permits
    np.random.seed(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Random smooth initial guess
        x = np.arange(N)
        # Random Gaussian-like bump
        center = np.random.uniform(0, N-1)
        width = np.random.uniform(5, 25)
        h_rand = np.exp(-((x - center) ** 2) / (2 * width ** 2))
        h_rand = np.maximum(h_rand, 0.0)
        
        val_rand = compute_value(h_rand)
        if val_rand > best_val:
            best_val = val_rand
            best_h = h_rand.copy()
        
        # Quick local optimization from random start
        try:
            h_start_rand = np.maximum(h_rand, 1e-6)
            result_r = minimize(
                obj, h_start_rand, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 200, 'ftol': 1e-11}
            )
            h_opt_r = np.maximum(result_r.x, 0.0)
            val_opt_r = compute_value(h_opt_r)
            if val_opt_r > best_val:
                best_val = val_opt_r
                best_h = h_opt_r.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if best_h is not None and time.time() < deadline - 10:
        try:
            h_start_final = np.maximum(best_h, 1e-8)
            result_final = minimize(
                obj, h_start_final, method='L-BFGS-B',
                bounds=[(0.0, None)] * N,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            h_final = np.maximum(result_final.x, 0.0)
            val_final = compute_value(h_final)
            if val_final > best_val:
                best_val = val_final
                best_h = h_final.copy()
        except Exception:
            pass
    
    if best_h is None:
        return [1.0] * N
    
    # Ensure nonnegative and finite
    best_h = np.maximum(best_h, 0.0)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)
    
    return list(best_h)
