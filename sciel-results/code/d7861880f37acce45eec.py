"""New program using direct optimization of h with soft-max objective."""
import numpy as np
from scipy.optimize import minimize

N = 100
TARGET_SUM = N / 2.0

def _soft_max(corr, tau=10.0):
    """Approximate max of corr vector using soft-max."""
    # Shift to avoid overflow
    m = np.max(corr)
    if np.isinf(m):
        return m
    return m + np.log(np.sum(np.exp(tau * (corr - m)))) / tau

def _objective(h):
    """
    Objective to minimize: approximate max correlation of h and 1-h.
    h is flattened array of size N.
    """
    h = h.reshape(N)
    # Compute full correlation
    # np.correlate(h, 1-h, mode='full')
    # Note: 1-h is computed on the fly
    inv_h = 1.0 - h
    corr = np.correlate(h, inv_h, mode="full")
    
    # We want to minimize the max value
    # Using soft-max for differentiability
    val = _soft_max(corr, tau=5.0)
    
    return val

def _constraints(h):
    """
    Return list of constraint dicts for SLSQP.
    """
    h = h.reshape(N)
    # Equality constraint: sum(h) = N/2
    # Inequality constraints: h >= 0, h <= 1 are handled by bounds
    
    return [
        {
            'type': 'eq',
            'fun': lambda x: np.sum(x.reshape(N)) - TARGET_SUM,
            'jac': lambda x: np.ones(N)
        }
    ]

def construct():
    best_h = np.full(N, 0.5)
    best_val = np.max(np.correlate(best_h, 1.0 - best_h, mode="full")) / N * 2.0
    
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # Multi-start
    np.random.seed(100)
    
    initial_guesses = []
    
    # 1. Random
    for _ in range(5):
        h0 = np.random.uniform(0, 1, N)
        # Normalize sum
        h0 = h0 * (TARGET_SUM / np.sum(h0))
        h0 = np.clip(h0, 0, 1)
        # Re-adjust sum if clipping changed it significantly
        # Simple correction
        diff = TARGET_SUM - np.sum(h0)
        h0 += diff / N
        h0 = np.clip(h0, 0, 1)
        initial_guesses.append(h0)
        
    # 2. Low frequency sine
    i = np.arange(N)
    for freq in [0.1, 0.2, 0.5]:
        h0 = 0.5 + 0.4 * np.sin(2 * np.pi * freq * i / N)
        h0 = np.clip(h0, 0, 1)
        # Adjust sum
        diff = TARGET_SUM - np.sum(h0)
        h0 += diff / N
        h0 = np.clip(h0, 0, 1)
        initial_guesses.append(h0)
        
    # 3. Step function
    h0 = np.zeros(N)
    h0[:N//2] = 1.0
    h0 = np.clip(h0, 0, 1)
    # Sum is N/2, so it's valid
    initial_guesses.append(h0)
    
    # 4. Triangular wave
    for phase in [0, 0.5]:
        t = (i / N + phase) % 1.0
        h0 = 0.5 + 0.4 * np.where(t < 0.5, 2*t, 2*(1-t))
        h0 = np.clip(h0, 0, 1)
        diff = TARGET_SUM - np.sum(h0)
        h0 += diff / N
        h0 = np.clip(h0, 0, 1)
        initial_guesses.append(h0)

    for h0 in initial_guesses:
        if not np.all(np.isfinite(h0)):
            continue
            
        try:
            res = minimize(
                _objective,
                h0,
                method='SLSQP',
                bounds=bounds,
                constraints=_constraints(h0),
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            h_opt = res.x.reshape(N)
            
            # Verify validity
            if not np.all(np.isfinite(h_opt)):
                continue
            if abs(np.sum(h_opt) - TARGET_SUM) > 1e-9:
                continue
            if np.any(h_opt < -1e-9) or np.any(h_opt > 1 + 1e-9):
                continue
            
            # Clip tiny numerical errors
            h_opt = np.clip(h_opt, 0, 1)
            
            # Recalculate true value
            corr = np.correlate(h_opt, 1.0 - h_opt, mode="full")
            val_opt = np.max(corr) / N * 2.0
            
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt
                
        except Exception as e:
            pass
            
    return best_h
