import numpy as np
from scipy.optimize import differential_evolution

N = 100

def construct():
    def objective(h):
        g = 1.0 - h
        c = np.correlate(h, g, mode="full")
        return np.max(c) / N * 2.0

    # Bounds for each h_i
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # Constraint: sum(h) = 50
    # DE doesn't handle equality constraints directly.
    # We can penalize the objective.
    def objective_penalized(h):
        sum_h = np.sum(h)
        penalty = (sum_h - 50.0) ** 2
        return objective(h) + 1000.0 * penalty

    # Run DE
    res = differential_evolution(objective_penalized, bounds, seed=42, maxiter=100, tol=1e-6, polish=False)
    
    h_opt = res.x
    
    # Fix sum exactly
    diff = 50.0 - np.sum(h_opt)
    if abs(diff) > 1e-9:
        # Distribute diff evenly
        h_opt += diff / N
        
    # Clip
    h_opt = np.clip(h_opt, 0.0, 1.0)
    
    # Final sum fix (clipping might break sum)
    diff = 50.0 - np.sum(h_opt)
    if abs(diff) > 1e-9:
        # Add to h[0] if possible
        if 0 <= h_opt[0] + diff <= 1:
            h_opt[0] += diff
        else:
            # Distribute to indices with room
            idxs = np.argsort(np.abs(h_opt - 0.5))
            for idx in idxs:
                if abs(diff) < 1e-12: break
                step = diff / (len(idxs) - idx)
                if 0 <= h_opt[idx] + step <= 1:
                    h_opt[idx] += step
                    diff -= step
                else:
                    pass
                    
    return h_opt
