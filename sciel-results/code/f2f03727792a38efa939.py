import numpy as np
import time
from scipy.optimize import differential_evolution

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    def project_to_simplex(h):
        """Project h onto the set {x in [0,1]^N : sum(x) = N/2}"""
        for _ in range(200):
            h = np.clip(h, 0, 1)
            diff = N/2 - np.sum(h)
            if abs(diff) < 1e-14:
                break
            mask = (h > 1e-12) & (h < 1 - 1e-12)
            if not np.any(mask):
                break
            h[mask] += diff / np.sum(mask)
        # Final exact fix
        diff = N/2 - np.sum(h)
        if abs(diff) > 1e-15:
            mask = (h > 1e-12) & (h < 1 - 1e-12)
            if np.any(mask):
                h[mask] += diff / np.sum(mask)
            else:
                h += diff / N
        h = np.clip(h, 0, 1)
        return h
    
    def get_overlap(h):
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        c[N-1] = -np.inf
        return np.max(c)
    
    def objective(x):
        h = project_to_simplex(x)
        return get_overlap(h)
    
    bounds = [(0, 1)] * N
    
    # Run differential evolution
    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=42,
        maxiter=50,
        tol=1e-4,
        polish=False,
        init='sobol'
    )
    
    best_h = project_to_simplex(result.x)
    best_score = get_overlap(best_h)
    
    # Local refinement with simulated annealing
    T = 0.05
    h = best_h.copy()
    current_score = best_score
    
    while T > 0.0001 and time.time() - start_time < time_limit:
        for _ in range(30):
            if time.time() - start_time > time_limit:
                break
            i, j = np.random.choice(N, 2, replace=False)
            max_delta = min(1 - h[i], h[j])
            if max_delta <= 0:
                continue
            scale = min(1.0, T * 10)
            delta = np.random.uniform(-max_delta * scale, max_delta * scale)
            new_h = h.copy()
            new_h[i] += delta
            new_h[j] -= delta
            if new_h[i] < 0 or new_h[i] > 1 or new_h[j] < 0 or new_h[j] > 1:
                continue
            new_score = get_overlap(new_h)
            delta_E = new_score - current_score
            if delta_E < 0 or (T > 0 and np.random.random() < np.exp(-delta_E / T)):
                h = new_h
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
        T *= 0.999
    
    # Final projection
    best_h = project_to_simplex(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
