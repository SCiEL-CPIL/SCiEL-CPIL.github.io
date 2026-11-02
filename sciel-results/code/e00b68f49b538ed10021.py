import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50

def compute_score(h):
    """
    Compute the ratio of squared L2 norm to (L1 norm * sup norm of autoconvolution).
    Scale-invariant: score(alpha*h) = score(h) for alpha > 0.
    """
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    
    l2_sq = np.sum(h**2)
    
    c = np.correlate(h, h, mode='full')
    sup_c = np.max(c)
    
    if sup_c < 1e-15:
        return 0.0
    
    return l2_sq / (l1 * sup_c)

def negative_score(h):
    return -compute_score(h)

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin before 150s kill
    
    best_h = np.zeros(N)
    best_h[0] = 1.0  # Baseline: single spike gives score 1.0
    best_score = 1.0
    
    # Strategy 1: Differential evolution over compact support
    # Try different support sizes to balance L2 concentration vs autoconvolution spread
    
    for support_size in [2, 3, 4, 5, 6, 8, 10, 12, 15]:
        if time.time() > deadline:
            break
        
        # Use first 'support_size' elements, rest are zero
        # This keeps autoconvolution compact and manageable
        
        def de_objective(x):
            h = np.zeros(N)
            h[:support_size] = x
            return negative_score(h)
        
        bounds = [(0.0, 1.0)] * support_size
        
        try:
            de_res = differential_evolution(
                de_objective,
                bounds,
                maxiter=150,
                tol=1e-8,
                seed=42,
                polish=True,
                init='latinhypercube',
                popsize=15
            )
            
            if de_res.fun < 0:
                h_full = np.zeros(N)
                h_full[:support_size] = de_res.x
                score = -de_res.fun
                if score > best_score:
                    best_score = score
                    best_h = h_full.copy()
                    
        except Exception:
            pass
    
    if time.time() > deadline:
        return list(np.maximum(best_h, 0.0))
    
    # Strategy 2: Local refinement from multiple starting points
    # Try various initial patterns and refine with Nelder-Mead
    
    start_patterns = [
        # Two spikes
        lambda: np.array([0.7, 0.3] + [0.0]*(N-2)),
        # Three spikes
        lambda: np.array([0.5, 0.3, 0.2] + [0.0]*(N-3)),
        # Four spikes
        lambda: np.array([0.4, 0.3, 0.2, 0.1] + [0.0]*(N-4)),
        # Uniform on first 5
        lambda: np.array([0.2]*5 + [0.0]*(N-5)),
        # Uniform on first 8
        lambda: np.array([0.125]*8 + [0.0]*(N-8)),
        # Asymmetric: heavy first, light rest
        lambda: np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05] + [0.0]*(N-6)),
    ]
    
    for pattern_func in start_patterns:
        if time.time() > deadline:
            break
        
        x0 = pattern_func()
        
        # Find support
        support = np.where(x0 > 1e-6)[0]
        if len(support) == 0:
            continue
        
        def objective(weights):
            h = np.zeros(N)
            h[support] = np.maximum(weights, 0.0)
            return negative_score(h)
        
        try:
            res = minimize(objective, x0[support], method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < 0:
                h_refined = np.zeros(N)
                h_refined[support] = np.maximum(res.x, 0.0)
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_h = h_refined.copy()
        except Exception:
            pass
    
    if time.time() > deadline:
        return list(np.maximum(best_h, 0.0))
    
    # Strategy 3: Try random sparse supports with random weights, then refine
    rng = np.random.RandomState(123)
    
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Choose a random support size
        K = rng.randint(2, 12)
        indices = rng.choice(N, K, replace=False)
        indices.sort()
        
        # Random positive weights
        weights = rng.dirichlet(np.ones(K))
        
        x0 = np.zeros(N)
        x0[indices] = weights
        
        # Quick local refinement
        def objective2(w):
            h = np.zeros(N)
            h[indices] = np.maximum(w, 0.0)
            return negative_score(h)
        
        try:
            res = minimize(objective2, weights, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7})
            if res.fun < 0:
                h_trial = np.zeros(N)
                h_trial[indices] = np.maximum(res.x, 0.0)
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_h = h_trial.copy()
        except Exception:
            pass
    
    # Final: ensure nonnegative and return
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(f"Score: {compute_score(h)}")
    print(f"Sum: {np.sum(h)}")
