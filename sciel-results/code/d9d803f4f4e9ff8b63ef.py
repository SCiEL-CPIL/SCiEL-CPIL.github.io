import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    def value_from_h(h):
        h = np.asarray(h)
        h = np.maximum(h, 0)
        if np.sum(h) == 0:
            return 0.0
        h = h / np.sum(h)
        
        g = np.convolve(h, h)
        m = len(g)
        
        y = np.zeros(m + 2)
        y[1:m+1] = g
        
        s1 = 2.0 * np.sum(y**2)
        s2 = np.sum(y[:-1] * y[1:])
        sum_val = s1 + s2
        
        w = 1.0 / (m + 1)
        l2 = (w / 3.0) * sum_val
        
        norm_1 = np.sum(np.abs(g)) / (m + 1)
        
        if np.max(g) < 1e-10:
            return 0.0
        norm_inf = np.max(g)
        
        denom = norm_1 * norm_inf
        if denom < 1e-10:
            return 0.0
            
        return l2 / denom

    def objective(h):
        return -value_from_h(h)

    cons = ({'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0})
    bounds = [(0.0, None) for _ in range(N)]
    
    # 1. Direct evaluation of sparse candidates
    sparse_candidates = []
    
    # k=2: All pairs
    for i in range(N):
        for j in range(i+1, N):
            h = np.zeros(N)
            h[i] = 0.5
            h[j] = 0.5
            sparse_candidates.append(h)
            
    # k=3: Random samples
    np.random.seed(123)
    for _ in range(200):
        idx = np.random.choice(N, 3, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0 / 3.0
        sparse_candidates.append(h)
        
    # k=5: Random samples
    for _ in range(100):
        idx = np.random.choice(N, 5, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0 / 5.0
        sparse_candidates.append(h)

    # Evaluate all sparse candidates
    best_sparse_val = -np.inf
    best_sparse_h = None
    for h in sparse_candidates:
        v = value_from_h(h)
        if v > best_sparse_val:
            best_sparse_val = v
            best_sparse_h = h

    # 2. Dense/Other initial guesses
    guesses = []
    
    # Uniform
    guesses.append(np.ones(N) / N)
    
    # Gaussian
    x = np.linspace(-1, 1, N)
    g1 = np.exp(-x**2 * 10)
    guesses.append(g1 / np.sum(g1))
    g2 = np.exp(-x**2 * 2)
    guesses.append(g2 / np.sum(g2))
    
    # Random
    np.random.seed(42)
    r = np.random.rand(N)
    guesses.append(r / np.sum(r))
    
    # Add best sparse candidate as a guess for SLSQP
    if best_sparse_h is not None:
        guesses.append(best_sparse_h)
        
    # Add a few random perturbations of the best sparse candidate
    for _ in range(5):
        h_pert = best_sparse_h + np.random.randn(N) * 0.1
        h_pert = np.maximum(h_pert, 0)
        h_pert /= np.sum(h_pert)
        guesses.append(h_pert)

    best_val = -np.inf
    best_h = guesses[0]
    
    # Run SLSQP on all guesses
    for h0 in guesses:
        try:
            res = minimize(objective, h0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12})
            if res.success:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_h = res.x
        except Exception:
            continue
            
    # Compare with best sparse value (in case SLSQP failed or got stuck)
    if best_sparse_val > best_val:
        best_val = best_sparse_val
        best_h = best_sparse_h

    # Final cleanup
    best_h = np.maximum(best_h, 0)
    best_h /= np.sum(best_h)
    
    return list(best_h)
# EVOLVE-BLOCK-END
