import numpy as np
from scipy.optimize import minimize

N = 50

def construct():
    # Use a Sidon set to minimize the maximum autoconvolution peak
    # relative to the L2 norm, assuming L1 normalization.
    # Sidon set in [0, 49]: {0, 1, 3, 7, 15, 31, 48}
    # Size K = 7.
    sidon_indices = [0, 1, 3, 7, 15, 31, 48]
    K = len(sidon_indices)
    
    best_h = [0.0] * N
    best_score = -1
    
    # Function to evaluate a weight vector
    def eval_weights(w):
        h = np.zeros(N)
        for i, idx in enumerate(sidon_indices):
            h[idx] = w[i]
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-15:
            return 0, h
        ratio = l2_sq / max_conv
        return ratio, h

    # Initial uniform
    w0 = np.ones(K) / K
    ratio0, h0 = eval_weights(w0)
    best_score = ratio0
    best_h = h0

    # Try a few random restarts
    np.random.seed(42)
    for _ in range(10):
        w_rand = np.random.dirichlet(np.ones(K))
        
        def objective(w):
            r, _ = eval_weights(w)
            return -r
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 1) for _ in range(K)]
        
        try:
            res = minimize(objective, w_rand, method='SLSQP', bounds=bounds, constraints=constraints, options={'ftol': 1e-10, 'maxiter': 500})
            if res.success:
                r, h_opt = eval_weights(res.x)
                if r > best_score:
                    best_score = r
                    best_h = h_opt
        except Exception:
            pass
            
    return list(best_h)
