import numpy as np
from scipy.optimize import minimize
import random

N = 50

# EVOLVE-BLOCK-START
def construct():
    best_h = None
    best_score = 0.0
    
    def score(h):
        h = np.array(h)
        s = np.sum(h)
        if s == 0:
            return 0.0
        h_norm = h / s
        q = np.sum(h_norm**2)
        conv = np.convolve(h_norm, h_norm, mode='full')
        m = np.max(conv)
        if m == 0:
            return 0.0
        return q / m

    def objective(weights):
        # weights are for the selected indices
        w = np.array(weights)
        if np.any(w < 0):
            return 1e10
        s = np.sum(w)
        if s == 0:
            return 1e10
        w = w / s
        
        # Construct full h
        h = np.zeros(N)
        h[idxs] = w
        
        q = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        m = np.max(conv)
        
        if m == 0:
            return 1e10
            
        return m / q # Minimize this

    # Strategy: Try various K and various index configurations
    # 1. Contiguous blocks (like original but with Nelder-Mead)
    # 2. Spread out blocks
    # 3. Random subsets
    
    configs = []
    
    # Contiguous
    for K in range(2, 15):
        configs.append(list(range(K)))
        
    # Spread out (uniformly distributed)
    for K in range(2, 15):
        idxs = np.linspace(0, N-1, K, dtype=int)
        # Ensure unique and sorted
        idxs = sorted(set(idxs))
        if len(idxs) >= 2:
            configs.append(idxs)
            
    # Random subsets
    random.seed(42)
    for _ in range(20):
        K = random.randint(3, 12)
        idxs = random.sample(range(N), K)
        configs.append(idxs)
        
    for idxs in configs:
        K = len(idxs)
        if K < 2:
            continue
            
        # Initial guess: uniform weights
        w0 = np.ones(K) / K
        
        # Use Nelder-Mead
        try:
            res = minimize(objective, w0, args=(idxs,), method='Nelder-Mead', 
                           options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})
            
            if res.success:
                w_opt = res.x
                # Clamp to non-negative
                w_opt = np.maximum(w_opt, 0)
                s = np.sum(w_opt)
                if s > 0:
                    w_opt = w_opt / s
                    h = np.zeros(N)
                    h[idxs] = w_opt
                    
                    sc = score(h)
                    if sc > best_score:
                        best_score = sc
                        best_h = h
                        
        except Exception:
            pass
            
    # Fallback
    if best_h is None or best_score < 1.2:
        h = [0.0] * N
        h[0] = 2.0 / 3.0
        h[1] = 1.0 / 3.0
        return h
        
    return best_h
# EVOLVE-BLOCK-END
