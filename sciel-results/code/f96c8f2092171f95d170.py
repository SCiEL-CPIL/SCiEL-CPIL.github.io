import numpy as np
from scipy.optimize import minimize
import itertools

N = 14
D = 3

def construct():
    n_pairs = N * (N - 1) // 2
    
    def edm_from_params(params):
        dist_sq = np.zeros((N, N))
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                dist_sq[i, j] = params[idx]
                dist_sq[j, i] = params[idx]
                idx += 1
        return dist_sq
    
    def gram_from_edm(dist_sq):
        J = np.eye(N) - np.ones((N, N)) / N
        G = -0.5 * (J @ dist_sq @ J)
        return G
    
    def penalty(params):
        dist_sq = edm_from_params(params)
        
        min_dist = np.min(dist_sq[~np.eye(N, dtype=bool)])
        if min_dist < 1e-8:
            return 1e10 + (1e-8 - min_dist) ** 2
        
        G = gram_from_edm(dist_sq)
        
        eigvals = np.linalg.eigvalsh(G)
        neg_eig = np.sum(np.maximum(0, -eigvals) ** 2)
        
        eigvals_sorted = np.sort(eigvals)[::-1]
        rank_penalty = np.sum(eigvals_sorted[3:] ** 2) if len(eigvals_sorted) > 3 else 0
        
        max_dist = np.max(dist_sq[~np.eye(N, dtype=bool)])
        min_dist = np.min(dist_sq[~np.eye(N, dtype=bool)])
        ratio = max_dist / min_dist
        
        return ratio + 1000 * neg_eig + 100 * rank_penalty
    
    best_params = None
    best_score = float('inf')
    
    for trial in range(20):
        if trial == 0:
            base_dist = 1.0
            params0 = np.full(n_pairs, base_dist)
            params0 += np.random.randn(n_pairs) * 0.1
        else:
            params0 = np.abs(np.random.randn(n_pairs)) * 0.5 + 0.1
        
        params0 = np.maximum(params0, 1e-6)
        
        try:
            res = minimize(penalty, params0, method='Powell', 
                         options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            continue
    
    if best_params is None:
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)
    
    dist_sq = edm_from_params(best_params)
    G = gram_from_edm(dist_sq)
    
    eigvals, eigvecs = np.linalg.eigh(G)
    
    eigvals_clipped = np.clip(eigvals, 0, None)
    idx_sorted = np.argsort(eigvals_clipped)[::-1]
    eigvals_top = eigvals_clipped[idx_sorted[:D]]
    eigvecs_top = eigvecs[:, idx_sorted[:D]]
    
    X = eigvecs_top @ np.diag(np.sqrt(eigvals_top))
    X = X - np.mean(X, axis=0)
    
    return X
