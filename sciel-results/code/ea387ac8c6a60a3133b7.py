import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 115.0
    
    A = np.asarray(adjacency, dtype=bool)
    N = A.shape[0]
    
    if N == 0:
        return np.array([])
        
    # 1. Compute Features
    D = A.sum(axis=1).astype(float)
    
    # Avoid division by zero
    D_safe = D + 1e-9
    
    SND = A @ D
    MeanND = SND / D_safe
    
    # Triangles: (A^2)_{ii} / 2 is number of triangles containing i? 
    # Actually (A @ A)[i,i] is number of length-2 paths from i to i.
    # Each triangle contributes 2 such paths (i->j->k->i and i->k->j->i).
    # So Tri_count = diag(A@A) / 2
    A_sq = A @ A
    Tri = np.diag(A_sq).astype(float) / 2.0
    
    # Clustering Coefficient
    # C(i) = 2 * Tri(i) / (D(i) * (D(i) - 1))
    Clus = np.zeros(N)
    mask = D > 1
    Clus[mask] = 2.0 * Tri[mask] / (D[mask] * (D[mask] - 1.0))
    
    # Perron Vector (Power Iteration)
    # A is symmetric, non-negative.
    v = np.ones(N) / np.sqrt(N)
    for _ in range(100):
        v_new = A @ v
        norm = np.linalg.norm(v_new)
        if norm < 1e-9:
            break
        v = v_new / norm
    PC = v
    
    # Normalize all features to [0, 1]
    def norm(x):
        min_val = x.min()
        max_val = x.max()
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)
        
    D_n = norm(D)
    MeanND_n = norm(MeanND)
    Tri_n = norm(Tri)
    Clus_n = norm(Clus)
    PC_n = norm(PC)
    
    # 2. Define Candidate Scores
    # We want to maximize IS size.
    # Base: Low Degree is good.
    # Secondary: High MeanND (neighbors are hubs, so blocking them is good) is good.
    # Secondary: Low PC (not a hub) is good.
    # Secondary: Low Clus (not in a clique) might be good?
    
    # Let's define a function to evaluate a score vector
    def evaluate(scores):
        indices = np.argsort(-scores)
        is_picked = np.zeros(N, dtype=bool)
        count = 0
        for v in indices:
            if not is_picked[v]:
                count += 1
                is_picked[v] = True
                # Vectorized neighbor marking
                is_picked[A[v]] = True
        return count

    best_size = -1
    best_scores = None
    
    # Baseline
    base_score = -D_n
    base_size = evaluate(base_score)
    if base_size > best_size:
        best_size = base_size
        best_scores = base_score.copy()
        
    # Random Search
    rng = np.random.default_rng(12345)
    
    # Number of trials depends on time
    # Estimate time per trial: ~1ms? 
    # 1000 trials = 1s.
    # Let's do 2000 trials if fast.
    
    num_trials = 2000
    
    for i in range(num_trials):
        if time.time() > DEADLINE:
            break
            
        # Weights for: -D_n, MeanND_n, -Tri_n, -Clus_n, -PC_n
        # Note: We subtract D_n, so w1 is positive weight on -D_n.
        # We add MeanND_n, so w2 is positive weight on MeanND_n.
        # We subtract Tri_n, Clus_n, PC_n, so w3, w4, w5 are positive weights on the negative terms.
        
        w1 = rng.uniform(0.5, 3.0) # Strong emphasis on low degree
        w2 = rng.uniform(0.0, 2.0) # Some emphasis on high mean neighbor degree
        w3 = rng.uniform(0.0, 1.0) # Some penalty for triangles
        w4 = rng.uniform(0.0, 1.0) # Some penalty for clustering
        w5 = rng.uniform(0.0, 2.0) # Penalty for high centrality
        
        scores = -w1 * D_n + w2 * MeanND_n - w3 * Tri_n - w4 * Clus_n - w5 * PC_n
        
        size = evaluate(scores)
        
        if size > best_size:
            best_size = size
            best_scores = scores.copy()
            
    # If random search didn't beat baseline, return baseline
    if best_scores is None:
        return base_score
        
    return best_scores
