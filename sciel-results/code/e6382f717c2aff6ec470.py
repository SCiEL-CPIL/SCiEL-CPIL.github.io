import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = 150.0 - 5.0  # 5 second safety margin
    
    n = adjacency.shape[0]
    adj = np.asarray(adjacency, dtype=float)
    
    # Compute principal eigenvector via power iteration
    np.random.seed(42)
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    for i in range(100):
        v_new = adj @ v
        norm = np.linalg.norm(v_new)
        if norm < 1e-12:
            break
        v_new = v_new / norm
        if np.linalg.norm(v_new - v) < 1e-10:
            break
        v = v_new
        if time.time() > deadline:
            break
    
    # Center the eigenvector
    eig = v - v.mean()
    eig_std = eig.std()
    if eig_std < 1e-12:
        eig_norm = np.zeros(n)
    else:
        eig_norm = eig / eig_std
    
    # Degree
    deg = adj.sum(axis=1)
    deg_std = deg.std()
    if deg_std < 1e-12:
        deg_norm = np.zeros(n)
    else:
        deg_norm = (deg - deg.mean()) / deg_std
    
    # Composite score: favor spectral color class, penalize high degree
    gamma = 0.3
    score = eig_norm - gamma * deg_norm
    
    return score
