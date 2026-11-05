import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s kill
    
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    
    # Calculate degrees
    degrees = A.sum(axis=1)
    
    # Calculate 2 * L(v) where L(v) = number of edges in N(v)
    # B = A^2, B[v,u] = number of common neighbors of v and u
    # two_L[v] = sum_{u in N(v)} B[v,u] = sum_{u in N(v)} |N(v) ∩ N(u)| = 2 * L(v)
    B = A @ A
    two_L = np.sum(A * B, axis=1)
    L = two_L / 2.0
    
    # Spectral centrality via power iteration
    # Compute principal eigenvector of A
    # Start with random positive vector
    np.random.seed(42)
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)
    
    # Power iteration
    for _ in range(50):
        if time.time() - start_time > deadline - 10:
            break
        x_new = A @ x
        norm = np.linalg.norm(x_new)
        if norm == 0:
            break
        x_new = x_new / norm
        # Check convergence
        if np.linalg.norm(x_new - x) < 1e-10:
            x = x_new
            break
        x = x_new
    
    # Eigenvector centrality
    eig_cent = x
    
    # Combine local density score with spectral centrality
    # Lower eigenvector centrality = more peripheral, less entangled
    # We want to prioritize vertices that are both locally favorable (high L - d) 
    # and spectrally peripheral (low eig_cent)
    
    # Normalize components for combination
    L_d = L - degrees
    L_d_min = L_d.min()
    L_d_max = L_d.max()
    if L_d_max - L_d_min > 1e-10:
        L_d_norm = (L_d - L_d_min) / (L_d_max - L_d_min)
    else:
        L_d_norm = np.zeros(n)
    
    eig_min = eig_cent.min()
    eig_max = eig_cent.max()
    if eig_max - eig_min > 1e-10:
        eig_norm = (eig_cent - eig_min) / (eig_max - eig_min)
    else:
        eig_norm = np.zeros(n)
    
    # Weighted combination: 60% local density, 40% spectral peripherality
    # We want high L_d (good local) and low eig_cent (peripheral)
    scores = 0.6 * L_d_norm + 0.4 * (1 - eig_norm)
    
    # Safety check
    if time.time() - start_time > deadline:
        return scores
    
    return scores
