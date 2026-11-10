import numpy as np
import time

def heuristics(demands, capacity):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    demands = np.array(demands, dtype=float)
    n = len(demands)
    
    # 1. Construct Graph Weights
    # W[i,j] is high if d[i] + d[j] is close to capacity (tight fit)
    # W[i,j] = 1 / (capacity - d[i] - d[j])
    
    residual = capacity - demands[:, None] - demands[None, :]
    valid = residual > 0
    
    # Initialize weights to 0
    weights = np.zeros((n, n), dtype=float)
    
    # Calculate weights for valid pairs
    # Use epsilon to avoid division by zero for perfect fits (residual=0)
    # A perfect fit should have very high weight
    eps = 1e-6
    valid_residuals = residual[valid]
    weights[valid] = 1.0 / (valid_residuals + eps)
    
    # Ensure diagonal is 0
    np.fill_diagonal(weights, 0)
    
    # 2. Construct Laplacian
    degree = np.sum(weights, axis=1)
    L = np.diag(degree) - weights
    
    # 3. Eigen Decomposition
    # Check time
    if time.time() > deadline:
        # Fallback to simple heuristic if eigendecomposition is slow (unlikely)
        # But N=120 is fast.
        pass
        
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except Exception:
        # Fallback to simple tightness heuristic if eig fails
        # H[i,j] = 1/(C - d[i] - d[j]) if valid else 1e-9
        H = np.zeros((n, n))
        H[valid] = 1.0 / (residual[valid] + eps)
        H[~valid] = 1e-9
        return H

    # Fiedler vector is the eigenvector for the second smallest eigenvalue
    # eigenvalues are sorted ascending
    fiedler_vector = eigenvectors[:, 1]
    
    # 4. Construct Heuristic Matrix
    # Normalize Fiedler vector to have max absolute value 1
    max_abs = np.max(np.abs(fiedler_vector))
    if max_abs > 0:
        fiedler_vector = fiedler_vector / max_abs
    else:
        # If vector is zero, fallback to uniform
        fiedler_vector = np.ones(n)
        
    H = np.outer(fiedler_vector, fiedler_vector)
    
    # Entries not positive become 1e-9
    H = np.where(H > 0, H, 1e-9)
    
    return H
