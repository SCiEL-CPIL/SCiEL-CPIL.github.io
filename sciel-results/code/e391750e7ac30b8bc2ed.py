import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing task.
    
    Combines the squared-inverse-waste direct heuristic with a spectral
    (Fiedler vector) component that captures global compatibility structure.
    
    Args:
        demands: list or array of item sizes (floats).
        capacity: bin capacity (float).
        
    Returns:
        numpy array of shape (n, n) with heuristic values.
    """
    start_time = time.time()
    DEADLINE = 320.0  # Safety margin before 330s kill
    
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    
    if n == 0:
        return np.empty((0, 0))
    
    # Vectorized computation of pair sums
    sums = d[:, None] + d[None, :]
    fits = sums <= capacity
    waste = capacity - sums  # >= 0 for fitting pairs
    
    # --- Direct heuristic component (squared inverse waste) ---
    # H_direct[i,j] = 1 / (waste + eps)^2 for fitting pairs
    # This strongly rewards tight fits
    eps = 1e-6
    safe_waste = np.where(fits, waste + eps, 1.0)
    H_direct = np.where(fits, 1.0 / (safe_waste ** 2), 0.0)
    
    # --- Spectral clustering component ---
    # Build a graph where edge weights reflect pair compatibility
    # Weight: 1/(1+waste) for fitting pairs, 0 for non-fitting
    W = np.where(fits, 1.0 / (1.0 + waste), 0.0)
    
    # Compute degree matrix
    degrees = W.sum(axis=1)
    safe_degrees = np.where(degrees > 0, degrees, 1.0)
    
    # Construct symmetric normalized Laplacian: L_sym = I - D^(-1/2) W D^(-1/2)
    d_inv_sqrt = 1.0 / np.sqrt(safe_degrees)
    L_sym = np.eye(n) - (W * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
    
    # Ensure symmetry (numerical safety)
    L_sym = (L_sym + L_sym.T) / 2.0
    
    # Compute Fiedler vector (eigenvector of second smallest eigenvalue)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
        
        if n >= 2:
            fiedler = eigenvectors[:, 1]
        else:
            fiedler = np.ones(n)
        
        # Normalize
        norm = np.linalg.norm(fiedler)
        if norm > 0:
            fiedler = fiedler / norm
        else:
            fiedler = np.ones(n) / np.sqrt(n)
        
        # Spectral heuristic: outer product of Fiedler vector
        H_spectral = np.outer(fiedler, fiedler)
        
        # Only keep for fitting pairs
        H_spectral = np.where(fits, H_spectral, 0.0)
        
        # Scale to be comparable to direct component
        max_spectral = H_spectral[fits].max() if np.any(fits) else 1.0
        if max_spectral > 0:
            H_spectral = H_spectral / max_spectral
        else:
            H_spectral = np.zeros_like(H_spectral)
            
    except Exception:
        H_spectral = np.zeros((n, n))
    
    # --- Combine components ---
    # Normalize H_direct to have max value of 1 for fitting pairs
    max_direct = H_direct[fits].max() if np.any(fits) else 1.0
    if max_direct > 0:
        H_direct_norm = H_direct / max_direct
    else:
        H_direct_norm = np.zeros_like(H_direct)
    
    # Weight direct component more heavily (local guidance)
    alpha = 0.75  # Direct heuristic weight
    beta = 0.25   # Spectral heuristic weight
    
    H_combined = alpha * H_direct_norm + beta * H_spectral
    
    # For non-fitting pairs, set to 1e-9
    H = np.where(fits, H_combined, 1e-9)
    
    # Ensure all values are positive and finite
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    
    # Time guard
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        pass  # Return immediately; computation is fast anyway
    
    return H
