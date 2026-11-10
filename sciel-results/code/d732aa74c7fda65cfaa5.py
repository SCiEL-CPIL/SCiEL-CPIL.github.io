"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ant colony optimization.
    
    Strategy: Spectral Embedding of the Complementarity Graph.
    Instead of using a direct pairwise formula, we model the items as nodes in a graph 
    where edge weights represent the "tightness" of the fit (how close the sum is to capacity).
    We then embed these items into a low-dimensional latent space using the eigenvectors 
    of the normalized graph Laplacian. The heuristic is derived from the cosine similarity 
    of these embeddings. This captures global structural relationships between items 
    (e.g., clusters of items that fit well together) rather than just local pairwise arithmetic.
    """
    start_time = time.time()
    
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    
    if n == 0:
        return np.empty((0, 0))
        
    # 1. Construct the Complementarity Weight Matrix W
    # W[i, j] is high if d[i] + d[j] is close to capacity and fits.
    sums = d[:, None] + d[None, :]
    fits = sums <= capacity
    
    # Calculate waste
    waste = capacity - sums
    
    # We want a sharp peak for tight fits.
    # Use an exponential decay based on waste.
    # alpha controls the sharpness. Higher alpha = sharper peak.
    # With capacity 150 and items 20-99, waste ranges from 0 to ~110.
    # We want W to be significant only when waste is small.
    # Let's use alpha = 0.5 or 1.0. 
    # If waste=0, exp(0)=1. If waste=10, exp(-5)~0.006 (for alpha=0.5).
    # This creates a sparse graph focusing on tight pairs.
    
    alpha = 0.5
    # Only compute for fitting pairs to avoid negative waste issues in exp
    # For non-fitting, waste > capacity, but we mask them anyway.
    # To be safe, clip waste to 0 for the calculation, then mask.
    safe_waste = np.where(fits, waste, 0.0)
    
    W = np.exp(-alpha * safe_waste)
    
    # Mask non-fitting pairs
    W = np.where(fits, W, 0.0)
    
    # Ensure symmetry (should be symmetric already)
    W = (W + W.T) / 2.0
    
    # 2. Compute Normalized Laplacian
    # D is the degree matrix
    degrees = np.sum(W, axis=1)
    
    # Avoid division by zero for isolated nodes
    # If a node has no connections, its embedding might be unstable.
    # Add a small epsilon to degrees or handle separately.
    # For bin packing, every item should have *some* potential partner, 
    # but with sharp alpha, some might be isolated.
    
    # Use symmetric normalization: L_sym = I - D^(-1/2) W D^(-1/2)
    # D^(-1/2) is 1/sqrt(degrees)
    
    # Handle zero degrees
    inv_sqrt_deg = np.zeros_like(degrees)
    nonzero_deg = degrees > 1e-10
    inv_sqrt_deg[nonzero_deg] = 1.0 / np.sqrt(degrees[nonzero_deg])
    
    # D^(-1/2) W
    D_inv_sqrt_W = inv_sqrt_deg[:, None] * W
    
    # D^(-1/2) W D^(-1/2)
    W_norm = D_inv_sqrt_W * inv_sqrt_deg[None, :]
    
    # Laplacian
    I = np.eye(n)
    L_sym = I - W_norm
    
    # 3. Eigendecomposition
    # We want the smallest eigenvalues (excluding 0)
    # np.linalg.eigh is for symmetric matrices
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
    except np.linalg.LinAlgError:
        # Fallback to simple heuristic if eigendecomp fails
        return _fallback_heuristic(d, capacity)
    
    # Sort by eigenvalue (eigh already returns sorted in ascending order)
    # The first eigenvector corresponds to eigenvalue 0 (trivial).
    # We take the next k eigenvectors.
    k = min(5, n - 1) # Take 5 dimensions, or less if n is small
    
    # Embeddings are the eigenvectors
    # eigenvectors is (n, n). We take columns 1 to k (indices 1..k)
    Z = eigenvectors[:, 1:1+k]
    
    # 4. Compute Heuristic from Embeddings
    # Use Cosine Similarity
    # Normalize rows of Z
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms < 1e-10] = 1.0
    Z_norm = Z / norms
    
    # Cosine similarity matrix
    H_cos = Z_norm @ Z_norm.T
    
    # The cosine similarity can be negative.
    # We want larger values to be more attractive.
    # We can shift and scale, or just use the raw value.
    # In ACO, negative values are usually problematic if not handled.
    # The task says: "Heuristic entries that are not positive become 1e-9".
    # So we will clamp negative values to 1e-9 later.
    
    # However, cosine similarity in [-1, 1] might be too small or noisy.
    # Let's scale it to be more comparable to the previous heuristic magnitudes.
    # Or just use it as is, since ACO uses it in probability calculation.
    # P ~ pheromone * heuristic.
    # If heuristic is small, pheromone dominates.
    # If heuristic is large, it dominates.
    
    # Let's boost the signal.
    # H = H_cos * 10.0 + 1.0  # Shift to positive?
    # No, the task says "not positive become 1e-9".
    # So we should keep the sign, and let the masking handle negatives.
    
    # Apply fit mask
    H = np.where(fits, H_cos, 1e-9)
    
    # Ensure positivity for fitting pairs?
    # If H_cos is negative for a fitting pair, it becomes 1e-9.
    # This effectively removes that pair from consideration.
    # This is a strong structural filter: only pairs that are "similar" in the latent space AND fit are considered.
    
    # Sanitize
    if not np.all(np.isfinite(H)):
        H = np.where(np.isfinite(H), H, 1e-9)
        
    # Ensure shape
    if H.shape != (n, n):
        raise ValueError("Wrong shape")
        
    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 325:
        pass
        
    return H

def _fallback_heuristic(d, capacity):
    """
    Fallback if spectral method fails.
    Uses the sharp product * inv_waste^2 heuristic.
    """
    n = len(d)
    sums = d[:, None] + d[None, :]
    fits = sums <= capacity
    waste = capacity - sums
    safe_waste = np.where(fits, waste + 1e-6, 1.0)
    products = d[:, None] * d[None, :]
    tightness = 1.0 / (safe_waste ** 2)
    H = products * tightness
    H = np.where(fits, H, 1e-9)
    if not np.all(np.isfinite(H)):
        H = np.where(np.isfinite(H), H, 1e-9)
    return H
# EVOLVE-BLOCK-END
