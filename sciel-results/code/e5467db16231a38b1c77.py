"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Assigns a priority score to each vertex for the greedy independent set decoder.
    Higher scores are picked first.
    
    Strategy: Combine spectral centrality (eigenvector centrality via power iteration)
    with local degree information. Vertices with low eigenvector centrality are 
    structurally peripheral and less entangled in the core, making them safer to pick
    early without blocking many other potential picks. We blend this with degree 
    to handle local structures.
    """
    start_time = time.time()
    
    # Ensure float matrix
    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]
    
    if N == 0:
        return np.array([])
        
    # 1. Degree
    deg = A.sum(axis=1)
    
    # 2. Eigenvector Centrality via Power Iteration
    # This is more robust and faster than full eig for sparse-ish or large graphs,
    # though N=200 is small. Power iteration converges quickly for the principal eigenvalue.
    # We want the principal eigenvector.
    # Initialize with random or ones. Ones is fine for connected graphs.
    # If disconnected, it might bias, but ER graphs with N=200 are likely connected or have few components.
    # To be safe, we can use random initialization with a fixed seed for reproducibility if needed,
    # but standard power iteration with ones usually works for the dominant eigenvector of a non-negative matrix.
    
    # However, for Independent Set, we want to AVOID high centrality.
    # So we compute centrality, then score = -centrality.
    
    # Power Iteration
    v = np.ones(N)
    v_norm = np.linalg.norm(v)
    if v_norm > 0:
        v = v / v_norm
        
    # Iterate
    # A is symmetric, so eigenvalues are real.
    # Principal eigenvalue is positive for connected graphs with at least one edge.
    max_iter = 100
    tol = 1e-8
    
    for _ in range(max_iter):
        if time.time() - start_time > 10.0:
            # Fallback to degree if taking too long
            return -deg.astype(float)
            
        v_new = A @ v
        v_norm_new = np.linalg.norm(v_new)
        
        if v_norm_new < 1e-12:
            # Matrix might be zero or similar, break
            break
            
        v_new = v_new / v_norm_new
        
        # Check convergence
        if np.linalg.norm(v_new - v) < tol:
            v = v_new
            break
            
        v = v_new
        
    # Ensure v is non-negative (eigenvector can have arbitrary sign)
    # The principal eigenvector of a non-negative matrix can be chosen non-negative (Perron-Frobenius)
    # If it's mostly negative, flip it.
    if np.sum(v) < 0:
        v = -v
        
    # 3. Combine Degree and Eigenvector Centrality
    # Normalize both to zero mean, unit variance to combine fairly
    def zscore(x):
        m = x.mean()
        s = x.std()
        if s < 1e-9:
            return np.zeros_like(x)
        return (x - m) / s
        
    z_deg = zscore(deg)
    z_eig = zscore(v)
    
    # We want to pick vertices with LOW degree and LOW eigenvector centrality.
    # So score should be negative of a weighted sum.
    # Weighting: Eigenvector centrality captures global structure, degree captures local.
    # Inspiration 2 used 0.5 for eig. Let's try a balance.
    # If we rely too much on degree, we get LDF. If too much on eig, we might ignore local traps.
    # Let's try 0.5 * degree + 0.5 * eig.
    
    score = -(0.5 * z_deg + 0.5 * z_eig)
    
    # Safety check
    if time.time() - start_time > 15.0:
         # Fallback to simple degree
         return -deg.astype(float)
         
    return score
# EVOLVE-BLOCK-END
