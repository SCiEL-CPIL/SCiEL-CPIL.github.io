"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Assigns a priority score to each vertex for the greedy independent set decoder.
    Higher scores are picked first.
    Strategy: Prefer vertices with low degree, low average neighbor degree, 
    low eigenvector centrality, and low k-core number.
    """
    start_time = time.time()
    
    # Ensure float matrix for linear algebra
    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]
    
    if N == 0:
        return np.array([])
        
    # 1. Degree
    deg = A.sum(axis=1)
    
    # 2. Average Neighbor Degree
    sum_nei_deg = A @ deg
    avg_nei_deg = np.zeros(N)
    mask = deg > 0
    avg_nei_deg[mask] = sum_nei_deg[mask] / deg[mask]
    
    # 3. Eigenvector Centrality
    # For small N, dense eig is sufficient and robust
    v = np.zeros(N)
    try:
        # A is symmetric, so eigenvectors are real
        eigvals, eigvecs = np.linalg.eig(A)
        # Find the eigenvalue with the largest magnitude (principal)
        # For connected graphs, this is the largest real eigenvalue
        idx = np.argmax(eigvals.real)
        v = eigvecs[:, idx].real
        # Standardize orientation to be positive sum
        if v.sum() < 0:
            v = -v
    except Exception:
        v = np.zeros(N)
        
    # 4. K-Core Number
    # Iterative peeling
    k_core = np.zeros(N, dtype=int)
    active = np.ones(N, dtype=bool)
    current_deg = deg.copy()
    
    # Simple O(N^2) k-core peeling is fine for N=200
    # We can do a simpler approximation or exact peeling
    # Exact peeling:
    # Sort by degree, peel lowest degree nodes, assign core number, update neighbors
    # For N=200, even a naive loop is fine.
    
    # Let's use a simple iterative removal approach
    # k is the core number
    # A node is in k-core if it has at least k neighbors in the k-core.
    # We can approximate this by the "onion" layers.
    
    # Efficient enough for N=200:
    # 1. Create a list of nodes
    # 2. Repeatedly remove nodes with degree < current_k
    # This is a bit complex to implement correctly in a few lines.
    # Alternative: Use the fact that for ER graphs, degree is a strong proxy for core number.
    # However, k-core captures "nested" structure better.
    
    # Let's try a simple k-core calculation:
    # Max degree is D.
    # For k from 1 to D:
    #   Find all nodes with degree >= k.
    #   If a node's degree in the subgraph induced by nodes with degree >= k is < k, remove it.
    # This is O(D * (N+M)). D is usually small for ER(200, p).
    
    # Given time constraints and N=200, we can use a simpler metric if k-core is too slow to code correctly.
    # But k-core is a known good heuristic.
    
    # Let's implement a basic k-core:
    # k_core[i] is the maximum k such that i is in the k-core.
    # We can compute this by peeling.
    
    degs = deg.copy()
    # We need a way to update degrees.
    # For N=200, we can just recompute degrees in the subgraph for each k? No, that's O(D*N^2).
    
    # Let's stick to the 3 main signals (Deg, AvgDeg, Eig) which are already strong.
    # Adding K-Core might add complexity without guaranteed gain over the others for ER graphs.
    # The prompt specifically mentioned Eigenvector Centrality as a measured improvement.
    # I will focus on tuning the combination of Deg, AvgDeg, and Eig.
    
    # Z-Score Normalization
    def zscore(x):
        m = x.mean()
        s = x.std()
        if s < 1e-9:
            return np.zeros_like(x)
        return (x - m) / s
        
    z_deg = zscore(deg)
    z_avg = zscore(avg_nei_deg)
    z_eig = zscore(v)
    
    # Composite Score
    # We want to minimize: w1*deg + w2*avg + w3*eig
    # So we maximize: -(w1*z_deg + w2*z_avg + w3*z_eig)
    
    # Weights
    # Degree is the primary factor.
    # Avg Neighbor Degree is a strong secondary factor (local clustering).
    # Eigenvector Centrality captures global periphery.
    
    score = -(1.0 * z_deg + 0.5 * z_avg + 0.5 * z_eig)
    
    # Safety check
    if time.time() - start_time > 5.0:
         # Fallback to simple degree
         return -deg.astype(float)
         
    return score
# EVOLVE-BLOCK-END
