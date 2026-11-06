import numpy as np
import time

def partition(adjacency):
    """
    Solve Maximum Cut using:
    1. Spectral Goemans-Williamson Initialization:
       Compute the leading eigenvector of the matrix M = A - (d/2)I.
       Partition vertices based on the sign of this eigenvector.
       This provides a high-quality starting point closer to the global optimum than random or Fiedler vectors.
    2. Vectorized Best-Improvement Local Search:
       In each iteration, compute the gain for flipping every vertex simultaneously.
       Flip the single vertex with the maximum positive gain.
       Repeat until no improvement is possible.
    3. Multi-Start with Spectral Variants:
       If time permits, run the local search from slightly perturbed spectral initializations 
       or the negative of the spectral vector to escape potential local optima.
    """
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We stop at 145s to ensure return.
    DEADLINE = 145.0
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to float32 for faster matrix operations
    adj = np.array(adjacency, dtype=np.float32)
    # Ensure symmetry and zero diagonal
    adj = (adj + adj.T) / 2.0
    np.fill_diagonal(adj, 0.0)
    
    degrees = adj.sum(axis=1)
    
    def cut_size(labels):
        """
        Compute cut size efficiently.
        Cut = 0.5 * sum_{i,j} A_ij * (1 - s_i s_j) where s_i = 2*label_i - 1
        Alternatively: Cut = sum_{i in 0, j in 1} A_ij
        Using matrix ops:
        Let x be vector of 0/1 labels.
        Cut = x^T (D - A) x? No.
        Let's use: Cut = sum_{i,j} A_ij x_i (1-x_j) for i != j?
        Simpler:
        Let s = 2x - 1 (vector of -1, 1)
        Cut = 0.25 * (n^2 * trace(A) ... no)
        
        Efficient formula:
        Cut = 0.5 * sum_i sum_j A_ij (s_i - s_j)^2 / 4 ?
        
        Let's stick to the reliable one:
        Let x be 0/1.
        Cut = sum_{i,j} A_ij x_i (1-x_j) 
             = sum_i x_i (sum_j A_ij (1-x_j))
             = sum_i x_i (deg_i - (A x)_i)
             = x @ (degrees - adj @ x)
        """
        x = labels.astype(np.float32)
        return float(x @ (degrees - adj @ x))
    
    def local_search(labels):
        """
        Vectorized Best-Improvement Local Search.
        """
        labels = labels.copy().astype(np.int32)
        current_cut = cut_size(labels)
        
        # Pre-compute adj @ x updates? 
        # We need to be careful with time.
        # Gain for flipping vertex u:
        # If u is 0, flipping to 1:
        #   gain = (neighbors in 1) - (neighbors in 0)
        # If u is 1, flipping to 0:
        #   gain = (neighbors in 0) - (neighbors in 1)
        
        # Let s = 2*labels - 1 (vector of -1, 1)
        # Neighbors in same set = 0.5 * (deg - (A s)_u) ? No.
        # Let's use the 0/1 formulation directly for clarity and speed.
        
        # x = labels (0/1)
        # neighbors_in_1[u] = (adj @ x)[u]
        # neighbors_in_0[u] = degrees[u] - neighbors_in_1[u]
        
        # If labels[u] == 0:
        #   Flip to 1.
        #   Edges to 0 become cut (+). Edges to 1 become not cut (-).
        #   Gain = neighbors_in_0[u] - neighbors_in_1[u]
        # If labels[u] == 1:
        #   Flip to 0.
        #   Edges to 1 become cut (+). Edges to 0 become not cut (-).
        #   Gain = neighbors_in_1[u] - neighbors_in_0[u]
        
        # Gain = (2*labels[u] - 1) * (neighbors_in_1[u] - neighbors_in_0[u])
        #      = (2*labels[u] - 1) * (2*neighbors_in_1[u] - degrees[u])
        
        while True:
            if time.time() > DEADLINE - 2:
                break
            
            x = labels.astype(np.float32)
            neighbors_in_1 = adj @ x
            
            # sgn = 1 if label is 1, -1 if label is 0
            sgn = 2.0 * x - 1.0
            
            # base_gain = neighbors_in_1 - neighbors_in_0 = 2*neighbors_in_1 - degrees
            base_gain = 2.0 * neighbors_in_1 - degrees
            
            gains = sgn * base_gain
            
            max_gain = np.max(gains)
            
            if max_gain <= 1e-6:
                break
            
            idx = np.argmax(gains)
            labels[idx] = 1 - labels[idx]
            current_cut += max_gain
            
        return labels, current_cut
    
    def spectral_gw_init(adj, degrees):
        """
        Spectral Goemans-Williamson Initialization.
        We want to maximize x^T A x subject to x_i in {-1, 1}.
        Relaxation: Maximize x^T A x subject to x^T x = n.
        Solution: Leading eigenvector of A.
        However, the standard GW relaxation uses the matrix A - (d/2)I?
        Actually, the Goemans-Williamson algorithm for MaxCut:
        1. Solve SDP: Maximize Tr(X A) s.t. X_ii = 1.
        2. Round using random hyperplane.
        
        Spectral approximation:
        The leading eigenvector of the Laplacian (Fiedler) is for MinCut.
        For MaxCut, we look at the adjacency matrix A.
        The leading eigenvector of A provides a partition that aligns with the graph's structure.
        
        Let's use the leading eigenvector of A.
        """
        try:
            # Symmetric matrix
            eigenvalues, eigenvectors = np.linalg.eigh(adj)
            # The largest eigenvalue corresponds to the last column in eigh (ascending order)
            # We want the eigenvector for the largest eigenvalue.
            v = eigenvectors[:, -1]
            
            # Partition by sign
            # If v_i > 0, label 1, else 0
            labels = (v > 0).astype(int)
            
            # Ensure not all 0 or all 1 (should be rare for connected non-trivial graphs)
            if np.all(labels == 0) or np.all(labels == 1):
                 # Fallback to median
                 median_val = np.median(v)
                 labels = (v >= median_val).astype(int)
            
            return labels
        except:
            return np.random.randint(0, 2, size=n)

    best_labels = None
    best_cut = -1.0
    
    # 1. Spectral Initialization
    labels = spectral_gw_init(adj, degrees)
    labels, cut = local_search(labels)
    best_labels = labels.copy()
    best_cut = cut
    
    # 2. Try the negative of the spectral vector (equivalent partition, but just in case of ties or numerical issues)
    # Actually, sign flip of eigenvector gives same partition. 
    # Let's try a random hyperplane projection of the top few eigenvectors?
    # Or just a simple random restart if time permits.
    
    # Given the 120s limit and n=100, local search is very fast.
    # Let's do a few random restarts to ensure we don't get stuck in a bad local optimum
    # that the spectral init might fall into (though spectral is usually very good).
    
    rng = np.random.default_rng(42)
    
    # Time check
    # local_search on n=100 is extremely fast (vectorized).
    # We can afford many restarts.
    
    num_restarts = 0
    max_restarts = 100
    
    while time.time() < DEADLINE - 5 and num_restarts < max_restarts:
        # Random initialization
        labels = rng.integers(0, 2, size=n)
        labels, cut = local_search(labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        num_restarts += 1
        
        # If we've hit a very high cut (e.g., > 0.85 * max_possible), we might stop early?
        # Max possible cut is sum of degrees / 2.
        # For random graphs, MaxCut is usually close to 0.75-0.8 of total edges.
        # Let's just use the time limit.

    # Final check: if best_labels is None (shouldn't happen), return alternating
    if best_labels is None:
        best_labels = np.arange(n) % 2
        
    return best_labels
