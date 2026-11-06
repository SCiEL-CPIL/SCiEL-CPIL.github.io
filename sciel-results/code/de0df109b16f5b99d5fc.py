"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """
    Maximum Cut via Goemans-Williamson SDP Relaxation, Randomized Rounding,
    Simulated Annealing, and Local Search.
    
    1. SDP Relaxation via Projected Gradient Descent:
       Solve the Goemans-Williamson relaxation:
         maximize 0.5 * sum_{ij} A_ij * (1 - v_i^T v_j) / 2
         subject to v_i^T v_i = 1
       using projected gradient descent on the matrix X = V^T V.
       
    2. Randomized Hyperplane Rounding:
       Draw random unit vectors and round based on sign(v_i^T r) to generate
       diverse high-quality initial partitions from the SDP solution.
       
    3. Simulated Annealing with Fast Delta Updates:
       - Maintain sign vector s in {-1, 1}^n.
       - Maintain As = A @ s for O(n) delta calculation per flip.
       - Accept moves probabilistically based on delta and temperature.
       - Gradually cool temperature.
       
    4. Local Polish:
       - After SA, do a greedy local search pass to fix any remaining
         improving moves.
       
    5. Time Guard:
       - Track elapsed time and return best solution found if deadline
         approaches (145s safety margin for 150s limit).
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Precompute adjacency as a numpy array
    adj = np.asarray(adjacency, dtype=np.float64)
    adj = (adj + adj.T) / 2.0
    np.fill_diagonal(adj, 0)
    
    # Helper to compute cut size
    def calc_cut(s):
        total_edges = 0.5 * np.sum(adj)
        val = s @ adj @ s
        return total_edges - 0.5 * val
    
    # 1. SDP Relaxation via Projected Gradient Descent
    # X is n x n PSD matrix with unit diagonal
    # Objective: maximize 0.25 * sum_{ij} A_ij * (1 - X_ij) = const - 0.25 * sum A_ij X_ij
    # Gradient of -0.25 * sum A_ij X_ij w.r.t. X is -0.25 * A (symmetric)
    # So we want to minimize 0.25 * sum A_ij X_ij, gradient is 0.25 * A
    
    # Initialize X as identity (all vectors orthogonal)
    X = np.eye(n, dtype=np.float64)
    
    # Number of SDP iterations (keep modest since we have time budget)
    sdp_iters = 200
    lr = 0.01
    
    for it in range(sdp_iters):
        if time.time() > deadline - 100:  # Leave time for SA
            break
        
        # Gradient of objective 0.25 * trace(A^T X) is 0.25 * A (since A is symmetric)
        grad = 0.25 * adj
        
        # Update
        X = X - lr * grad
        
        # Project onto PSD cone with unit diagonal
        # Eigendecomposition
        try:
            eigvals, eigvecs = np.linalg.eigh(X)
            # Clip negative eigenvalues
            eigvals = np.maximum(eigvals, 0)
            X = eigvecs @ np.diag(eigvals) @ eigvecs.T
            # Normalize diagonal to 1
            diag = np.sqrt(np.diag(X))
            diag[diag < 1e-10] = 1e-10
            X = X / np.outer(diag, diag)
        except Exception:
            break
    
    # Extract vectors V from X = V^T V
    # Use eigendecomposition of X
    try:
        eigvals, eigvecs = np.linalg.eigh(X)
        # V is n x n where rows are unit vectors
        V = eigvecs * np.sqrt(np.maximum(eigvals, 0))
        # Ensure rows have unit norm
        row_norms = np.sqrt(np.sum(V**2, axis=1, keepdims=True))
        row_norms[row_norms < 1e-10] = 1e-10
        V = V / row_norms
    except Exception:
        V = np.eye(n)
    
    # 2. Randomized Hyperplane Rounding to generate diverse initializations
    rng = np.random.default_rng()
    
    def generate_rounding(V, num_rounds=10):
        """Generate multiple partitions using random hyperplane rounding."""
        partitions = []
        for _ in range(num_rounds):
            # Random unit vector
            r = rng.standard_normal(n)
            r /= (np.linalg.norm(r) + 1e-10)
            # Round: sign(V @ r)
            s = np.sign(V @ r)
            s[s == 0] = 1  # Handle zeros
            partitions.append(s)
        return partitions
    
    # Generate diverse initial partitions
    initial_partitions = generate_rounding(V, num_rounds=15)
    
    # Also add spectral initialization as a fallback
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(adj)
        idx_min = np.argmin(eigenvalues)
        u = eigenvectors[:, idx_min]
        s_spec = (u >= 0).astype(np.float64) * 2 - 1
        initial_partitions.append(s_spec)
    except Exception:
        pass
    
    # 3. Simulated Annealing on each initial partition, keep best
    best_cut = -1
    best_s = None
    
    T0 = 10.0
    T_min = 0.1
    cooling_rate = 0.995
    
    for init_s in initial_partitions:
        if time.time() > deadline - 5:
            break
        
        s = init_s.copy()
        labels = ((s + 1) / 2).astype(np.int8)
        As = adj @ s
        current_cut = calc_cut(s)
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_s = s.copy()
        
        T = T0
        
        while T > T_min:
            if time.time() > deadline - 5:
                break
            
            # Randomly select a vertex to flip
            v = rng.integers(n)
            
            # Calculate delta for flipping v
            delta = 2 * s[v] * As[v]
            
            # Accept if improving, or with probability exp(delta/T) if worsening
            if delta >= 0:
                accept = True
            else:
                accept = rng.random() < np.exp(delta / T)
            
            if accept:
                # Apply flip
                old_s_v = s[v]
                s[v] = -old_s_v
                labels[v] = 1 - labels[v]
                
                # Update As
                As -= 2 * old_s_v * adj[v]
                
                # Update current cut
                current_cut += delta
                
                # Update best
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_s = s.copy()
            
            # Cool down
            T *= cooling_rate
            
            # Occasionally do a local polish to escape local optima
            if rng.random() < 0.01:
                # Quick local polish: flip any vertex with positive delta
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    for v in range(n):
                        delta = 2 * s[v] * As[v]
                        if delta > 0:
                            old_s_v = s[v]
                            s[v] = -old_s_v
                            labels[v] = 1 - labels[v]
                            As -= 2 * old_s_v * adj[v]
                            current_cut += delta
                            improved = True
                            if current_cut > best_cut:
                                best_cut = current_cut
                                best_s = s.copy()
        
        # Final local polish for this partition
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for v in range(n):
                delta = 2 * s[v] * As[v]
                if delta > 0:
                    old_s_v = s[v]
                    s[v] = -old_s_v
                    labels[v] = 1 - labels[v]
                    As -= 2 * old_s_v * adj[v]
                    current_cut += delta
                    improved = True
                    if current_cut > best_cut:
                        best_cut = current_cut
                        best_s = s.copy()
    
    # 4. Final local polish on the best solution
    if best_s is not None:
        s = best_s.copy()
        labels = ((s + 1) / 2).astype(np.int8)
        As = adj @ s
        current_cut = calc_cut(s)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v in range(n):
                delta = 2 * s[v] * As[v]
                if delta > 0:
                    old_s_v = s[v]
                    s[v] = -old_s_v
                    labels[v] = 1 - labels[v]
                    As -= 2 * old_s_v * adj[v]
                    current_cut += delta
                    improved = True
                    if current_cut > best_cut:
                        best_cut = current_cut
                        best_s = s.copy()
    
    # Return best labels found
    best_labels = ((best_s + 1) / 2).astype(np.int8)
    return best_labels
# EVOLVE-BLOCK-END
