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
    # Work directly with the vector matrix V (n x n) where rows are unit vectors
    # X = V @ V.T, objective: maximize 0.25 * sum_ij A_ij * (1 - X_ij)
    # = const - 0.25 * sum_ij A_ij * (V_i . V_j)
    # Gradient w.r.t. V_i: -0.25 * sum_j A_ij * V_j
    
    # Initialize V as random unit vectors
    rng_init = np.random.default_rng(42)
    V = rng_init.standard_normal((n, n))
    row_norms = np.linalg.norm(V, axis=1, keepdims=True)
    row_norms[row_norms < 1e-10] = 1e-10
    V = V / row_norms
    
    # Number of SDP iterations
    sdp_iters = 300
    lr = 0.05
    
    for it in range(sdp_iters):
        if time.time() > deadline - 100:  # Leave time for SA
            break
        
        # Gradient w.r.t. V: -0.25 * A @ V
        grad_V = -0.25 * (adj @ V)
        
        # Update
        V = V + lr * grad_V
        
        # Project each row to unit norm (simplest constraint satisfaction)
        row_norms = np.linalg.norm(V, axis=1, keepdims=True)
        row_norms[row_norms < 1e-10] = 1e-10
        V = V / row_norms
        
        # Periodically check PSD-ness by ensuring X = V@V.T has non-negative eigenvalues
        # For efficiency, only do full eigendecomp every 50 iterations
        if (it + 1) % 50 == 0:
            try:
                X_check = V @ V.T
                eigvals_check = np.linalg.eigvalsh(X_check)
                if np.min(eigvals_check) < -1e-6:
                    # Full projection: eigendecompose and clip
                    eigvals_full, eigvecs_full = np.linalg.eigh(X_check)
                    eigvals_clipped = np.maximum(eigvals_full, 0)
                    X_projected = eigvecs_full @ np.diag(eigvals_clipped) @ eigvecs_full.T
                    # Re-extract V from projected X
                    eigvals_v, eigvecs_v = np.linalg.eigh(X_projected)
                    V_new = eigvecs_v * np.sqrt(np.maximum(eigvals_v, 0))
                    row_norms_v = np.linalg.norm(V_new, axis=1, keepdims=True)
                    row_norms_v[row_norms_v < 1e-10] = 1e-10
                    V = V_new / row_norms_v
            except Exception:
                pass
    
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
    
    # Add a few random initializations for diversity
    for _ in range(3):
        s_rand = rng.choice([-1.0, 1.0], size=n)
        initial_partitions.append(s_rand)
    
    # 3. Simulated Annealing on each initial partition, keep best
    best_cut = -1
    best_s = None
    
    T0 = 15.0
    T_min = 0.05
    cooling_rate = 0.998
    
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
    
    # 4. Final high-precision local polish on the best solution
    # Use multiple passes with decreasing tolerance for tighter convergence
    if best_s is not None:
        s = best_s.copy()
        labels = ((s + 1) / 2).astype(np.int8)
        As = adj @ s
        current_cut = calc_cut(s)
        
        # High-precision polish: multiple passes, tighter tolerance
        for polish_pass in range(3):
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for v in range(n):
                    delta = 2 * s[v] * As[v]
                    if delta > 1e-10:  # Very tight tolerance
                        old_s_v = s[v]
                        s[v] = -old_s_v
                        labels[v] = 1 - labels[v]
                        As -= 2 * old_s_v * adj[v]
                        current_cut += delta
                        improved = True
                        if current_cut > best_cut:
                            best_cut = current_cut
                            best_s = s.copy()
        
        # Pairwise swap polish: try swapping pairs of vertices
        if time.time() < deadline - 2:
            s = best_s.copy()
            As = adj @ s
            current_cut = calc_cut(s)
            
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try all pairs (sample if too many)
                num_pairs = n * (n - 1) // 2
                if num_pairs <= 5000:
                    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
                else:
                    pairs = [(rng.integers(n), rng.integers(n)) for _ in range(5000)]
                    pairs = [(i, j) for i, j in pairs if i != j]
                
                for i, j in pairs:
                    if time.time() > deadline:
                        break
                    
                    # Delta for swapping i and j
                    # Flipping i: delta_i = 2 * s[i] * As[i]
                    # Flipping j: delta_j = 2 * s[j] * As[j]
                    # But when both flip, there's a cross term
                    delta_i = 2 * s[i] * As[i]
                    delta_j = 2 * s[j] * As[j]
                    # Cross term: -2 * s[i] * s[j] * adj[i,j] * 2 (from the interaction)
                    # Actually: new_cut - old_cut = delta_i + delta_j - 2*adj[i,j]*s[i]*s[j]*2
                    # Let me recompute: flipping both i and j changes cut by
                    # (2*s[i]*As[i] - 2*adj[i,j]*s[i]*s[j]) + (2*s[j]*As[j] - 2*adj[i,j]*s[i]*s[j])
                    # Wait, let's just compute directly
                    delta_swap = delta_i + delta_j - 4 * adj[i,j] * s[i] * s[j]
                    
                    if delta_swap > 1e-10:
                        old_si = s[i]
                        old_sj = s[j]
                        s[i] = -old_si
                        s[j] = -old_sj
                        labels[i] = 1 - labels[i]
                        labels[j] = 1 - labels[j]
                        
                        # Update As for all vertices affected by i and j flips
                        As -= 2 * old_si * adj[i]
                        As -= 2 * old_sj * adj[j]
                        
                        current_cut += delta_swap
                        improved = True
                        if current_cut > best_cut:
                            best_cut = current_cut
                            best_s = s.copy()
    
    # Return best labels found
    best_labels = ((best_s + 1) / 2).astype(np.int8)
    return best_labels
# EVOLVE-BLOCK-END
