"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """ADMM + Local Search for Max-Cut with fast O(n) gain computation."""
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Deadline guard: 150s wall-clock limit, return best with safety margin
    deadline = time.time() + 138.0
    rng = np.random.default_rng(42)
    
    def cut_size(labels):
        """Compute cut size: number of edges crossing the partition."""
        cross = np.zeros(n, dtype=np.int32)
        for v in range(n):
            cross[v] = adj[v][labels != labels[v]].sum()
        return int(np.sum(cross) // 2)
    
    def flip_gain(labels, v):
        """O(deg(v)) computation of gain if we flip vertex v."""
        cur_cross = adj[v][labels != labels[v]].sum()
        new_cross = adj[v][labels == labels[v]].sum()
        return int(new_cross - cur_cross)
    
    def local_improve(labels):
        """Greedy local search: flip any vertex that improves the cut."""
        improved = True
        while improved:
            improved = False
            for v in range(n):
                if flip_gain(labels, v) > 0:
                    labels[v] = 1 - labels[v]
                    improved = True
        return labels
    
    def admm_maxcut(adj_matrix, rho=1.0, sigma=0.01, max_iter=1000):
        """ADMM solver for Max-Cut formulated as QUBO.
        
        Max-Cut: maximize sum_{i<j} A_ij * (1 - x_i * x_j) / 2
                 = constant - 1/4 * sum_{i,j} A_ij * (x_i - x_j)^2
        
        Equivalent to minimizing: 1/4 * sum_{i,j} A_ij * (x_i - x_j)^2
        subject to x_i in {-1, +1}
        
        ADMM formulation:
        min  1/4 * ||Dx||_A^2
        s.t. x = z, z in {-1,+1}^n
        
        Where D is the incidence matrix and A is the adjacency matrix.
        """
        # Convert to signed representation: x in {-1, +1}
        # Objective: minimize 1/4 * sum_{i,j} A_ij * (x_i - x_j)^2
        #           = 1/2 * x^T * L * x
        # where L = D - A is the graph Laplacian
        
        degrees = adj_matrix.sum(axis=1).astype(np.float64)
        L = np.diag(degrees) - adj_matrix.astype(np.float64)
        
        # ADMM variables
        x = np.ones(n, dtype=np.float64)  # Primal
        z = np.ones(n, dtype=np.float64)  # Dual/auxiliary
        u = np.zeros(n, dtype=np.float64)  # Dual variable
        
        best_x = x.copy()
        best_obj = np.inf
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # x-update: minimize 1/2 * x^T L x + rho/2 * ||x - z + u||^2
            # Gradient: L*x + rho*(x - z + u) = 0
            # (L + rho*I) * x = rho*(z - u)
            try:
                x_new = np.linalg.solve(L + rho * np.eye(n), rho * (z - u))
            except np.linalg.LinAlgError:
                # Fallback: use inverse if solve fails
                try:
                    x_new = np.linalg.inv(L + rho * np.eye(n)) @ (rho * (z - u))
                except:
                    x_new = x.copy()
            
            # z-update: project onto {-1, +1}
            z_new = np.sign(x_new + u)
            z_new[z_new == 0] = 1.0  # Handle zeros
            
            # Update dual variable
            u = u + (x_new - z_new)
            
            x, z = x_new, z_new
            
            # Track best solution
            x_sign = np.sign(x)
            x_sign[x_sign == 0] = 1.0
            obj = 0.5 * x_sign @ L @ x_sign
            if obj < best_obj:
                best_obj = obj
                best_x = x_sign.copy()
            
            # Convergence check
            if it > 10:
                dx = np.linalg.norm(x_new - x) / max(np.linalg.norm(x), 1e-10)
                dz = np.linalg.norm(z_new - z) / max(np.linalg.norm(z), 1e-10)
                if dx < 1e-6 and dz < 1e-6:
                    break
        
        # Convert back to 0/1 labels
        labels = (best_x > 0).astype(np.int8)
        return labels
    
    def admm_with_restart(adj_matrix, rho_values=None, max_iter=500):
        """Run ADMM with multiple rho values for robustness."""
        if rho_values is None:
            rho_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        best_labels = None
        best_cut = -1
        
        for rho in rho_values:
            if time.time() > deadline:
                break
            
            labels = admm_maxcut(adj_matrix, rho=rho, max_iter=max_iter)
            cut = cut_size(labels)
            
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
        
        return best_labels, best_cut
    
    # Multiple restarts with diverse initial solutions
    best_labels = None
    best_cut = -1
    
    # Initial solutions
    initial_solutions = []
    initial_solutions.append(np.arange(n) % 2)
    initial_solutions.append(rng.integers(0, 2, n).astype(np.int8))
    initial_solutions.append(rng.integers(0, 2, n).astype(np.int8))
    degrees = adj.sum(axis=1)
    initial_solutions.append((degrees > degrees.mean()).astype(np.int8))
    
    # Spectral initialization
    try:
        from scipy.linalg import eigh
        L = np.diag(degrees.astype(np.float64)) - adj.astype(np.float64)
        vals, vecs = eigh(L)
        v = vecs[:, 1]
        spectral_labels = (v > v.mean()).astype(np.int8)
        initial_solutions.append(spectral_labels)
    except:
        pass
    
    # ADMM initialization
    if time.time() < deadline - 30:
        admm_labels, admm_cut = admm_with_restart(adj, rho_values=[0.5, 1.0, 2.0], max_iter=300)
        if admm_labels is not None:
            initial_solutions.append(admm_labels)
    
    for init_labels in initial_solutions:
        if time.time() > deadline:
            break
        
        labels = init_labels.copy().astype(np.int8)
        labels = local_improve(labels)
        cur_cut = cut_size(labels)
        
        if cur_cut > best_cut:
            best_cut = cur_cut
            best_labels = labels.copy()
    
    # Final high-precision polish pass on the best solution
    if best_labels is not None and time.time() < deadline - 10:
        # Run ADMM from the best solution as a starting point
        if time.time() < deadline - 20:
            # Try ADMM with the current best as initialization
            x_init = (2 * best_labels.astype(np.float64) - 1)  # Convert to {-1, +1}
            
            # Run a few ADMM iterations starting from this point
            rho = 1.0
            z = x_init.copy()
            u = np.zeros(n, dtype=np.float64)
            
            degrees_float = degrees.astype(np.float64)
            L = np.diag(degrees_float) - adj.astype(np.float64)
            
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                
                try:
                    x_new = np.linalg.solve(L + rho * np.eye(n), rho * (z - u))
                except:
                    break
                
                z_new = np.sign(x_new + u)
                z_new[z_new == 0] = 1.0
                u = u + (x_new - z_new)
                x, z = x_new, z_new
            
            # Convert and polish
            admm_polished = (x > 0).astype(np.int8)
            admm_polished = local_improve(admm_polished)
            admm_polished_cut = cut_size(admm_polished)
            
            if admm_polished_cut > best_cut:
                best_cut = admm_polished_cut
                best_labels = admm_polished
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
