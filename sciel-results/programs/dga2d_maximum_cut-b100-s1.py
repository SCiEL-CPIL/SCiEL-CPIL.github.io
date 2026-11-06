"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def partition(adjacency):
    """
    Maximum Cut via Spectral Initialization and Tabu Search.
    
    1. Spectral Initialization: 
       Max Cut is equivalent to minimizing x^T A x for x in {-1,1}^n.
       Relaxing x to unit vectors, the minimum is the smallest eigenvalue.
       We use the eigenvector of the smallest eigenvalue to initialize signs.
       
    2. Tabu Search:
       A metaheuristic that explores the neighborhood of the current solution,
       forbidding recent moves (tabu list) to escape local optima.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Precompute adjacency as a numpy array
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Ensure symmetry (just in case)
    adj = (adj + adj.T) / 2.0
    np.fill_diagonal(adj, 0)
    
    # 1. Spectral Initialization
    # Compute eigenvalues and eigenvectors
    # For n=100, this is fast.
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(adj)
        # We want the eigenvector corresponding to the smallest eigenvalue
        # to minimize x^T A x, which maximizes the cut.
        idx_min = np.argmin(eigenvalues)
        u = eigenvectors[:, idx_min]
        
        # Round to 0/1 based on sign
        # If u[i] >= 0, label 1, else 0.
        labels = (u >= 0).astype(np.int8)
        
    except Exception:
        # Fallback to random if eigh fails
        labels = np.random.randint(0, 2, size=n).astype(np.int8)
        
    # Helper to compute cut size efficiently
    def calc_cut(labels):
        # Map 0/1 to -1/1
        s = 2 * labels - 1
        # Cut size = 0.5 * (n_edges_total - s^T A s)
        # Or simply sum of edges where s[i] != s[j]
        # Efficient way:
        # s^T A s = sum_i sum_j s_i A_ij s_j
        val = s @ adj @ s
        # Total weight of edges = 0.5 * sum(adj)
        total_edges = 0.5 * np.sum(adj)
        return total_edges - 0.5 * val

    current_cut = calc_cut(labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    # 2. Tabu Search
    # Parameters
    tabu_size = 30
    max_iter = 20000
    no_improve_limit = 500
    
    tabu_list = {} # vertex -> iteration when it becomes non-tabu
    iteration = 0
    no_improve_count = 0
    
    # Precompute degrees for quick delta calculation?
    # Delta for flipping vertex v:
    # Let s_v be current sign. New sign is -s_v.
    # Change in s^T A s:
    # Old term involving v: 2 * s_v * (A @ s)_v  (since A is symmetric and diag 0)
    # New term: 2 * (-s_v) * (A @ (-s_v) + sum_{u!=v} A_vu s_u) ? No.
    # Let's derive delta for cut size directly.
    # Cut size C = 0.5 * sum_{i,j} A_ij (1 - s_i s_j)
    # Flipping v changes s_v to -s_v.
    # Terms affected: (v, u) and (u, v) for all u.
    # Old contribution of (v,u) to sum: A_vu (1 - s_v s_u)
    # New contribution: A_vu (1 - (-s_v) s_u) = A_vu (1 + s_v s_u)
    # Diff for (v,u): A_vu (2 s_v s_u)
    # Sum over u: 2 s_v sum_u A_vu s_u
    # Total sum change (double sum): 2 * (2 s_v sum_u A_vu s_u) = 4 s_v (A @ s)_v
    # Cut size change = 0.5 * (Double Sum Change) = 2 s_v (A @ s)_v
    # Note: (A @ s)_v is the weighted sum of neighbors' signs.
    
    s = 2 * labels - 1
    As = adj @ s # A @ s vector
    
    for _ in range(max_iter):
        # Find best move
        best_delta = -np.inf
        best_v = -1
        
        for v in range(n):
            # Calculate delta
            # delta = 2 * s[v] * As[v]
            delta = 2 * s[v] * As[v]
            
            # Tabu check
            is_tabu = False
            if v in tabu_list and tabu_list[v] > iteration:
                is_tabu = True
            
            # Aspiration criterion
            if is_tabu:
                if current_cut + delta > best_cut:
                    is_tabu = False
            
            if not is_tabu:
                if delta > best_delta:
                    best_delta = delta
                    best_v = v
        
        if best_v == -1:
            # All moves are tabu and not improving best. 
            # This is rare with size 30 and n=100, but possible.
            # Just pick a random move to escape.
            best_v = np.random.randint(n)
            best_delta = 2 * s[best_v] * As[best_v]
        
        # Apply move
        v = best_v
        old_s_v = s[v]
        
        # Update s
        s[v] = -old_s_v
        labels[v] = 1 - labels[v]
        
        # Update As vector
        # As[u] changes by -2 * old_s_v * A[v, u]
        # Because As[u] = sum_k A[u, k] s[k]. 
        # Term k=v was A[u, v] old_s_v. Now it is A[u, v] (-old_s_v).
        # Change = -2 * old_s_v * A[u, v].
        # Since A is symmetric, A[u, v] = A[v, u].
        As -= 2 * old_s_v * adj[v]
        
        # Update current cut
        current_cut += best_delta
        
        # Update best
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Update Tabu List
        tabu_list[v] = iteration + tabu_size
        
        iteration += 1
        
        # Diversification / Restart if stuck
        if no_improve_count > no_improve_limit:
            # Perturb: flip a few random vertices
            for _ in range(5):
                v_rand = np.random.randint(n)
                old_s_v = s[v_rand]
                s[v_rand] = -old_s_v
                labels[v_rand] = 1 - labels[v_rand]
                As -= 2 * old_s_v * adj[v_rand]
            # Recalculate cut accurately after perturbation
            current_cut = calc_cut(labels)
            no_improve_count = 0
            # Clear tabu list? Or keep? Let's clear to allow exploration.
            tabu_list.clear()

    return best_labels
# EVOLVE-BLOCK-END
