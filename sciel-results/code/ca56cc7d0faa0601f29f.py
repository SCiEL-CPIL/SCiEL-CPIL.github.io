"""Improved program for Maximum Independent Set greedy scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_size(adjacency_bool, order):
    """Compute independent set size given a vertex order (indices sorted by priority)."""
    n = len(order)
    available = np.ones(n, dtype=bool)
    count = 0
    for v in order:
        if available[v]:
            count += 1
            available[v] = False
            available[adjacency_bool[v]] = False
    return count


def _evaluate_order(adjacency_bool, order):
    """Evaluate a priority vector by running the greedy decoder."""
    return _greedy_size(adjacency_bool, order)


def _power_iteration_eigenvector(M, n, iters=100, tol=1e-10):
    """
    Fast power iteration for the principal eigenvector of a symmetric matrix.
    Returns the eigenvector (normalized to unit L2 norm).
    """
    v = np.random.default_rng(0).normal(size=n)
    v /= np.linalg.norm(v)
    for _ in range(iters):
        new_v = M @ v
        norm = np.linalg.norm(new_v)
        if norm < 1e-12:
            break
        new_v /= norm
        if np.abs(np.dot(new_v, v)) > 1 - tol:
            v = new_v
            break
        v = new_v
    return v


def _spectral_priority(A, n):
    """
    Generate spectral scores using power iteration on the adjacency matrix.
    Vertices with high principal eigenvector values are in dense regions.
    For MIS greedy, we want to pick peripheral (low centrality) vertices first.
    """
    A_float = A.astype(float)
    principal = _power_iteration_eigenvector(A_float, n, iters=200)
    principal = np.abs(principal)
    
    p_min = principal.min()
    p_max = principal.max()
    if p_max > p_min:
        principal_norm = (principal - p_min) / (p_max - p_min)
    else:
        principal_norm = np.zeros(n)
    
    return principal_norm


def _laplacian_fiedler(A, n):
    """
    Compute the Fiedler vector using power iteration on (2*L_max*I - L)
    to get the smallest non-trivial eigenvalue, or use inverse iteration.
    For simplicity, use a shifted power method or just use the second
    eigenvector via a different approach.
    
    Alternative: Use the fact that for a connected graph, the Fiedler vector
    is the eigenvector of L with the second smallest eigenvalue.
    We can approximate this by power iteration on (c*I - L) where c > lambda_2.
    """
    deg = A.sum(axis=1).astype(float)
    L = np.diag(deg) - A.astype(float)
    
    # Estimate lambda_max of L
    # For ER graphs, lambda_max(L) ≈ 2*avg_degree + something
    avg_deg = deg.mean()
    c = 2 * avg_deg + 5  # Should be > lambda_2 for connected graph
    
    # Power iteration on (c*I - L) to get the smallest eigenvalue of L
    # Actually, this gives the eigenvector for the largest eigenvalue of (cI - L),
    # which corresponds to the smallest eigenvalue of L.
    # We want the second smallest, so we need to shift differently.
    
    # Let's just use np.linalg.eig for n=200, it's fast enough
    try:
        eigenvalues, eigenvectors = np.linalg.eig(L)
        idx = np.argsort(eigenvalues.real)
        fiedler = np.abs(eigenvectors[:, idx[1]])
    except:
        fiedler = np.ones(n) / n
    
    f_min = fiedler.min()
    f_max = fiedler.max()
    if f_max > f_min:
        fiedler_norm = (fiedler - f_min) / (f_max - f_min)
    else:
        fiedler_norm = np.zeros(n)
    
    return fiedler_norm


def _marginal_loss_scores(A, n, deg):
    """
    Compute scores based on marginal loss of selecting a vertex.
    When we select vertex v, we lose all its neighbors.
    The "cost" of selecting v is the number of neighbors that would be
    unavailable. But we also gain v itself.
    
    A better heuristic: for each vertex, compute how many of its neighbors
    have high degree (i.e., selecting v "saves" those high-degree neighbors
    from being selected, which is bad for MIS).
    
    Actually, for greedy MIS, picking a low-degree vertex first is good
    because it removes fewer vertices. But if a low-degree vertex is
    connected to many high-degree vertices, picking it removes those
    high-degree vertices, which might be good (since high-degree vertices
    would remove even more if picked).
    
    Let's compute: for each vertex v, the sum of degrees of its neighbors.
    Lower is better (picking v removes low-degree neighbors).
    """
    # Sum of neighbor degrees
    nbr_deg_sum = A.astype(float) @ deg
    return nbr_deg_sum


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    n = adjacency.shape[0]
    A = adjacency.astype(bool)
    
    # Precompute degrees
    deg = A.sum(axis=1).astype(float)
    
    # Compute neighborhood edge counts for each vertex (clustering proxy)
    C = np.zeros(n)
    for v in range(n):
        if deg[v] < 2:
            C[v] = 0.0
        else:
            nbrs = np.where(A[v])[0]
            if len(nbrs) > 0:
                sub = A[np.ix_(nbrs, nbrs)]
                C[v] = sub.sum() / 2.0
    
    # Spectral scores
    spectral = _spectral_priority(A, n)
    fiedler = _laplacian_fiedler(A, n)
    
    # Marginal loss scores
    nbr_deg_sum = _marginal_loss_scores(A, n, deg)
    
    # Heuristic 1: Ascending Degree
    order1 = np.argsort(deg, kind='stable')
    
    # Heuristic 2: Ascending Degree, then Ascending Clustering
    score_h2 = deg + 0.1 * C
    order2 = np.argsort(score_h2, kind='stable')
    
    # Heuristic 3: Random
    rng = np.random.default_rng(42)
    order3 = rng.permutation(n)
    
    # Heuristic 4: Degree + Noise
    noise = rng.normal(0, 0.5, size=n)
    score_h4 = deg + noise
    order4 = np.argsort(score_h4, kind='stable')
    
    # Heuristic 5: Spectral-based - low spectral score first (peripheral vertices)
    score_h5 = deg * 0.5 + spectral * 0.5
    order5 = np.argsort(score_h5, kind='stable')
    
    # Heuristic 6: Fiedler-based - vertices with extreme Fiedler values
    score_h6 = deg * 0.5 + fiedler * 0.5
    order6 = np.argsort(score_h6, kind='stable')
    
    # Heuristic 7: Combined - degree, clustering, spectral
    score_h7 = deg * 0.4 + C * 0.1 + spectral * 0.3 + fiedler * 0.2
    order7 = np.argsort(score_h7, kind='stable')
    
    # Heuristic 8: Marginal loss based
    score_h8 = deg * 0.5 + nbr_deg_sum * 0.05
    order8 = np.argsort(score_h8, kind='stable')
    
    # Heuristic 9: Combined with marginal loss
    score_h9 = deg * 0.3 + C * 0.1 + spectral * 0.2 + fiedler * 0.1 + nbr_deg_sum * 0.05
    order9 = np.argsort(score_h9, kind='stable')
    
    candidates = [order1, order2, order3, order4, order5, order6, order7, order8, order9]
    
    best_size = -1
    best_order = None
    
    # Evaluate initial candidates
    for order in candidates:
        if time.time() > deadline:
            break
        size = _evaluate_order(A, order)
        if size > best_size:
            best_size = size
            best_order = order.copy()
            
    # Local Search: Multi-start Hill Climbing with insertion moves
    num_restarts = 30
    max_iters_per_restart = 2000
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize current order for this restart
        if r == 0:
            current_order = best_order.copy()
        elif r % 3 == 1:
            # Perturb the best order found so far
            current_order = best_order.copy()
            for _ in range(10):
                i, j = rng.integers(0, n, size=2)
                current_order[i], current_order[j] = current_order[j], current_order[i]
        elif r % 3 == 2:
            # Start from a random heuristic
            current_order = candidates[r % len(candidates)].copy()
            for _ in range(5):
                i, j = rng.integers(0, n, size=2)
                current_order[i], current_order[j] = current_order[j], current_order[i]
        else:
            # Start from a new random order
            current_order = rng.permutation(n)
                
        current_size = _evaluate_order(A, current_order)
        
        # If this restart starts better than global best, update
        if current_size > best_size:
            best_size = current_size
            best_order = current_order.copy()
            
        no_improve_count = 0
        max_no_improve = 100
        
        for i in range(max_iters_per_restart):
            if time.time() > deadline - 5:
                break
                
            if no_improve_count > max_no_improve:
                break
                
            # Move: Pick a random vertex and move it to a random position
            v_idx = rng.integers(0, n)
            target_pos = rng.integers(0, n)
            
            if v_idx == target_pos:
                continue
                
            v = current_order[v_idx]
            
            new_order = current_order.copy()
            new_order = np.delete(new_order, v_idx)
            if v_idx < target_pos:
                insert_idx = target_pos - 1
            else:
                insert_idx = target_pos
            new_order = np.insert(new_order, insert_idx, v)
            
            new_size = _evaluate_order(A, new_order)
            
            if new_size > current_size:
                current_order = new_order
                current_size = new_size
                no_improve_count = 0
                if current_size > best_size:
                    best_size = current_size
                    best_order = current_order.copy()
            else:
                no_improve_count += 1
                
    # Final polish: try adjacent swaps around the best order
    if time.time() < deadline - 2:
        polish_order = best_order.copy()
        polish_size = _evaluate_order(A, polish_order)
        for _ in range(500):
            if time.time() > deadline - 1:
                break
            i = rng.integers(0, n - 1)
            j = i + 1
            polish_order[i], polish_order[j] = polish_order[j], polish_order[i]
            swap_size = _evaluate_order(A, polish_order)
            if swap_size > polish_size:
                polish_size = swap_size
                if polish_size > best_size:
                    best_size = polish_size
                    best_order = polish_order.copy()
            else:
                polish_order[i], polish_order[j] = polish_order[j], polish_order[i]
                
    # Convert best_order to priority vector
    # Higher priority means selected earlier.
    # So vertex at index 0 in order gets highest priority.
    # priority[v] = n - index(v)
    priority_vec = np.zeros(n)
    for idx, v in enumerate(best_order):
        priority_vec[v] = n - idx
        
    return priority_vec.astype(float)
# EVOLVE-BLOCK-END
