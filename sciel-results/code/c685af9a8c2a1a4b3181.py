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


def _power_iteration_eigenvector(M, n, iters=50, tol=1e-8):
    """
    Fast power iteration for the principal eigenvector of a symmetric matrix.
    Returns the eigenvector (normalized to unit L2 norm).
    """
    rng = np.random.default_rng(0)
    v = rng.normal(size=n)
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


def _extract_features(A, n):
    """
    Extract features for each vertex:
    1. Degree
    2. Mean neighbor degree
    3. Spectral centrality (principal eigenvector)
    4. Fiedler vector value
    Returns a matrix of shape (n, 4).
    """
    deg = A.sum(axis=1).astype(float)
    
    # Mean neighbor degree
    # To avoid division by zero, add small epsilon
    mean_nbr_deg = np.zeros(n)
    non_zero_deg = deg > 0
    if np.any(non_zero_deg):
        nbr_deg_sum = A.astype(float) @ deg
        mean_nbr_deg[non_zero_deg] = nbr_deg_sum[non_zero_deg] / deg[non_zero_deg]
    
    # Spectral
    A_float = A.astype(float)
    principal = _power_iteration_eigenvector(A_float, n, iters=50)
    principal = np.abs(principal)
    p_min, p_max = principal.min(), principal.max()
    if p_max > p_min:
        spectral = (principal - p_min) / (p_max - p_min)
    else:
        spectral = np.zeros(n)
        
    # Fiedler (approximate using second smallest eigenvalue of Laplacian)
    # For speed, we might skip full eigendecomposition if n is large, but n=200 is fine.
    # However, to be safe and fast, we can use a simpler proxy or just use spectral again with different weighting?
    # Let's stick to a cheap proxy for the 4th feature: "Boundary" measure.
    # Vertices with high degree but low clustering are "boundary".
    # Clustering coefficient per node is expensive. 
    # Let's use: Degree - MeanNeighborDegree (negative if connected to hubs)
    fiedler_proxy = deg - mean_nbr_deg
    
    # Normalize features to [0, 1] roughly to help learning
    def norm_vec(v):
        mn, mx = v.min(), v.max()
        if mx > mn:
            return (v - mn) / (mx - mn)
        return np.zeros_like(v)
        
    deg_n = norm_vec(deg)
    mnd_n = norm_vec(mean_nbr_deg)
    spec_n = spectral # already normalized
    fiedler_n = norm_vec(fiedler_proxy)
    
    return np.column_stack([deg_n, mnd_n, spec_n, fiedler_n])


def _generate_er_graph(n, p, seed):
    rng = np.random.default_rng(seed)
    # Generate upper triangle
    u = rng.random((n, n))
    A = (u < p).astype(int)
    # Symmetrize
    A = np.maximum(A, A.T)
    # Zero diagonal
    np.fill_diagonal(A, 0)
    return A.astype(bool)


def _learn_weights(features_list, adj_list, time_limit=30.0):
    """
    Learn weights w for s = X @ w such that greedy size is maximized.
    Uses a simple random search / hill climbing on the weight vector.
    """
    start_time = time.time()
    n_features = features_list[0].shape[1]
    
    # Initialize weights randomly
    rng = np.random.default_rng(123)
    best_w = rng.normal(size=n_features)
    
    # Evaluate initial
    def score_w(w):
        total = 0
        for i, (X, A) in enumerate(zip(features_list, adj_list)):
            scores = X @ w
            order = np.argsort(-scores, kind='stable')
            total += _greedy_size(A, order)
        return total
        
    best_score = score_w(best_w)
    
    # Hill Climbing
    step_size = 0.1
    no_improve = 0
    max_no_improve = 20
    
    while time.time() - start_time < time_limit:
        if no_improve > max_no_improve:
            break
            
        # Perturb
        new_w = best_w + rng.normal(0, step_size, size=n_features)
        new_score = score_w(new_w)
        
        if new_score > best_score:
            best_w = new_w
            best_score = new_score
            no_improve = 0
            # Maybe reduce step size slightly
            step_size *= 0.95
        else:
            no_improve += 1
            # Increase step size if stuck
            if no_improve > 5:
                step_size *= 1.2
                if step_size > 1.0: step_size = 1.0
                no_improve = 0
                
    return best_w


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n = adjacency.shape[0]
    A = adjacency.astype(bool)
    
    # 1. Extract features for the input graph
    X_input = _extract_features(A, n)
    
    # 2. Generate training data (small ER graphs)
    # We want to learn weights that generalize to ER graphs.
    # The input graph is likely an ER graph.
    # Estimate p from input graph
    edges = A.sum() / 2.0
    possible_edges = n * (n - 1) / 2.0
    p_est = edges / possible_edges if possible_edges > 0 else 0.5
    
    # Generate 10 training graphs of size 50-100
    train_graphs = []
    train_features = []
    rng_gen = np.random.default_rng(42)
    
    for i in range(10):
        n_train = 50 + (i % 5) * 10 # 50 to 90
        p_train = p_est + rng_gen.normal(0, 0.05)
        p_train = np.clip(p_train, 0.05, 0.95)
        
        A_train = _generate_er_graph(n_train, p_train, seed=1000+i)
        X_train = _extract_features(A_train, n_train)
        
        train_graphs.append(A_train)
        train_features.append(X_train)
        
        if time.time() > deadline - 60:
            break
            
    # 3. Learn weights
    if len(train_graphs) > 0:
        w_learned = _learn_weights(train_features, train_graphs, time_limit=30.0)
    else:
        # Fallback to heuristic weights if no training time
        w_learned = np.array([1.0, 1.0, 0.5, 0.5]) # Prioritize low degree, low nbr deg
        
    # 4. Apply learned weights to input graph
    scores = X_input @ w_learned
    order = np.argsort(-scores, kind='stable')
    
    # 5. Local Search Refinement (if time permits)
    current_size = _greedy_size(A, order)
    best_size = current_size
    best_order = order.copy()
    
    rng_search = np.random.default_rng(42)
    
    # Simple hill climbing on the permutation
    # We only have ~100s left.
    # Each evaluation is O(N^2) ~ 40k ops. 
    # We can do roughly 1000-5000 evaluations.
    
    max_iters = 2000
    no_improve = 0
    max_no_improve = 100
    
    for i in range(max_iters):
        if time.time() > deadline - 5:
            break
        if no_improve > max_no_improve:
            break
            
        # Random swap
        i_idx, j_idx = rng_search.integers(0, n, size=2)
        if i_idx == j_idx:
            continue
            
        new_order = best_order.copy()
        new_order[i_idx], new_order[j_idx] = new_order[j_idx], new_order[i_idx]
        
        new_size = _greedy_size(A, new_order)
        
        if new_size > best_size:
            best_size = new_size
            best_order = new_order
            no_improve = 0
        else:
            no_improve += 1
            
    # Convert best_order to priority vector
    priority_vec = np.zeros(n)
    for idx, v in enumerate(best_order):
        priority_vec[v] = n - idx
        
    return priority_vec.astype(float)
# EVOLVE-BLOCK-END
