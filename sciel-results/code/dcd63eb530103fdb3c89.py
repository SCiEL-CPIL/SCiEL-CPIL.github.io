import numpy as np
import time
import random

def _compute_fiedler(adj, n, iters=100):
    """
    Compute the Fiedler vector (eigenvector corresponding to the second smallest 
    eigenvalue of the Laplacian) using power iteration on the shifted Laplacian.
    """
    if n <= 1:
        return np.zeros(n)
    
    # Laplacian L = D - A
    degrees = adj.sum(axis=1)
    D = np.diag(degrees)
    L = D - adj
    
    # Estimate lambda_max for shifting
    # lambda_max <= 2 * max_degree
    max_deg = np.max(degrees)
    if max_deg == 0:
        return np.zeros(n)
    
    c = 2.0 * max_deg + 1.0
    M = c * np.eye(n) - L
    
    # We want the eigenvector of M corresponding to the second largest eigenvalue.
    # The largest eigenvalue of M corresponds to lambda_0 = 0 of L (all-ones vector).
    # So we need to deflation or orthogonalize against the all-ones vector.
    
    ones = np.ones(n) / np.sqrt(n)
    
    # Initialize random vector orthogonal to ones
    v = np.random.randn(n)
    v = v - np.dot(v, ones) * ones
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-10:
        v = np.random.randn(n)
        v = v - np.dot(v, ones) * ones
        norm_v = np.linalg.norm(v)
    v /= norm_v
    
    for _ in range(iters):
        w = M @ v
        # Project out the all-ones component to stay in the orthogonal subspace
        w = w - np.dot(w, ones) * ones
        norm_w = np.linalg.norm(w)
        if norm_w < 1e-10:
            break
        v = w / norm_w
        
    return v

def _compute_ldf_features(adj, n):
    """
    Simulate LDF peeling to get removal degrees and times.
    Returns:
    ldf_deg: degree at removal
    ldf_time: time step of removal (normalized to 0-1, inverted so early=high? 
              Let's return raw time and let the search decide. 
              Actually, current program used removal_degree directly.
              Let's return removal_degree and (max_time - time) as 'survival'.
    """
    active = np.ones(n, dtype=bool)
    deg = adj.sum(axis=1).astype(float)
    
    removal_degree = np.zeros(n)
    removal_time = np.zeros(n)
    
    t = 0
    max_t = 0
    while np.any(active):
        current_deg = np.where(active, deg, np.inf)
        min_idx = np.argmin(current_deg)
        
        if not active[min_idx]:
            break
            
        removal_degree[min_idx] = deg[min_idx]
        removal_time[min_idx] = t
        max_t = t
        
        neighbors = np.where(adj[min_idx])[0]
        
        removed_mask = np.zeros(n, dtype=bool)
        removed_mask[min_idx] = True
        removed_mask[neighbors] = True
        
        counts = adj @ removed_mask
        deg -= counts
        deg[deg < 0] = 0
        
        active[min_idx] = False
        active[neighbors] = False
        
        t += 1
        
    # Normalize time to 0..1
    if max_t > 0:
        removal_time_norm = removal_time / max_t
    else:
        removal_time_norm = removal_time
        
    return removal_degree, removal_time_norm

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Precompute features
    # 1. Degree
    degrees = adj.sum(axis=1).astype(float)
    
    # 2. Mean Neighbor Degree
    neighbor_deg_sum = adj @ degrees
    safe_deg = degrees + 1e-6
    mean_neighbor_deg = neighbor_deg_sum / safe_deg
    
    # 3. Triangles
    # Triangles involving i = 0.5 * (A^2)[i,i]
    A_sq = adj @ adj
    triangles = 0.5 * np.diag(A_sq)
    
    # 4. Clustering Coefficient
    # C_i = 2 * T_i / (d_i * (d_i - 1))
    clus = np.zeros(n)
    mask = degrees > 1
    clus[mask] = 2.0 * triangles[mask] / (degrees[mask] * (degrees[mask] - 1.0))
    
    # 5. Core Number (approximate k-core)
    # Simple greedy peeling for core number
    core = np.zeros(n, dtype=int)
    deg_copy = degrees.astype(int).copy()
    remaining = np.ones(n, dtype=bool)
    # We can use a bucket sort approach or simple iteration for N=200
    # For N=200, simple O(N^2) is fine
    for _ in range(n):
        if not np.any(remaining):
            break
        # Find min degree among remaining
        min_deg = np.inf
        min_idx = -1
        for i in range(n):
            if remaining[i]:
                if deg_copy[i] < min_deg:
                    min_deg = deg_copy[i]
                    min_idx = i
        if min_idx == -1:
            break
        # Assign core number
        # Actually, standard k-core peeling assigns k to all nodes removed at step k
        # This simple "pick min" is a valid way to compute the core number decomposition
        # But it's O(N^2).
        # Let's just use the LDF removal order as a proxy for "peripheralness"
        # Or skip core number to save time and rely on LDF features.
        pass 
        
    # Let's use a simpler "core" proxy: degree in the 2-core?
    # Actually, let's just use the LDF features which are more dynamic.
    
    # 6. Fiedler Vector
    try:
        fiedler = _compute_fiedler(adj, n, iters=50)
        abs_fiedler = np.abs(fiedler)
    except:
        abs_fiedler = np.zeros(n)
        
    # 7. LDF Features
    ldf_deg, ldf_time = _compute_ldf_features(adj, n)
    
    # Feature Matrix (n_samples, n_features)
    # Features:
    # 0: degree
    # 1: mean_neighbor_deg
    # 2: triangles
    # 3: clustering
    # 4: abs_fiedler
    # 5: ldf_deg
    # 6: ldf_time (normalized 0-1, 0=early, 1=late)
    
    features = np.column_stack([
        degrees, 
        mean_neighbor_deg, 
        triangles, 
        clus, 
        abs_fiedler, 
        ldf_deg, 
        ldf_time
    ])
    
    # Normalize features to 0-1 range to make weight search more stable
    # Avoid division by zero
    f_min = features.min(axis=0)
    f_max = features.max(axis=0)
    f_range = f_max - f_min
    f_range[f_range == 0] = 1.0
    features_norm = (features - f_min) / f_range
    
    # Define candidate weight vectors
    # We want to find weights w such that scores = w @ features_norm.T
    # and greedy(scores) maximizes independent set size.
    
    # Inspiration 2 used specific hand-picked vectors.
    # Current program logic: score ~ ldf_deg + 0.5 * (hub term).
    # Hub term is roughly mean_neighbor_deg.
    # So a good starting point is: [0, 0.5, 0, 0, 0, 1.0, 0] (scaled)
    
    candidates = []
    
    # 1. Baseline: LDF Degree only (Current program's core)
    candidates.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    
    # 2. LDF Degree + Mean Neighbor Deg (Hub blocking)
    candidates.append(np.array([0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0]))
    candidates.append(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    
    # 3. LDF Degree + Fiedler (Global structure)
    candidates.append(np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0]))
    candidates.append(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]))
    
    # 4. LDF Degree + Time (Survival)
    candidates.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5]))
    candidates.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]))
    
    # 5. Degree based (Static LDF)
    candidates.append(np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])) # Pick low degree? No, greedy picks high score. 
    # If we want to pick low degree, score should be -degree.
    candidates.append(np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0])) # Pick low mean neighbor deg?
    
    # 6. Random exploration around best candidates
    # We will do a small grid search or random restarts
    
    def _greedy_size(scores, adj, n):
        """Fast greedy evaluation."""
        order = np.argsort(-scores, kind='stable')
        blocked = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not blocked[v]:
                count += 1
                blocked[v] = True
                # Block neighbors
                if np.any(adj[v]):
                    blocked[adj[v]] = True
        return count
    
    best_score_val = -1
    best_weights = candidates[0]
    
    # Evaluate initial candidates
    for w in candidates:
        scores = w @ features_norm.T
        size = _greedy_size(scores, adj, n)
        if size > best_score_val:
            best_score_val = size
            best_weights = w
        if time.time() > deadline:
            break
            
    # If we have time, do a local search / random perturbation around best_weights
    if time.time() < deadline:
        # Random perturbations
        num_random = 50
        for i in range(num_random):
            if time.time() > deadline:
                break
            # Perturb best_weights
            noise = np.random.randn(len(best_weights)) * 0.2
            w_new = best_weights + noise
            scores = w_new @ features_norm.T
            size = _greedy_size(scores, adj, n)
            if size > best_score_val:
                best_score_val = size
                best_weights = w_new
                
    # Final scores
    final_scores = best_weights @ features_norm.T
    
    return final_scores
