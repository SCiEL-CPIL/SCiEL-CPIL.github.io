"""Combined program: mean-field reaction-diffusion initialization + simulated annealing local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _simulate_fast(nbrs, scores):
    """Fast evaluation: greedy pick in descending score order, skip neighbors of picked."""
    n = len(nbrs)
    if n == 0:
        return 0
    indices = np.argsort(-scores, kind='stable')
    blocked = np.zeros(n, dtype=bool)
    count = 0
    for v in indices:
        if not blocked[v]:
            count += 1
            blocked[v] = True
            if len(nbrs[v]) > 0:
                blocked[nbrs[v]] = True
    return count

def _mean_field_init(n, nbrs, degree, iters=300):
    """
    Mean-field reaction-diffusion on the graph.
    Source term favors low-degree nodes; diffusion smooths across edges.
    Returns scores (higher = better).
    """
    # Source: low degree is good -> high source for low-degree nodes
    source = -degree  # negative degree => low degree gives less negative (higher)
    
    # Diffusion: Laplacian-based smoothing. Use graph Laplacian L = D - A.
    # Reaction-diffusion: s_{t+1} = s_t + dt * (L @ s_t + source)
    # But we want to converge to a steady state that favors structural independence.
    # Use a simple iterative scheme:
    #   s <- alpha * (L @ s / max_deg) + (1 - alpha) * source
    # This is a damped diffusion toward the source, with L acting as a mixing operator.
    
    # Build Laplacian as matrix (n=200, fine)
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in nbrs[i]:
            A[i, j] = 1.0
    D_mat = np.diag(degree)
    L = D_mat - A
    
    max_deg = np.max(degree) if n > 0 else 1.0
    if max_deg < 1e-10:
        max_deg = 1.0
    
    # Normalize Laplacian for stable diffusion
    L_norm = L / max_deg
    
    # Initialize scores from source
    s = source.astype(float).copy()
    
    # Iterate reaction-diffusion
    alpha = 0.5  # diffusion strength
    for _ in range(iters):
        s_new = alpha * (L_norm @ s) + (1.0 - alpha) * source
        s = s_new
    
    # Add a small Fiedler-like component for global structure
    # Use power iteration on shifted Laplacian to get second eigenvector
    try:
        c = 2.0 * max_deg + 1.0
        M = c * np.eye(n) - L
        ones = np.ones(n) / np.sqrt(n)
        v = np.random.randn(n)
        v = v - np.dot(v, ones) * ones
        nv = np.linalg.norm(v)
        if nv < 1e-10:
            v = np.random.randn(n)
            v = v - np.dot(v, ones) * ones
            nv = np.linalg.norm(v)
        v /= nv
        for _ in range(100):
            w = M @ v
            w = w - np.dot(w, ones) * ones
            nw = np.linalg.norm(w)
            if nw < 1e-10:
                break
            v = w / nw
        # Use absolute Fiedler as a tie-breaker / structural feature
        s = s + 0.3 * np.abs(v)
    except Exception:
        pass
    
    return s

def priority(adjacency):
    """
    Compute vertex scores for the Maximum Independent Set greedy decoder.
    Uses mean-field reaction-diffusion initialization + simulated annealing local search.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit
    
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Precompute neighbor lists
    nbrs = [np.where(adjacency[i])[0] for i in range(n)]
    degree = np.array([len(nbrs[i]) for i in range(n)], dtype=float)
    
    # --- 1. Mean-field reaction-diffusion initialization ---
    scores = _mean_field_init(n, nbrs, degree, iters=300)
    
    # Also compute a simple heuristic for comparison
    deg_sum_nbrs = np.zeros(n)
    for i in range(n):
        if degree[i] > 0:
            deg_sum_nbrs[i] = np.sum(degree[nbrs[i]])
    avg_nbr_deg = np.where(degree > 0, deg_sum_nbrs / np.maximum(degree, 1e-10), 0.0)
    heuristic_scores = -(degree + 0.5 * avg_nbr_deg)
    
    # Pick the better initial score set
    count_mf = _simulate_fast(nbrs, scores)
    count_heur = _simulate_fast(nbrs, heuristic_scores)
    if count_heur > count_mf:
        scores = heuristic_scores
        best_count = count_heur
    else:
        best_count = count_mf
    
    best_scores = scores.copy()
    
    # --- 2. Simulated Annealing Local Search ---
    rng = np.random.default_rng(42)
    T_start = 2.0
    T_end = 0.01
    max_search_time = DEADLINE - 5.0  # Leave 5s for final polish
    
    current_scores = scores.copy()
    current_size = best_count
    
    while time.time() - start_time < max_search_time:
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / max_search_time)
        T = T_start * (T_end / T_start) ** progress
        
        # Pick two random distinct vertices
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        
        # Swap scores
        current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
        
        new_size = _simulate_fast(nbrs, current_scores)
        delta = new_size - current_size
        
        if delta > 0:
            current_size = new_size
            if new_size > best_count:
                best_count = new_size
                best_scores = current_scores.copy()
        else:
            if T > 0:
                prob = np.exp(delta / T)
            else:
                prob = 0.0
            if rng.random() < prob:
                current_size = new_size
            else:
                current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
    
    # --- 3. Final Polish: targeted local improvements ---
    if time.time() - start_time < DEADLINE - 1.0:
        scores = best_scores.copy()
        improved_any = True
        max_sweeps = 50
        sweep = 0
        while improved_any and sweep < max_sweeps and time.time() - start_time < DEADLINE - 1.0:
            improved_any = False
            sweep += 1
            n_try = min(n, 80)
            try_indices = np.random.choice(n, size=n_try, replace=False)
            for i_idx in try_indices:
                if time.time() - start_time > DEADLINE - 1.0:
                    break
                i = int(i_idx)
                for _ in range(3):
                    j = int(np.random.randint(n))
                    if i == j:
                        continue
                    scores[i], scores[j] = scores[j], scores[i]
                    count = _simulate_fast(nbrs, scores)
                    if count > best_count:
                        best_count = count
                        best_scores = scores.copy()
                        improved_any = True
                    else:
                        scores[i], scores[j] = scores[j], scores[i]
        
        # Fine-tune individual scores
        if time.time() - start_time < DEADLINE - 0.5:
            scores = best_scores.copy()
            for sweep in range(15):
                if time.time() - start_time > DEADLINE - 0.5:
                    break
                improved = False
                n_try = min(n, 100)
                try_indices = np.random.choice(n, size=n_try, replace=False)
                for i_idx in try_indices:
                    i = int(i_idx)
                    for delta_val in [0.005, -0.005, 0.01, -0.01]:
                        old_val = scores[i]
                        scores[i] = old_val + delta_val
                        count = _simulate_fast(nbrs, scores)
                        if count > best_count:
                            best_count = count
                            best_scores = scores.copy()
                            improved = True
                        else:
                            scores[i] = old_val
                if not improved:
                    break
    
    return best_scores
# EVOLVE-BLOCK-END
