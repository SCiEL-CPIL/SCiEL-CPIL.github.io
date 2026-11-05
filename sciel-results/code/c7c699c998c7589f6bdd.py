"""Combined approach: strong static heuristic initialization + extended local search with simulated annealing."""
import numpy as np
import time

def priority(adjacency):
    """
    Compute vertex scores for the Maximum Independent Set greedy decoder.
    
    Strategy:
    1. Initialize with a strong static heuristic (low degree + hub penalty + Fiedler).
    2. Perform simulated annealing local search to directly maximize IS size.
    3. Final polish with 2-opt swaps.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Precompute neighbor lists
    nbrs = [np.where(adjacency[i])[0] for i in range(n)]
    degree = np.array([len(nbrs[i]) for i in range(n)], dtype=float)
    
    def norm(x):
        x = np.asarray(x, dtype=float)
        min_val = x.min()
        max_val = x.max()
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)
    
    # Compute features
    D_n = norm(degree)
    
    mean_nd = np.zeros(n)
    for i in range(n):
        if degree[i] > 0:
            mean_nd[i] = np.mean(degree[nbrs[i]])
        else:
            mean_nd[i] = 0
    MeanND_n = norm(mean_nd)
    
    # Fast evaluation function
    def _evaluate_fast(scores):
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
    
    # Initial heuristic: combine low degree, hub penalty, and Fiedler
    # Start with -degree (low degree first)
    scores = -D_n.copy()
    
    # Add hub penalty: penalize vertices connected to high-degree neighbors
    scores -= 0.3 * MeanND_n
    
    # Try to add Fiedler vector if time permits
    try:
        A = adjacency.astype(float)
        D_mat = np.diag(degree)
        L = D_mat - A
        
        # Power iteration for Fiedler vector
        max_deg = int(np.max(degree)) if n > 0 else 0
        c = 2.0 * max_deg + 1.0
        M = c * np.eye(n) - L
        
        ones = np.ones(n) / np.sqrt(n)
        rng = np.random.default_rng(42)
        v = rng.standard_normal(n)
        v = v - np.dot(v, ones) * ones
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-10:
            v = rng.standard_normal(n)
            v = v - np.dot(v, ones) * ones
            norm_v = np.linalg.norm(v)
        v /= norm_v
        
        for _ in range(100):
            w = M @ v
            w = w - np.dot(w, ones) * ones
            norm_w = np.linalg.norm(w)
            if norm_w < 1e-10:
                break
            v = w / norm_w
        
        abs_fiedler = np.abs(v)
        AbsFiedler_n = norm(abs_fiedler)
        scores += 0.2 * AbsFiedler_n
    except Exception:
        pass
    
    # Evaluate initial score
    best_count = _evaluate_fast(scores)
    best_scores = scores.copy()
    current_scores = scores.copy()
    current_count = best_count
    
    # Simulated Annealing
    rng = np.random.default_rng(42)
    T_start = 1.0
    T_end = 0.001
    max_sa_time = 80.0  # 80 seconds for SA
    
    steps = 0
    while time.time() < start_time + max_sa_time and time.time() < deadline - 30:
        # Calculate temperature
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / max_sa_time)
        T = T_start * (T_end / T_start) ** progress
        
        # Pick two random distinct vertices
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        
        # Swap scores
        current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
        
        # Evaluate
        new_count = _evaluate_fast(current_scores)
        delta = new_count - current_count
        
        if delta > 0:
            # Accept better move
            current_count = new_count
            if new_count > best_count:
                best_count = new_count
                best_scores = current_scores.copy()
        else:
            # Accept worse move with probability exp(delta / T)
            if T > 0:
                prob = np.exp(delta / T)
            else:
                prob = 0.0
            
            if rng.random() < prob:
                current_count = new_count
            else:
                # Revert
                current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
        
        steps += 1
    
    # Final polish: 2-opt swaps
    if time.time() < deadline - 5:
        order = np.argsort(-best_scores, kind='stable')
        order_list = order.tolist()
        n_order = len(order_list)
        
        improved_total = True
        max_sweeps = 50
        sweep = 0
        while improved_total and sweep < max_sweeps and time.time() < deadline - 5:
            improved_total = False
            sweep += 1
            for i in range(n_order - 1):
                if time.time() > deadline - 5:
                    break
                vi = order_list[i]
                for j in range(i + 1, n_order):
                    if time.time() > deadline - 5:
                        break
                    vj = order_list[j]
                    # Swap vi and vj
                    order_list[i], order_list[j] = vj, vi
                    # Reconstruct scores from order
                    new_scores = np.empty(n, dtype=float)
                    for pos, v in enumerate(order_list):
                        new_scores[v] = float(n_order - pos)
                    count = _evaluate_fast(new_scores)
                    if count > best_count:
                        best_count = count
                        best_scores = new_scores.copy()
                        improved_total = True
                    else:
                        # Swap back
                        order_list[i], order_list[j] = vi, vj
    
    return best_scores
