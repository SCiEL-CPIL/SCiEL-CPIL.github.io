"""Guided local search penalty for TSP using model-and-optimize approach."""
import numpy as np
import time

_DEADLINE = 330.0
_start_time = None


def _two_opt(D, tour, max_passes=10, tol=-1e-10):
    """2-opt on distance matrix D starting from tour. Returns improved tour."""
    n = len(tour)
    if n < 4:
        return tour
    tour = list(tour)
    D_local = D  # use directly for speed
    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 1):
            a, b = tour[i - 1], tour[i]
            for j in range(i + 1, n):
                c, d = tour[j], tour[(j + 1) % n]
                delta = D_local[a, c] + D_local[b, d] - D_local[a, b] - D_local[c, d]
                if delta < tol:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return tour


def _two_opt_precise(D, tour, max_passes=20, tol=-1e-12):
    """High-precision 2-opt polish with tighter tolerance and more passes."""
    n = len(tour)
    if n < 4:
        return tour
    tour = list(tour)
    D_local = D
    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 1):
            a, b = tour[i - 1], tour[i]
            for j in range(i + 1, n):
                c, d = tour[j], tour[(j + 1) % n]
                delta = D_local[a, c] + D_local[b, d] - D_local[a, b] - D_local[c, d]
                if delta < tol:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return tour


def _closed_len(D, tour):
    """Compute the length of the closed tour using distance matrix D."""
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += D[tour[i], tour[(i + 1) % n]]
    return total


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = edge_distance.shape[0]
    tour = list(local_opt_tour)
    
    # Check if we're close to the deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5:
        # Return a simple penalty to be safe
        max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
        if max_used < 1e-10:
            max_used = 1.0
        return edge_distance + 0.1 * edge_n_used / max_used
    
    # Get the edges in the current local optimum tour
    tour_edges = set()
    for i in range(n):
        u, v = tour[i], tour[(i + 1) % n]
        tour_edges.add((min(u, v), max(u, v)))
    
    # Get usage counts for tour edges
    tour_usage = []
    for (u, v) in tour_edges:
        tour_usage.append(edge_n_used[u, v])
    
    if tour_usage:
        max_tour_usage = max(tour_usage)
        min_tour_usage = min(tour_usage)
    else:
        max_tour_usage = 1.0
        min_tour_usage = 0.0
    
    # The penalty should scale with how many times the current tour has been found
    # Use max_tour_usage as an indicator of how "stuck" we are
    if max_tour_usage < 1e-10:
        max_tour_usage = 1.0
    
    # Model-and-optimize approach:
    # We model the penalty as: D_pen = D_true + alpha * f(edge_n_used, tour)
    # where f is a function that we optimize against the true objective.
    #
    # Key insight: The penalty should be applied to edges in the current tour
    # to encourage exploration. The optimal alpha depends on the problem instance.
    # We can estimate a good alpha by trying a few values and picking the one
    # that gives the best 2-opt result on the penalized matrix.
    
    best_D_pen = None
    best_true_len = float('inf')
    best_tour = tour
    
    # Try a range of alpha values (penalty scaling factors)
    # The penalty is: for each edge in the current tour, add alpha * usage_count
    # We also consider a global penalty based on overall usage
    
    alpha_candidates = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
    
    # Also try a different penalty model: exponential decay based on usage
    # D_pen[u,v] = D_true[u,v] + alpha * exp(beta * usage[u,v])
    beta_candidates = [0.5, 1.0, 1.5, 2.0]
    
    # Collect all candidate results for soft-min selection
    candidate_results = []  # list of (true_len, D_pen, tour)
    
    for alpha in alpha_candidates:
        # Check deadline
        if time.time() - _start_time > _DEADLINE - 15:
            break
        
        D_pen = edge_distance.copy()
        
        # Apply penalty to edges in the current tour
        for (u, v) in tour_edges:
            usage = edge_n_used[u, v]
            if usage > 0:
                # Linear penalty scaled by usage
                penalty = alpha * usage
                D_pen[u, v] += penalty
                D_pen[v, u] += penalty
        
        # Ensure symmetry
        D_pen = (D_pen + D_pen.T) / 2.0
        np.fill_diagonal(D_pen, 0.0)
        
        # Run 2-opt on the penalized matrix
        tour_pen = _two_opt(D_pen, tour, max_passes=10)
        
        # Final polish: run 2-opt on true distances
        if time.time() - _start_time < _DEADLINE - 10:
            tour_polished = _two_opt_precise(edge_distance, tour_pen, max_passes=15, tol=-1e-12)
            true_len = _closed_len(edge_distance, tour_polished)
            candidate_results.append((true_len, D_pen.copy(), tour_polished))
            if true_len < best_true_len:
                best_true_len = true_len
                best_tour = tour_polished
                best_D_pen = D_pen.copy()
        else:
            true_len = _closed_len(edge_distance, tour_pen)
            candidate_results.append((true_len, D_pen.copy(), tour_pen))
            if true_len < best_true_len:
                best_true_len = true_len
                best_tour = tour_pen
                best_D_pen = D_pen.copy()
    
    # Also try exponential penalty model
    for alpha in [0.1, 0.2, 0.3]:
        for beta in beta_candidates:
            # Check deadline
            if time.time() - _start_time > _DEADLINE - 15:
                break
            
            D_pen = edge_distance.copy()
            
            # Apply exponential penalty to edges in the current tour
            for (u, v) in tour_edges:
                usage = edge_n_used[u, v]
                if usage > 0:
                    # Exponential penalty
                    penalty = alpha * np.exp(beta * usage)
                    D_pen[u, v] += penalty
                    D_pen[v, u] += penalty
            
            # Ensure symmetry
            D_pen = (D_pen + D_pen.T) / 2.0
            np.fill_diagonal(D_pen, 0.0)
            
            # Run 2-opt on the penalized matrix
            tour_pen = _two_opt(D_pen, tour, max_passes=10)
            
            # Final polish: run 2-opt on true distances
            if time.time() - _start_time < _DEADLINE - 10:
                tour_polished = _two_opt_precise(edge_distance, tour_pen, max_passes=15, tol=-1e-12)
                true_len = _closed_len(edge_distance, tour_polished)
                candidate_results.append((true_len, D_pen.copy(), tour_polished))
                if true_len < best_true_len:
                    best_true_len = true_len
                    best_tour = tour_polished
                    best_D_pen = D_pen.copy()
            else:
                true_len = _closed_len(edge_distance, tour_pen)
                candidate_results.append((true_len, D_pen.copy(), tour_pen))
                if true_len < best_true_len:
                    best_true_len = true_len
                    best_tour = tour_pen
                    best_D_pen = D_pen.copy()
    
    # Try directed edge-reversal heuristic: penalize edges based on 2-opt improvement potential
    if time.time() - _start_time < _DEADLINE - 20:
        # For each edge in the tour, compute the best 2-opt improvement if we remove it
        # and penalize that edge proportionally to the improvement
        n_tour = len(tour)
        directed_penalties = np.zeros((n, n))
        
        for i in range(n_tour):
            a, b = tour[i], tour[(i + 1) % n_tour]
            # Find the best 2-opt move that removes edge (a,b)
            # This is equivalent to: for each pair (c,d) not involving a,b,
            # check if swapping edges (a,b) and (c,d) improves the tour
            best_improvement = 0.0
            for j in range(i + 2, n_tour):
                if j == (i + n_tour - 1) % n_tour:
                    continue  # adjacent edge, skip
                c, d = tour[j], tour[(j + 1) % n_tour]
                # Check if (a,c) and (b,d) are valid (not same as original)
                if c == a or c == b or d == a or d == b:
                    continue
                improvement = edge_distance[a, b] + edge_distance[c, d] - edge_distance[a, c] - edge_distance[b, d]
                if improvement > best_improvement:
                    best_improvement = improvement
            
            # Penalize edge (a,b) proportionally to the improvement available
            # Scale by a factor to keep penalties reasonable
            scale = 0.3
            directed_penalties[a, b] += scale * best_improvement
            directed_penalties[b, a] += scale * best_improvement
        
        # Also add usage-based penalty
        usage_penalty = 0.1 * edge_n_used
        total_penalty = directed_penalties + usage_penalty
        
        D_pen_directed = edge_distance + total_penalty
        D_pen_directed = (D_pen_directed + D_pen_directed.T) / 2.0
        np.fill_diagonal(D_pen_directed, 0.0)
        
        tour_directed = _two_opt(D_pen_directed, tour, max_passes=10)
        if time.time() - _start_time < _DEADLINE - 10:
            tour_directed_polished = _two_opt_precise(edge_distance, tour_directed, max_passes=15, tol=-1e-12)
            true_len_directed = _closed_len(edge_distance, tour_directed_polished)
            candidate_results.append((true_len_directed, D_pen_directed.copy(), tour_directed_polished))
            if true_len_directed < best_true_len:
                best_true_len = true_len_directed
                best_tour = tour_directed_polished
                best_D_pen = D_pen_directed.copy()
        else:
            true_len_directed = _closed_len(edge_distance, tour_directed)
            candidate_results.append((true_len_directed, D_pen_directed.copy(), tour_directed))
            if true_len_directed < best_true_len:
                best_true_len = true_len_directed
                best_tour = tour_directed
                best_D_pen = D_pen_directed.copy()
    
    # Use soft-min (log-sum-exp) to blend top candidates for smoother selection
    if len(candidate_results) >= 2 and time.time() - _start_time < _DEADLINE - 5:
        # Sort by true length
        candidate_results.sort(key=lambda x: x[0])
        
        # Take top 3 candidates and blend using soft-min weighting
        top_k = min(3, len(candidate_results))
        top_candidates = candidate_results[:top_k]
        
        # Compute weights using log-sum-exp for numerical stability
        lens = np.array([c[0] for c in top_candidates])
        min_len = lens.min()
        # Use inverse length as weight (lower length = higher weight)
        inv_lens = 1.0 / (lens - min_len + 1e-10)
        # Normalize
        weights = inv_lens / inv_lens.sum()
        
        # Blend the penalty matrices
        blended_D = np.zeros_like(edge_distance)
        for w, (ln, D_c, t_c) in zip(weights, top_candidates):
            blended_D += w * D_c
        
        np.fill_diagonal(blended_D, 0.0)
        blended_D = (blended_D + blended_D.T) / 2.0
        
        # Verify the blended result is valid
        if np.all(np.isfinite(blended_D)):
            # Run 2-opt on blended matrix to get final tour
            tour_blended = _two_opt(blended_D, tour, max_passes=10)
            if time.time() - _start_time < _DEADLINE - 3:
                tour_blended_polished = _two_opt_precise(edge_distance, tour_blended, max_passes=15, tol=-1e-12)
                true_len_blended = _closed_len(edge_distance, tour_blended_polished)
                if true_len_blended < best_true_len:
                    best_true_len = true_len_blended
                    best_tour = tour_blended_polished
                    best_D_pen = blended_D.copy()
    
    if best_D_pen is not None:
        return best_D_pen
    else:
        # Fallback: return edge_distance with mild penalty
        max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
        if max_used < 1e-10:
            max_used = 1.0
        return edge_distance + 0.1 * edge_n_used / max_used
