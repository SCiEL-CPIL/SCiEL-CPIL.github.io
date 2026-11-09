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
    
    for alpha in alpha_candidates:
        # Check deadline
        if time.time() - _start_time > _DEADLINE - 10:
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
        
        # Final polish: run high-precision 2-opt on true distances
        if time.time() - _start_time < _DEADLINE - 5:
            tour_polished = _two_opt_precise(edge_distance, tour_pen)
            true_len = _closed_len(edge_distance, tour_polished)
            if true_len < best_true_len:
                best_true_len = true_len
                best_tour = tour_polished
                best_D_pen = D_pen
        else:
            true_len = _closed_len(edge_distance, tour_pen)
            if true_len < best_true_len:
                best_true_len = true_len
                best_tour = tour_pen
                best_D_pen = D_pen
    
    # Also try exponential penalty model
    for alpha in [0.1, 0.2, 0.3]:
        for beta in beta_candidates:
            # Check deadline
            if time.time() - _start_time > _DEADLINE - 10:
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
            
            # Final polish: run high-precision 2-opt on true distances
            if time.time() - _start_time < _DEADLINE - 5:
                tour_polished = _two_opt_precise(edge_distance, tour_pen)
                true_len = _closed_len(edge_distance, tour_polished)
                if true_len < best_true_len:
                    best_true_len = true_len
                    best_tour = tour_polished
                    best_D_pen = D_pen
            else:
                true_len = _closed_len(edge_distance, tour_pen)
                if true_len < best_true_len:
                    best_true_len = true_len
                    best_tour = tour_pen
                    best_D_pen = D_pen
    
    if best_D_pen is not None:
        # Final high-precision polish on the best tour found
        if time.time() - _start_time < _DEADLINE - 3:
            final_tour = _two_opt_precise(edge_distance, best_tour)
            # Return the penalized matrix that led to the best tour
            return best_D_pen
        return best_D_pen
    else:
        # Fallback: return edge_distance with mild penalty
        max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
        if max_used < 1e-10:
            max_used = 1.0
        return edge_distance + 0.1 * edge_n_used / max_used
