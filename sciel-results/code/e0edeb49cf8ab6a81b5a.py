"""Guided local search penalty for TSP; direct GLS approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 300.0
_start_time = None


def _two_opt(D, tour, max_passes=10, tol=-1e-10):
    """2-opt on distance matrix D starting from tour. Returns improved tour."""
    n = len(tour)
    if n < 4:
        return tour
    tour = list(tour)
    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 1):
            a, b = tour[i - 1], tour[i]
            for j in range(i + 1, n):
                c, d = tour[j], tour[(j + 1) % n]
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
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


def _spectral_laplacian_penalty(D_true, usage_matrix, scale=0.05):
    """
    Compute a spectral graph Laplacian regularization term.
    
    The Laplacian L = D - A where D is the degree matrix and A is the adjacency.
    For a distance matrix, we use the negative of the distance as weights to
    create a smoothness prior that penalizes high-frequency oscillations.
    
    We add scale * L @ usage_matrix to create a global topological bias.
    """
    n = D_true.shape[0]
    # Create weighted adjacency from negative distances (closer = stronger connection)
    # Use a soft max to avoid issues with very large distances
    max_dist = D_true.max()
    if max_dist < 1e-10:
        max_dist = 1.0
    weights = np.exp(-D_true / (max_dist * 0.5))  # Soft attention: closer points have higher weight
    np.fill_diagonal(weights, 0.0)
    
    # Compute degree matrix
    degrees = weights.sum(axis=1)
    L = np.diag(degrees) - weights
    
    # Apply the Laplacian to the usage matrix to get a spectral penalty
    # This creates a smooth, global bias that penalizes high-frequency patterns
    spectral_penalty = scale * (L @ usage_matrix)
    
    # Symmetrize
    spectral_penalty = (spectral_penalty + spectral_penalty.T) / 2.0
    np.fill_diagonal(spectral_penalty, 0.0)
    
    return spectral_penalty


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
    
    # Compute the spectral Laplacian penalty once (reused across strategies)
    max_used_global = edge_n_used.max()
    if max_used_global < 1e-10:
        max_used_global = 1.0
    usage_normalized = edge_n_used / max_used_global
    
    # Try multiple penalty strategies with smooth surrogate
    best_D_pen = None
    best_len = float('inf')
    best_tour = tour
    
    # Strategy parameters: (base_penalty, exponent, scope, spectral_scale)
    # scope: 0 = penalize only current tour edges, 1 = penalize all edges by usage
    # spectral_scale: amount of spectral Laplacian regularization to add
    strategies = [
        (0.1, 1.0, 0, 0.05),
        (0.2, 1.0, 0, 0.10),
        (0.3, 1.0, 0, 0.15),
        (0.1, 2.0, 0, 0.05),
        (0.2, 2.0, 0, 0.10),
        (0.05, 1.5, 0, 0.05),
        (0.15, 1.5, 0, 0.10),
        # Also try penalizing all edges based on usage with spectral component
        (0.05, 1.0, 1, 0.05),
        (0.1, 1.0, 1, 0.10),
        (0.2, 1.0, 1, 0.15),
        # Pure spectral strategies
        (0.0, 1.0, 0, 0.20),
        (0.0, 1.0, 0, 0.30),
    ]
    
    for base_pen, exponent, scope, spectral_scale in strategies:
        # Check deadline
        if time.time() - _start_time > _DEADLINE - 10:
            break
        
        D_pen = edge_distance.copy()
        
        if scope == 0:
            # Penalize only edges in the current tour with smooth scaling
            for (u, v) in tour_edges:
                usage = edge_n_used[u, v]
                if usage > 0:
                    # Smooth penalty: use log-sum-exp style soft-max for usage scaling
                    # This creates a smoother surrogate for the max/usage objective
                    normalized_usage = usage / max_tour_usage
                    # Add a small linear term to ensure non-zero penalty even for low usage
                    smooth_penalty = base_pen * (normalized_usage ** exponent + 0.02 * normalized_usage)
                    D_pen[u, v] += smooth_penalty
                    D_pen[v, u] += smooth_penalty
        else:
            # Penalize all edges based on their usage with smooth scaling
            usage_matrix = edge_n_used / max_used_global
            # Smooth penalty with linear term for better gradient behavior
            penalty_matrix = base_pen * (usage_matrix ** exponent + 0.02 * usage_matrix)
            D_pen = D_pen + penalty_matrix
        
        # Add spectral Laplacian regularization
        if spectral_scale > 0:
            spectral_penalty = _spectral_laplacian_penalty(edge_distance, usage_normalized, scale=spectral_scale)
            D_pen = D_pen + spectral_penalty
        
        # Ensure symmetry
        D_pen = (D_pen + D_pen.T) / 2.0
        np.fill_diagonal(D_pen, 0.0)
        
        # Run 2-opt on the penalized matrix
        tour_pen = _two_opt(D_pen, tour, max_passes=10)
        
        # Final polish: run 2-opt on true distances with tighter tolerance
        if time.time() - _start_time < _DEADLINE - 5:
            tour_polished = _two_opt(edge_distance, tour_pen, max_passes=10, tol=-1e-12)
            true_len = _closed_len(edge_distance, tour_polished)
            if true_len < best_len:
                best_len = true_len
                best_tour = tour_polished
                best_D_pen = D_pen
        else:
            true_len = _closed_len(edge_distance, tour_pen)
            if true_len < best_len:
                best_len = true_len
                best_tour = tour_pen
                best_D_pen = D_pen
    
    if best_D_pen is not None:
        return best_D_pen
    else:
        # Fallback: return edge_distance with mild penalty
        max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
        if max_used < 1e-10:
            max_used = 1.0
        return edge_distance + 0.1 * edge_n_used / max_used
# EVOLVE-BLOCK-END
