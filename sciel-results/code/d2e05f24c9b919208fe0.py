"""Optimized packing of 50 equal circles in a unit circle."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 50

# EVOLVE-BLOCK-START

def _score_from_points(points):
    """Compute the packing score (largest common radius) for given centers."""
    # Distance from origin to each center
    r_orig = np.linalg.norm(points, axis=1)
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    min_edge_dist = np.min(1.0 - r_orig)
    return min(min_pair_dist / 2.0, min_edge_dist)


def _smooth_score(points, beta=100.0):
    """
    Smooth surrogate for the packing score using log-sum-exp soft-min.
    We want to maximize min(min_pair_dist/2, min_edge_dist).
    Equivalently, minimize -min(...).
    Using soft-min: soft_min(x) = -logsumexp(-beta*x)/beta approx min(x) for large beta.
    We want to maximize soft_min of the constraints.
    So we minimize -soft_min = logsumexp(-beta*x)/beta.
    """
    r_orig = np.linalg.norm(points, axis=1)
    edge_terms = 1.0 - r_orig  # shape (N,)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_terms = dists / 2.0  # shape (N, N)
    
    # Combine all constraint values
    all_terms = np.concatenate([edge_terms, pair_terms.ravel()])
    
    # Soft min: we want to maximize the minimum, so minimize the negative soft min
    # soft_min(all_terms) ≈ -logsumexp(-beta * all_terms) / beta
    # We minimize -soft_min = logsumexp(-beta * all_terms) / beta
    neg_beta_terms = -beta * all_terms
    max_val = np.max(neg_beta_terms)
    lse = max_val + np.log(np.sum(np.exp(neg_beta_terms - max_val)))
    return lse / beta


def _smooth_score_grad(points, beta=100.0):
    """Compute gradient of the smooth score objective (to be minimized)."""
    r_orig = np.linalg.norm(points, axis=1)
    edge_terms = 1.0 - r_orig
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_terms = dists / 2.0
    
    all_terms = np.concatenate([edge_terms, pair_terms.ravel()])
    
    # Compute weights for log-sum-exp
    neg_beta_terms = -beta * all_terms
    max_val = np.max(neg_beta_terms)
    exp_terms = np.exp(neg_beta_terms - max_val)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp  # shape (N + N*(N-1),)
    
    # Gradient of lse/beta w.r.t. all_terms is -beta * weights
    # Then chain rule to get gradient w.r.t. points
    grad_points = np.zeros_like(points)
    
    # Edge terms: edge_terms[i] = 1 - |c_i|
    # d edge_terms[i] / d c_i = -c_i / |c_i|
    edge_weights = weights[:N]
    for i in range(N):
        if r_orig[i] > 1e-12:
            grad_points[i] += -edge_weights[i] * (-points[i] / r_orig[i])
        # Actually: d(-soft_min)/d c_i where soft_min uses edge_terms
        # Objective = lse/beta, d obj/d all_terms[k] = -beta * weights[k]
        # d obj/d c_i = sum_k (d obj/d all_terms[k]) * (d all_terms[k]/d c_i)
        # For edge: all_terms[i] = 1 - |c_i|, d all_terms[i]/d c_i = -c_i/|c_i|
        # So contribution = -beta * weights[i] * (-c_i/|c_i|) = beta * weights[i] * c_i/|c_i|
        # Wait, I need to be more careful.
    
    # Let me redo this properly.
    # Objective f = (1/beta) * log(sum_j exp(-beta * t_j)) where t_j are the constraint values
    # df/dt_j = (1/beta) * (-beta * exp(-beta*t_j)) / sum_k exp(-beta*t_k) = -exp(-beta*t_j)/sum_k exp(-beta*t_k) = -weights[j]
    # So df/dt_j = -weights[j]
    
    # For edge constraints: t_i = 1 - |c_i|, dt_i/d c_i = -c_i/|c_i|
    # df/d c_i (from edge) = sum_i df/dt_i * dt_i/d c_i = df/dt_i * (-c_i/|c_i|) = -weights[i] * (-c_i/|c_i|) = weights[i] * c_i/|c_i|
    
    grad_points = np.zeros_like(points)
    for i in range(N):
        if r_orig[i] > 1e-12:
            grad_points[i] += edge_weights[i] * points[i] / r_orig[i]
    
    # For pair constraints: t_ij = |c_i - c_j|/2
    # dt_ij/d c_i = (c_i - c_j)/|c_i - c_j|, dt_ij/d c_j = (c_j - c_i)/|c_i - c_j|
    pair_weights = weights[N:].reshape(N, N)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if dists[i, j] > 1e-12:
                dvec = (points[i] - points[j]) / dists[i, j]
                # t_ij = |c_i - c_j|/2, dt_ij/d c_i = (c_i - c_j)/(2*|c_i - c_j|)
                grad_points[i] += pair_weights[i, j] * dvec / 2.0
                grad_points[j] += pair_weights[j, i] * (-dvec) / 2.0
    
    return grad_points


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            points = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.98
    
    best_points = points.copy()
    best_score = _score_from_points(best_points)
    
    # Phase 1: Differential Evolution on ring-parametric model
    # Represent points by (radius, angle) for each point
    # Parametrize: x_i = r_i * cos(theta_i), y_i = r_i * sin(theta_i)
    
    def ring_params_to_points(params):
        r = params[:N]
        theta = params[N:2*N]
        return np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    def objective_ring(params):
        pts = ring_params_to_points(params)
        return _smooth_score(pts, beta=50.0)
    
    # Bounds: radius in [0, 1], angle in [0, 2*pi]
    bounds_ring = [(0.0, 1.0)] * N + [(0.0, 2 * np.pi)] * N
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 30.0:
        try:
            de_result = differential_evolution(
                objective_ring, bounds_ring,
                seed=42, maxiter=100, popsize=15,
                tol=1e-6, mutation=(0.5, 1.5), recombination=0.8,
                polish=False, init='latinhypercube'
            )
            if de_result.success or de_result.fun < 1e10:
                ring_params = de_result.x
                pts = ring_params_to_points(ring_params)
                # Ensure points are inside the unit circle
                r_norm = np.linalg.norm(pts, axis=1)
                valid = r_norm <= 1.0
                if np.sum(valid) >= N:
                    pts = pts[valid][:N]
                    score = _score_from_points(pts)
                    if score > best_score:
                        best_points = pts
                        best_score = score
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 20.0:
        # Phase 2: L-BFGS polish on the full 2D coordinates
        def objective_2d(points_flat):
            pts = points_flat.reshape(N, 2)
            return _smooth_score(pts, beta=100.0)
        
        def grad_2d(points_flat):
            pts = points_flat.reshape(N, 2)
            return _smooth_score_grad(pts, beta=100.0).ravel()
        
        try:
            res = minimize(
                objective_2d, best_points.ravel(),
                jac=grad_2d, method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < 1e10:
                pts = res.x.reshape(N, 2)
                # Project back inside if needed
                r_norm = np.linalg.norm(pts, axis=1)
                mask = r_norm <= 1.0
                if np.sum(mask) >= N:
                    pts = pts[mask][:N]
                    score = _score_from_points(pts)
                    if score > best_score:
                        best_points = pts
                        best_score = score
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 10.0:
        # Phase 3: Additional L-BFGS polish with higher beta for sharper min
        def objective_2d_high_beta(points_flat):
            pts = points_flat.reshape(N, 2)
            return _smooth_score(pts, beta=200.0)
        
        def grad_2d_high_beta(points_flat):
            pts = points_flat.reshape(N, 2)
            return _smooth_score_grad(pts, beta=200.0).ravel()
        
        try:
            res = minimize(
                objective_2d_high_beta, best_points.ravel(),
                jac=grad_2d_high_beta, method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.fun < 1e10:
                pts = res.x.reshape(N, 2)
                r_norm = np.linalg.norm(pts, axis=1)
                mask = r_norm <= 1.0
                if np.sum(mask) >= N:
                    pts = pts[mask][:N]
                    score = _score_from_points(pts)
                    if score > best_score:
                        best_points = pts
                        best_score = score
        except Exception:
            pass
    
    # Final validity check
    r_norm = np.linalg.norm(best_points, axis=1)
    if np.any(r_norm > 1.0) or np.any(~np.isfinite(best_points)):
        # Fallback to lattice
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            points = points[mask]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98
    
    return best_points

# EVOLVE-BLOCK-END
