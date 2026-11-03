"""Improved packing of 50 equal circles in unit disk using smooth optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_score(points):
    """Compute the maximum radius for given points (exact)."""
    n = len(points)
    if n < 2:
        return 0.0
    # Min distance between any pair / 2
    min_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_pair:
                min_pair = d
    min_pair_r = min_pair / 2.0
    
    # Min distance to boundary
    norms = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.max(norms)
    
    return min(min_pair_r, min_boundary)

def _hexagonal_initial():
    """Generate initial hexagonal lattice packing."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points, radius = points[mask], radius[mask]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

def _soft_min_surrogate(points, beta=50.0):
    """
    Compute a smooth surrogate for the minimum of:
    - min_{i<j} |c_i - c_j| / 2
    - min_i (1 - |c_i|)
    
    Uses log-sum-exp for smooth min. Returns (value, gradient).
    """
    n = len(points)
    # All pairwise distances / 2
    # d_ij = |c_i - c_j| / 2 for i < j
    # boundary terms: 1 - |c_i|
    
    # Vectorized pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2)) / 2.0
    # Set diagonal to inf (we don't want self-pairs)
    np.fill_diagonal(dists, np.inf)
    
    # Boundary terms
    norms = np.linalg.norm(points, axis=1)
    boundary = 1.0 - norms
    
    # Combine all constraints into one array
    # Pairwise: N*(N-1)/2 values
    # Boundary: N values
    pair_vals = dists[np.triu_indices(n, k=1)]
    all_vals = np.concatenate([pair_vals, boundary])
    
    # Soft min via log-sum-exp: min(x) ≈ -log(sum(exp(-beta*x)))/beta
    # For numerical stability, subtract max
    shifted = -beta * all_vals
    shifted_max = np.max(shifted)
    log_sum = shifted_max + np.log(np.sum(np.exp(shifted - shifted_max)))
    soft_min_val = -log_sum / beta
    
    # Gradient of soft min with respect to points
    # d(soft_min)/d(points[k]) = -1/beta * sum_j w_j * d(all_vals_j)/d(points[k])
    # where w_j = exp(-beta * all_vals_j) / sum(exp(-beta * all_vals))
    
    weights = np.exp(shifted - shifted_max)
    weights /= np.sum(weights)
    
    grad = np.zeros_like(points)
    
    # Pairwise gradient contribution
    # For pair (i,j), all_vals = dists[i,j] = |c_i - c_j|/2
    # d/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|) = (c_i - c_j) / (2 * 2 * dists[i,j]) ... wait
    # dists[i,j] = |c_i - c_j| / 2
    # d(dists[i,j])/d(c_i) = (c_i - c_j) / |c_i - c_j| * (1/2)
    # Let's redo: dists[i,j] = 0.5 * ||c_i - c_j||
    # d/d(c_i) = 0.5 * (c_i - c_j) / ||c_i - c_j||
    
    # We need to accumulate gradients from all pairs
    # For each pair (i,j) with weight w, contribution to grad[i] += w * 0.5 * (c_i - c_j)/||c_i - c_j||
    # and grad[j] -= w * 0.5 * (c_i - c_j)/||c_i - c_j||
    
    # This is O(N^2) but N=50 so it's fine
    # Map from pair index to (i,j)
    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            w = weights[pair_idx]
            if w > 1e-15:
                d_vec = points[i] - points[j]
                d_norm = np.linalg.norm(d_vec)
                if d_norm > 1e-10:
                    unit = d_vec / d_norm
                    contrib = 0.5 * w * unit
                    grad[i] += contrib
                    grad[j] -= contrib
            pair_idx += 1
    
    # Boundary gradient
    # boundary[i] = 1 - ||c_i||
    # d/d(c_i) = -c_i / ||c_i||
    for i in range(n):
        w = weights[pair_idx + i]
        if w > 1e-15:
            norm_i = norms[i]
            if norm_i > 1e-10:
                grad[i] -= w * points[i] / norm_i
    
    return soft_min_val, grad

def _objective(x, beta=50.0):
    """Objective for L-BFGS-B: minimize -soft_min + penalty for boundary violation."""
    pts = x.reshape(N, 2)
    val, grad = _soft_min_surrogate(pts, beta)
    
    # Penalty for points outside unit circle
    norms = np.linalg.norm(pts, axis=1)
    boundary_violation = np.maximum(0, norms - 1.0)
    penalty = 10.0 * np.sum(boundary_violation)
    
    # Gradient of penalty
    grad_x = -grad
    for i in range(N):
        if norms[i] > 1.0:
            grad_x[i] += 10.0 * pts[i] / norms[i]
    
    return val + penalty, grad_x

def _ring_initialization(n_rings=5):
    """Generate initial points in concentric rings."""
    points = []
    # Center point
    points.append([0.0, 0.0])
    
    remaining = N - 1
    rings_per = max(1, remaining // n_rings)
    
    for ring in range(n_rings):
        r = (ring + 1) * 0.18  # approximate radius
        count = min(rings_per, remaining)
        remaining -= count
        for k in range(count):
            angle = 2 * np.pi * k / count + ring * np.pi / n_rings
            points.append([r * np.cos(angle), r * np.sin(angle)])
    
    points = np.array(points[:N])
    return points

def _polish_smooth(points, deadline):
    """Polish using L-BFGS-B with smooth surrogate, multi-stage beta."""
    best_points = points.copy()
    best_score = _compute_score(best_points)
    
    # Multi-stage beta annealing: start with low beta (smoother), increase
    betas = [10.0, 25.0, 50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        x0 = best_points.flatten()
        result = minimize(_objective, x0, args=(beta,), method='L-BFGS-B',
                         jac=True, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
        
        polished = result.x.reshape(N, 2)
        score = _compute_score(polished)
        
        if score > best_score:
            best_score = score
            best_points = polished
    
    return best_points, best_score

def _local_search_refine(points, deadline, rng):
    """Local search: try small perturbations of individual points."""
    best_points = points.copy()
    best_score = _compute_score(best_points)
    
    step_size = 0.001
    for iteration in range(20000):
        if time.time() > deadline:
            break
        
        idx = rng.integers(0, N)
        delta = rng.normal(0, step_size, size=2)
        new_points = best_points.copy()
        new_points[idx] += delta
        
        # Keep in unit circle
        norm = np.linalg.norm(new_points[idx])
        if norm > 0.999:
            new_points[idx] *= 0.999 / norm
        
        new_score = _compute_score(new_points)
        if new_score > best_score:
            best_score = new_score
            best_points = new_points
        else:
            # Occasionally shrink step
            if iteration % 5000 == 0 and iteration > 0:
                step_size *= 0.9
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with hexagonal lattice
    best_points = _hexagonal_initial()
    best_score = _compute_score(best_points)
    
    # Try ring initialization too
    ring_points = _ring_initialization()
    ring_score = _compute_score(ring_points)
    if ring_score > best_score:
        best_points = ring_points
        best_score = ring_score
    
    # Smooth polish with multi-stage beta
    if time.time() < deadline - 30:
        best_points, best_score = _polish_smooth(best_points, deadline - 30)
    
    # Local search refinement
    if time.time() < deadline - 10:
        rng = np.random.default_rng(42)
        best_points, best_score = _local_search_refine(best_points, deadline - 10, rng)
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        x0 = best_points.flatten()
        result = minimize(_objective, x0, args=(500.0,), method='L-BFGS-B',
                         jac=True, options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12})
        polished = result.x.reshape(N, 2)
        score = _compute_score(polished)
        if score > best_score:
            best_points = polished
            best_score = score
    
    return best_points
