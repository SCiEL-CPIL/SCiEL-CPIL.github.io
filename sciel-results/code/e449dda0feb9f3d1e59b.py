"""Packing 50 equal circles in a unit circle via multi-start L-BFGS-B with penalty."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50

def _score(pts):
    """Compute the minimum radius r such that all circles are disjoint and inside."""
    # Distance from origin
    norms = np.linalg.norm(pts, axis=1)
    # Max distance from boundary
    boundary = 1.0 - norms
    # Minimum boundary violation (should be >= 0 for valid)
    min_boundary = np.min(boundary)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    min_radius = min_dist / 2.0
    
    return min(min_boundary, min_radius)

def _energy_and_grad(pts, penalty_weight):
    """
    Energy function for optimization. We want to maximize the minimum radius.
    Instead, we minimize a penalty for violations + a term that encourages larger radius.
    
    Actually, let's use a different approach: minimize the sum of squared violations
    where violations are:
    - For each pair: max(0, 2*r - dist_ij) where r is the target radius
    - For each point: max(0, r + norm_i - 1)
    
    But we need to choose r. Let's use a different formulation:
    Minimize: sum_{i<j} max(0, 2*r - dist_ij)^2 + sum_i max(0, r + norm_i - 1)^2
    with r as a variable, and we want to maximize r.
    
    Better: minimize -r + penalty for violations.
    """
    r = _score(pts)
    # This is expensive. Let's use a direct penalty approach.
    norms = np.linalg.norm(pts, axis=1)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # We want all dists >= 2r and norms <= 1-r
    # Let's define violations relative to a target radius
    # Actually, let's just minimize the negative of the minimum radius using a smooth approximation
    
    # Use the minimum radius directly with a smooth min
    min_dist = np.min(dists)
    min_boundary = 1.0 - np.max(norms)
    r = min(min_dist / 2.0, min_boundary)
    
    # Gradient: we need to push points apart and toward center
    # For the objective f = min(min_dist/2, 1-max_norm), the gradient is complex.
    # Let's use a sum-based penalty instead.
    
    # Minimize: sum of (max(0, 2*r_target - dist_ij))^2 + sum of (max(0, r_target + norm_i - 1))^2
    # where r_target is fixed during each optimization phase
    
    return r

def _construct_single_start(pts_init, maxiter, penalty_weight):
    """Run L-BFGS-B from a single initial configuration."""
    def objective_and_grad(pts):
        pts = pts.reshape(-1, 2)
        norms = np.linalg.norm(pts, axis=1)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # We want to maximize r = min(min_dist/2, 1-max_norm)
        # Use a smooth approximation: minimize -log(min_dist/2) - log(1-max_norm) + penalty
        # Actually, let's use a direct approach: minimize the sum of violations
        
        # Target: all dists >= 2r, all norms <= 1-r
        # Let's minimize: sum_{i<j} max(0, 2r - dist_ij)^2 + sum_i max(0, r + norm_i - 1)^2
        # with r = min(min_dist/2, 1-max_norm)
        
        min_dist = np.min(dists)
        min_boundary = 1.0 - np.max(norms)
        r = min(min_dist / 2.0, min_boundary)
        
        if r <= 0:
            # Invalid configuration, return large penalty
            val = 1e6
            grad = np.zeros_like(pts)
            return val, grad
        
        # Penalty for pair violations
        pair_viol = np.maximum(0, 2*r - dists)
        pair_viol[np.diag_indices_from(pair_viol)] = 0
        
        # Penalty for boundary violations
        bound_viol = np.maximum(0, r + norms - 1.0)
        
        val = np.sum(pair_viol**2) + penalty_weight * np.sum(bound_viol**2)
        
        # Gradient computation
        grad = np.zeros_like(pts)
        
        # Pair gradient
        for i in range(N):
            for j in range(i+1, N):
                if pair_viol[i, j] > 0:
                    d = pts[i] - pts[j]
                    norm_d = np.linalg.norm(d)
                    if norm_d > 1e-12:
                        grad[i] += 4 * pair_viol[i, j] * d / norm_d
                        grad[j] -= 4 * pair_viol[i, j] * d / norm_d
        
        # Boundary gradient
        for i in range(N):
            if bound_viol[i] > 0:
                norm_i = norms[i]
                if norm_i > 1e-12:
                    grad[i] += 2 * penalty_weight * bound_viol[i] * pts[i] / norm_i
        
        return val, grad
    
    try:
        result = minimize(objective_and_grad, pts_init.reshape(-1), method='L-BFGS-B',
                          jac=True, options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})
        pts_opt = result.x.reshape(-1, 2)
        return pts_opt, _score(pts_opt)
    except Exception:
        return pts_init, _score(pts_init)

def _make_hexagonal_init(n):
    """Create a hexagonal lattice initialization."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= n:
            idx = np.argsort(radius, kind='stable')[:n]
            return points[idx]
        spacing *= 0.98

def _make_random_init(n, seed):
    """Create a random initialization inside the unit disk."""
    rng = np.random.RandomState(seed)
    pts = rng.uniform(-1, 1, size=(n, 2))
    norms = np.linalg.norm(pts, axis=1)
    mask = norms <= 1.0
    while np.sum(mask) < n:
        pts = rng.uniform(-1, 1, size=(n, 2))
        norms = np.linalg.norm(pts, axis=1)
        mask = norms <= 1.0
    return pts[mask][:n]

def _make_ring_init(n, seed):
    """Create a ring-based initialization."""
    rng = np.random.RandomState(seed)
    pts = []
    # Center
    pts.append([0.0, 0.0])
    # Rings
    ring_counts = [1, 6, 12, 18]  # 1+6+12+18 = 37, need 50
    remaining = n - sum(ring_counts)
    if remaining > 0:
        ring_counts.append(remaining)
    
    radii = [0.0, 0.25, 0.5, 0.75, 0.95]
    for ring_idx, count in enumerate(ring_counts):
        if ring_idx >= len(radii):
            break
        r = radii[ring_idx]
        angles = np.linspace(0, 2*np.pi, count, endpoint=False) + rng.uniform(0, 2*np.pi/count)
        for a in angles:
            pts.append([r * np.cos(a), r * np.sin(a)])
    
    pts = np.array(pts[:n])
    # Scale to fit
    norms = np.linalg.norm(pts, axis=1)
    if len(norms) > 0 and np.max(norms) > 0:
        scale = 0.95 / np.max(norms)
        pts *= scale
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    best_pts = _make_hexagonal_init(N)
    best_score = _score(best_pts)
    
    # Multi-start optimization
    initializations = []
    
    # 1. Hexagonal
    initializations.append(_make_hexagonal_init(N))
    
    # 2. Ring-based
    for seed in range(5):
        initializations.append(_make_ring_init(N, seed))
    
    # 3. Random
    for seed in range(10):
        initializations.append(_make_random_init(N, seed + 100))
    
    # 4. Perturbed hexagonal
    hex_pts = _make_hexagonal_init(N)
    for seed in range(5):
        rng = np.random.RandomState(seed + 200)
        perturbed = hex_pts + rng.normal(0, 0.05, size=hex_pts.shape)
        # Clamp to unit disk
        norms = np.linalg.norm(perturbed, axis=1)
        mask = norms > 1.0
        perturbed[mask] *= (0.95 / norms[mask])[:, np.newaxis]
        initializations.append(perturbed)
    
    # Optimize each initialization
    for i, init_pts in enumerate(initializations):
        if time.time() > deadline:
            break
        
        # Adaptive maxiter based on remaining time
        remaining = deadline - time.time()
        maxiter = max(50, int(remaining * 2))
        
        pts_opt, score = _construct_single_start(init_pts, maxiter, penalty_weight=10.0)
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt
        
        # If time is getting tight, reduce iterations for subsequent runs
        if time.time() > deadline - 20:
            maxiter = 20
    
    return best_pts
