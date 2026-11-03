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

def _compute_score_fast(points):
    """Vectorized score computation for speed."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2)) / 2.0
    np.fill_diagonal(dists, np.inf)
    min_pair_r = np.min(dists)
    
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

def _ring_initialization(n_rings=5, seed=0):
    """Generate initial points in concentric rings with optional jitter."""
    rng = np.random.default_rng(seed)
    points = []
    # Center point
    points.append([0.0, 0.0])
    
    remaining = N - 1
    rings_per = max(1, remaining // n_rings)
    
    for ring in range(n_rings):
        r = (ring + 1) * 0.18
        count = min(rings_per, remaining)
        remaining -= count
        for k in range(count):
            angle = 2 * np.pi * k / count + ring * np.pi / n_rings
            jitter = rng.normal(0, 0.01, size=2)
            pt = [r * np.cos(angle) + jitter[0], r * np.sin(angle) + jitter[1]]
            norm = np.linalg.norm(pt)
            if norm > 0.99:
                pt = [pt[0] * 0.99 / norm, pt[1] * 0.99 / norm]
            points.append(pt)
    
    points = np.array(points[:N])
    return points

def _random_initialization(seed):
    """Generate random initial points inside unit disk."""
    rng = np.random.default_rng(seed)
    points = []
    # Try to place points with minimum spacing
    attempts = 0
    while len(points) < N and attempts < 10000:
        attempts += 1
        r = rng.uniform(0, 0.95)
        theta = rng.uniform(0, 2 * np.pi)
        pt = np.array([r * np.cos(theta), r * np.sin(theta)])
        
        # Check distance to existing points
        if points:
            existing = np.array(points)
            dists = np.linalg.norm(existing - pt, axis=1)
            if np.min(dists) < 0.15:
                continue
        
        points.append(pt)
    
    # If we didn't get enough, fill with more random points
    while len(points) < N:
        r = rng.uniform(0, 0.95)
        theta = rng.uniform(0, 2 * np.pi)
        pt = np.array([r * np.cos(theta), r * np.sin(theta)])
        points.append(pt)
    
    return np.array(points[:N])

def _soft_min_surrogate(points, beta=50.0):
    """
    Compute a smooth surrogate for the minimum of:
    - min_{i<j} |c_i - c_j| / 2
    - min_i (1 - |c_i|)
    
    Uses log-sum-exp for smooth min. Returns (value, gradient).
    """
    n = len(points)
    # Vectorized pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2)) / 2.0
    # Set diagonal to inf (we don't want self-pairs)
    np.fill_diagonal(dists, np.inf)
    
    # Boundary terms
    norms = np.linalg.norm(points, axis=1)
    boundary = 1.0 - norms
    
    # Combine all constraints into one array
    pair_vals = dists[np.triu_indices(n, k=1)]
    all_vals = np.concatenate([pair_vals, boundary])
    
    # Soft min via log-sum-exp: min(x) ≈ -log(sum(exp(-beta*x)))/beta
    shifted = -beta * all_vals
    shifted_max = np.max(shifted)
    log_sum = shifted_max + np.log(np.sum(np.exp(shifted - shifted_max)))
    soft_min_val = -log_sum / beta
    
    # Gradient of soft min with respect to points
    weights = np.exp(shifted - shifted_max)
    weights /= np.sum(weights)
    
    grad = np.zeros_like(points)
    
    # Pairwise gradient contribution
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

def _polish_smooth(points, deadline):
    """Polish using L-BFGS-B with smooth surrogate, multi-stage beta."""
    best_points = points.copy()
    best_score = _compute_score_fast(best_points)
    
    # Multi-stage beta annealing: start with low beta (smoother), increase
    betas = [5.0, 15.0, 30.0, 60.0, 120.0, 250.0]
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        x0 = best_points.flatten()
        result = minimize(_objective, x0, args=(beta,), method='L-BFGS-B',
                         jac=True, options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-11})
        
        polished = result.x.reshape(N, 2)
        # Ensure points stay in unit disk
        norms = np.linalg.norm(polished, axis=1)
        for i in range(N):
            if norms[i] > 1.0:
                polished[i] *= 0.999 / norms[i]
        
        score = _compute_score_fast(polished)
        
        if score > best_score:
            best_score = score
            best_points = polished
    
    return best_points, best_score

def _exact_local_search(points, deadline, rng):
    """Direct local search on the exact nonsmooth objective."""
    best_points = points.copy()
    best_score = _compute_score_fast(best_points)
    
    step_size = 0.002
    no_improve = 0
    max_no_improve = 200
    
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        idx = rng.integers(0, N)
        # Try multiple step sizes
        for ss in [step_size, step_size * 0.5, step_size * 0.1]:
            delta = rng.normal(0, ss, size=2)
            new_points = best_points.copy()
            new_points[idx] += delta
            
            # Keep in unit circle
            norm = np.linalg.norm(new_points[idx])
            if norm > 0.999:
                new_points[idx] *= 0.999 / norm
            
            new_score = _compute_score_fast(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
                no_improve = 0
                break
        else:
            no_improve += 1
        
        if no_improve >= max_no_improve:
            # Reduce step size
            step_size *= 0.8
            no_improve = 0
            if step_size < 1e-6:
                break
    
    return best_points, best_score

def _final_polish(points, deadline):
    """Final high-precision polish with very high beta."""
    best_points = points.copy()
    best_score = _compute_score_fast(best_points)
    
    betas = [500.0, 1000.0, 2000.0]
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        x0 = best_points.flatten()
        result = minimize(_objective, x0, args=(beta,), method='L-BFGS-B',
                         jac=True, options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-13})
        
        polished = result.x.reshape(N, 2)
        norms = np.linalg.norm(polished, axis=1)
        for i in range(N):
            if norms[i] > 1.0:
                polished[i] *= 0.9999 / norms[i]
        
        score = _compute_score_fast(polished)
        if score > best_score:
            best_score = score
            best_points = polished
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Generate multiple initializations
    initializations = []
    
    # Hexagonal lattice
    hex_pts = _hexagonal_initial()
    initializations.append(hex_pts)
    
    # Ring initializations with different seeds
    for seed in [0, 1, 2]:
        ring_pts = _ring_initialization(seed=seed)
        initializations.append(ring_pts)
    
    # Random initializations
    for seed in [42, 123, 456]:
        rand_pts = _random_initialization(seed)
        initializations.append(rand_pts)
    
    # Score all initializations and pick the best
    best_points = None
    best_score = -np.inf
    for pts in initializations:
        score = _compute_score_fast(pts)
        if score > best_score:
            best_score = score
            best_points = pts
    
    # Smooth polish the best initialization
    if time.time() < deadline - 40:
        best_points, best_score = _polish_smooth(best_points, deadline - 40)
    
    # Try polishing a few other good initializations
    time_remaining = deadline - time.time()
    if time_remaining > 20:
        # Sort initializations by score and try the next best
        scored_inits = [(pts, _compute_score_fast(pts)) for pts in initializations]
        scored_inits.sort(key=lambda x: x[1], reverse=True)
        
        for pts, init_score in scored_inits[1:4]:
            if time.time() > deadline - 15:
                break
            if init_score < best_score - 0.01:  # Only try if close
                polished_pts, polished_score = _polish_smooth(pts, deadline - 15)
                if polished_score > best_score:
                    best_score = polished_score
                    best_points = polished_pts
    
    # Exact local search refinement
    if time.time() < deadline - 10:
        rng = np.random.default_rng(42)
        best_points, best_score = _exact_local_search(best_points, deadline - 10, rng)
    
    # Final high-precision polish
    if time.time() < deadline - 3:
        best_points, best_score = _final_polish(best_points, deadline - 3)
    
    # Final validation and cleanup
    norms = np.linalg.norm(best_points, axis=1)
    for i in range(N):
        if norms[i] > 1.0:
            best_points[i] *= 0.99999 / norms[i]
    
    return best_points
