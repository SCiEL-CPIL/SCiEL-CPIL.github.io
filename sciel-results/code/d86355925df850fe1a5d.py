"""Packing 300 equal circles in a unit circle.

Strategy:
1. Generate multiple initial configurations:
   a) Hexagonal lattice with fine spacing; pick the N closest-to-origin points.
   b) Fibonacci sphere distribution mapped to the disk for uniform, non-lattice start.
2. Refine with a vectorised local-search / repulsion loop:
   - Each iteration picks a random point, proposes a small perturbation,
     and accepts if the computed radius improves.
   - Also run a repulsion-based relaxation pass to untangle close pairs.
   - Use soft penalty function for overlaps and boundary violations.
3. Deadline guard ensures we return the best configuration found before
   the 150 s wall-clock kill.
"""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 300

def _compute_radius(pts):
    """Compute the radius for a configuration of N points."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Boundary constraint
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    boundary_min = 1.0 - np.max(dists_origin)
    if boundary_min <= 0:
        return 0.0
    # Pairwise distances using vectorized approach
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(min_pair / 2.0, boundary_min)


def _compute_radius_fast(pts):
    """Faster radius computation using only nearest-neighbor info."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    boundary_min = 1.0 - np.max(dists_origin)
    if boundary_min <= 0:
        return 0.0
    # Use scipy's cKDTree for efficient nearest neighbor
    tree = cKDTree(pts)
    # Get 2 nearest neighbors for each point (one is self, so we need 2)
    dists, _ = tree.query(pts, k=2)
    min_pair = np.min(dists[:, 1])  # exclude self-distance
    return min(min_pair / 2.0, boundary_min)


def _hex_initial(spacing, n_points=N):
    """Generate hexagonal lattice points and select the best n_points."""
    sqrt3 = np.sqrt(3.0)
    k = int(1.0 / spacing) + 3
    
    points_list = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * sqrt3 / 2.0
            r = np.hypot(x, y)
            if r < 1.0:
                points_list.append((x, y))
    
    if len(points_list) < n_points:
        return None, 0.0
    
    points = np.array(points_list)
    dists = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(dists, kind='stable')[:n_points]
    pts = points[idx]
    r = _compute_radius(pts)
    return pts, r


def _fibonacci_disk(n_points=N):
    """Generate points using Fibonacci sphere distribution mapped to disk."""
    # Use golden angle for uniform distribution in disk
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    pts = np.zeros((n_points, 2))
    
    for i in range(n_points):
        # Radius increases with sqrt(i/n) for uniform density
        r = np.sqrt((i + 0.5) / n_points)
        theta = golden_angle * i
        pts[i, 0] = r * np.cos(theta)
        pts[i, 1] = r * np.sin(theta)
    
    # Scale to fit within unit circle with margin
    r_mags = np.hypot(pts[:, 0], pts[:, 1])
    max_r = np.max(r_mags)
    if max_r > 0.99:
        pts *= 0.99 / max_r
    
    r = _compute_radius(pts)
    return pts, r


def _local_search(pts, deadline, seed=42, use_fast=True):
    """Local search: perturb random points, accept if radius improves."""
    rng = np.random.RandomState(seed)
    current = pts.copy()
    compute_r = _compute_radius_fast if use_fast else _compute_radius
    current_r = compute_r(current)
    best = current.copy()
    best_r = current_r
    
    step_size = 0.008
    patience = 0
    max_patience = 800
    best_streak = 0
    
    while time.time() < deadline and patience < max_patience:
        # Pick a point that's involved in a close pair or near boundary
        tree = cKDTree(current)
        dists, _ = tree.query(current, k=2)
        close_pairs = np.argmin(dists[:, 1])
        close_dist = dists[close_pairs, 1]
        
        # Prefer points with small nearest-neighbor distance
        if close_dist < 0.11 and rng.random() < 0.7:
            idx = close_pairs
        else:
            idx = rng.randint(N)
        
        old_pos = current[idx].copy()
        
        # Try multiple perturbations and pick the best
        best_new_r = current_r
        best_new_pos = old_pos.copy()
        
        n_attempts = 8 if patience < 100 else 5
        for attempt in range(n_attempts):
            # Scale step size based on current radius
            effective_step = step_size * (1.0 + 0.5 * (1.0 - current_r / 0.055))
            new_pos = old_pos + rng.uniform(-effective_step, effective_step, 2)
            
            # Keep inside circle
            new_r_mag = np.hypot(new_pos[0], new_pos[1])
            if new_r_mag > 0.995:
                new_pos = new_pos / new_r_mag * 0.995
            
            current[idx] = new_pos
            new_r = compute_r(current)
            
            if new_r > best_new_r:
                best_new_r = new_r
                best_new_pos = new_pos.copy()
        
        current[idx] = best_new_pos
        
        if best_new_r > current_r + 1e-12:
            current_r = best_new_r
            patience = 0
            best_streak += 1
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
        else:
            current[idx] = old_pos
            patience += 1
            best_streak = 0
            # Adaptive step size
            if patience % 30 == 0:
                step_size *= 0.93
            if step_size < 1e-6:
                break
    
    return best, best_r


def _repulsion_relax(pts, deadline, seed=123):
    """Repulsion-based relaxation to untangle close pairs."""
    rng = np.random.RandomState(seed)
    current = pts.copy()
    current_r = _compute_radius_fast(current)
    best = current.copy()
    best_r = current_r
    
    # Iterative repulsion: push apart multiple close pairs per iteration
    for iteration in range(1000):
        if time.time() > deadline:
            break
        
        # Find close pairs using KDTree
        tree = cKDTree(current)
        dists, pairs = tree.query(current, k=2)
        
        min_dist = np.min(dists[:, 1])
        target_spacing = 2 * (0.0532)  # slightly above known best radius
        if min_dist > target_spacing:
            break
        
        # Process all pairs that are too close (up to 5 per iteration)
        close_mask = dists[:, 1] < target_spacing
        close_indices = np.where(close_mask)[0]
        
        if len(close_indices) == 0:
            break
        
        # Process up to 5 closest pairs
        n_to_process = min(5, len(close_indices))
        # Sort by distance and take closest
        sorted_idx = close_indices[np.argsort(dists[close_indices, 1])][:n_to_process]
        
        for min_idx in sorted_idx:
            i, j = pairs[min_idx, 0], pairs[min_idx, 1]
            if i == j:
                continue
                
            # Push them apart
            direction = current[i] - current[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = rng.randn(2)
                norm = np.linalg.norm(direction)
            direction = direction / norm
            
            # Adaptive push: stronger for closer pairs
            close_ratio = dists[min_idx, 1] / target_spacing
            push = 0.004 * max(0.1, 1.0 - close_ratio)
            current[i] += direction * push
            current[j] -= direction * push
            
            # Keep inside circle
            for k in [i, j]:
                r_mag = np.hypot(current[k, 0], current[k, 1])
                if r_mag > 0.998:
                    current[k] = current[k] / r_mag * 0.998
        
        new_r = _compute_radius_fast(current)
        if new_r > current_r + 1e-12:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = current.copy()
    
    return best, best_r


def _soft_penalty_relax(pts, deadline, seed=789):
    """Relaxation using soft penalty function for overlaps and boundary violations."""
    rng = np.random.RandomState(seed)
    current = pts.copy()
    current_r = _compute_radius_fast(current)
    best = current.copy()
    best_r = current_r
    
    # Parameters for soft penalty
    beta_overlap = 50.0  # Sharpness of overlap penalty
    beta_boundary = 80.0  # Sharpness of boundary penalty
    learning_rate = 0.002
    
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        # Compute soft penalty gradient
        # Overlap penalty: for pairs with distance < 2r, push apart
        tree = cKDTree(current)
        dists, pairs = tree.query(current, k=3)  # Get 2 nearest neighbors
        
        # Compute forces from overlaps
        forces = np.zeros_like(current)
        target_r = max(current_r, 0.05)
        
        for i in range(N):
            for k in range(1, 3):  # Skip self
                j = pairs[i, k]
                if j < 0:
                    continue
                d = dists[i, k]
                if d < 2 * target_r:
                    # Overlap: push apart with soft penalty
                    direction = current[i] - current[j]
                    norm = np.linalg.norm(direction)
                    if norm < 1e-10:
                        direction = rng.randn(2)
                        norm = np.linalg.norm(direction)
                    direction = direction / norm
                    
                    # Soft penalty: exponential decay
                    overlap = (2 * target_r - d) / (2 * target_r)
                    force = np.exp(-beta_overlap * overlap) * overlap
                    forces[i] += direction * force
                    forces[j] -= direction * force
        
        # Boundary penalty: push inward if too close to boundary
        r_mags = np.hypot(current[:, 0], current[:, 1])
        boundary_violation = np.maximum(0, r_mags - (1 - target_r))
        boundary_force = np.exp(-beta_boundary * boundary_violation) * boundary_violation
        forces -= (boundary_force[:, np.newaxis] * current) / np.maximum(r_mags, 1e-10)[:, np.newaxis]
        
        # Apply forces with learning rate
        current += learning_rate * forces
        
        # Keep inside circle
        r_mags = np.hypot(current[:, 0], current[:, 1])
        mask = r_mags > 0.998
        if np.any(mask):
            current[mask] = current[mask] / r_mags[mask, np.newaxis] * 0.998
        
        new_r = _compute_radius_fast(current)
        if new_r > current_r + 1e-12:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = current.copy()
        
        # Decay learning rate
        learning_rate *= 0.995
    
    return best, best_r


def _gradient_ascent(pts, deadline, seed=456):
    """Gradient ascent on a smooth surrogate objective."""
    from scipy.optimize import minimize
    
    rng = np.random.RandomState(seed)
    current = pts.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    # Define smooth surrogate using epigraph formulation with auxiliary variable t
    # Minimize -t subject to: t <= min_pair/2 and t <= 1 - max_dist
    # Use soft constraints via log-sum-exp
    
    def objective(x):
        x = x.reshape(-1, 2)
        # Boundary term
        dists_origin = np.hypot(x[:, 0], x[:, 1])
        max_dist = np.max(dists_origin)
        
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        # Soft min of pairwise distances (using squared distances for stability)
        beta = 80.0
        # Only consider upper triangle to avoid double counting
        iu = np.triu_indices(N, k=1)
        dists_sq_iu = dists_sq[iu]
        
        # Soft min of squared distances
        exp_vals = np.exp(-beta * dists_sq_iu)
        log_sum_exp = np.log(np.sum(exp_vals))
        soft_min_sq = -log_sum_exp / beta
        
        soft_min_dist = np.sqrt(soft_min_sq)
        
        # Soft min of (soft_min_dist/2, 1 - max_dist) using log-sum-exp
        a = soft_min_dist / 2.0
        b = 1.0 - max_dist
        gamma = 100.0
        soft_min_ab = -np.log(np.exp(-gamma * a) + np.exp(-gamma * b)) / gamma
        
        # Objective: minimize -soft_min_ab (i.e., maximize soft_min_ab)
        return -soft_min_ab
    
    def objective_grad(x):
        x = x.reshape(-1, 2)
        # Boundary term
        dists_origin = np.hypot(x[:, 0], x[:, 1])
        max_dist = np.max(dists_origin)
        
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        iu = np.triu_indices(N, k=1)
        dists_sq_iu = dists_sq[iu]
        
        # Soft min of squared distances
        beta = 80.0
        exp_vals = np.exp(-beta * dists_sq_iu)
        log_sum_exp = np.log(np.sum(exp_vals))
        soft_min_sq = -log_sum_exp / beta
        soft_min_dist = np.sqrt(soft_min_sq)
        
        # Weights for soft min
        weights_iu = exp_vals / np.sum(exp_vals)
        
        # Gradient of soft_min_dist w.r.t. x
        # d(soft_min_dist)/dx = (1/(2*sqrt(soft_min_sq))) * d(soft_min_sq)/dx
        # d(soft_min_sq)/dx[i] = sum_j weights[i,j] * (-2*beta) * (x[i]-x[j]) * (x[i]-x[j])_k
        # This is complex; use numerical gradient for reliability
        eps = 1e-7
        grad = np.zeros_like(x)
        for i in range(min(N, 50)):  # Limit for speed
            for d in range(2):
                x_plus = x.copy()
                x_plus[i, d] += eps
                x_minus = x.copy()
                x_minus[i, d] -= eps
                grad[i, d] = (objective(x_plus.flatten()) - objective(x_minus.flatten())) / (2 * eps)
        
        # Add boundary gradient
        max_idx = np.argmax(dists_origin)
        boundary_weight = 0.5
        grad[max_idx] += boundary_weight * x[max_idx] / max_dist
        
        return grad.flatten()
    
    # Run gradient ascent with multiple restarts
    x = current.flatten()
    n_restarts = 3
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        try:
            result = minimize(objective, x, jac=objective_grad, method='L-BFGS-B',
                             options={'maxiter': 50, 'gtol': 1e-4, 'ftol': 1e-8})
            x_new = result.x.reshape(-1, 2)
            
            # Check if improvement
            new_r = _compute_radius_fast(x_new)
            if new_r > current_r + 1e-12:
                current = x_new
                current_r = new_r
                x = x_new.flatten()
                if new_r > best_r:
                    best_r = new_r
                    best = current.copy()
            else:
                # Perturb for next restart
                x = current.flatten() + rng.normal(0, 0.001, N * 2)
                x = x.reshape(-1, 2)
                # Keep inside circle
                r_mags = np.hypot(x[:, 0], x[:, 1])
                mask = r_mags > 0.995
                x[mask] = x[mask] / r_mags[mask, np.newaxis] * 0.995
                x = x.flatten()
        except Exception:
            break
    
    return best, best_r


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    def time_check():
        return time.time() < deadline
    
    # --- Stage 1: Construct a good initial configuration ---
    best_points = None
    best_radius = 0.0
    
    # Try multiple spacings with finer granularity
    for trial in range(80):
        if not time_check():
            break
        
        spacing = 0.078 + trial * 0.0008
        pts, r = _hex_initial(spacing)
        
        if pts is not None and r > best_radius:
            best_radius = r
            best_points = pts.copy()
    
    # Also try Fibonacci disk initialization
    if time_check():
        fib_pts, fib_r = _fibonacci_disk()
        if fib_r > best_radius:
            best_radius = fib_r
            best_points = fib_pts.copy()
    
    if best_points is None:
        # Fallback: uniform random in circle
        rng = np.random.RandomState(0)
        angles = rng.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(rng.uniform(0, 1, N))
        best_points = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles)
        ])
        best_radius = _compute_radius(best_points)
    
    # --- Stage 2: Local optimization ---
    # Run repulsion relaxation first
    rep_budget = deadline - 50.0
    rep_pts, rep_r = _repulsion_relax(best_points, rep_budget, seed=123)
    if rep_r > best_radius:
        best_points = rep_pts
        best_radius = rep_r
    
    # Run soft penalty relaxation
    if time_check():
        soft_budget = deadline - 40.0
        soft_pts, soft_r = _soft_penalty_relax(best_points, soft_budget, seed=789)
        if soft_r > best_radius:
            best_points = soft_pts
            best_radius = soft_r
    
    # Run local search with different seeds and more iterations
    for seed in [42, 777, 999, 1234]:
        if not time_check():
            break
        
        ls_budget = deadline - 25.0
        ls_pts, ls_r = _local_search(best_points, ls_budget, seed=seed, use_fast=True)
        if ls_r > best_radius:
            best_points = ls_pts
            best_radius = ls_r
    
    # Run gradient ascent if time permits
    if time_check():
        grad_budget = deadline - 10.0
        grad_pts, grad_r = _gradient_ascent(best_points, grad_budget, seed=456)
        if grad_r > best_radius:
            best_points = grad_pts
            best_radius = grad_r
    
    # Final high-precision polish with exact radius computation
    if time_check():
        polish_budget = deadline - 3.0
        polish_pts, polish_r = _local_search(best_points, polish_budget, seed=12345, use_fast=False)
        if polish_r > best_radius:
            best_points = polish_pts
            best_radius = polish_r
    
    # Final validation and return
    # Ensure points are strictly inside
    dists_origin = np.hypot(best_points[:, 0], best_points[:, 1])
    max_allowed = 1.0 - 1e-10
    mask = dists_origin >= max_allowed
    if np.any(mask):
        best_points[mask] *= (max_allowed / dists_origin[mask]) * 0.999
    
    # Ensure no coincident points (vectorized check)
    tree = cKDTree(best_points)
    dists, pairs = tree.query(best_points, k=2)
    close_mask = dists[:, 1] < 1e-6
    if np.any(close_mask):
        close_indices = np.where(close_mask)[0]
        rng_fix = np.random.RandomState(999)
        for idx in close_indices:
            j = pairs[idx, 1]
            angle = rng_fix.uniform(0, 2 * np.pi)
            best_points[idx, 0] += 0.01 * np.cos(angle)
            best_points[idx, 1] += 0.01 * np.sin(angle)
            best_points[j, 0] -= 0.01 * np.cos(angle)
            best_points[j, 1] -= 0.01 * np.sin(angle)
    
    # Final check
    if not np.all(np.isfinite(best_points)) or best_points.shape != (N, 2):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.sqrt(np.linspace(0, 1, N))
        best_points = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles)
        ])
        best_points *= 0.99
    
    return best_points
