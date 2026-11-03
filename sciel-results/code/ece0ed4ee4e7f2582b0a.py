"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _hex_lattice(n_target):
    """Generate points on a hexagonal lattice, then trim to fit in unit disk."""
    # Start with a spacing that gives roughly n_target points in the disk
    # Area of unit disk = pi. Each hex cell has area ~ (sqrt(3)/2) * s^2 for spacing s
    # We want ~ n_target cells, so s^2 ~ pi / (n_target * sqrt(3)/2)
    s = np.sqrt(2.0 * np.pi / (n_target * np.sqrt(3.0)))
    
    best_points = None
    best_count = 0
    
    for _ in range(10):
        k = int(1.0 / s) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0:
                    rows.append((x, y))
        points = np.array(rows)
        if len(points) >= n_target:
            # Sort by radius, take closest to center
            radii = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(radii, kind='stable')[:n_target]
            pts = points[idx]
            if len(pts) > best_count:
                best_count = len(pts)
                best_points = pts
            break
        s *= 0.95
    
    if best_points is None:
        # Fallback: uniform random
        angles = np.linspace(0, 2*np.pi, n_target, endpoint=False)
        radii = np.sqrt(np.random.rand(n_target)) * 0.9
        best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    return best_points[:n_target]


def _score(points):
    """Compute the score: min of (min pair dist / 2) and (min (1 - |c_i|))."""
    n = len(points)
    # Pairwise distances
    min_pair = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_pair:
                min_pair = d
    min_pair = min_pair / 2.0
    
    # Distance to boundary
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    return min(min_pair, min_boundary)


def _score_fast(points):
    """Faster score using numpy vectorization."""
    n = len(points)
    # Use broadcasting for pairwise distances (memory intensive for n=300 but ok)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    return min(min_pair, min_boundary)


def _local_optimize(points, max_iter=5000, timeout=100):
    """Optimize the packing using L-BFGS-B on the negative score."""
    start_time = time.time()
    
    def objective(x):
        if time.time() - start_time > timeout:
            return 1e10  # Penalize to stop
        pts = x.reshape(-1, 2)
        # Check validity
        norms = np.linalg.norm(pts, axis=1)
        if np.any(norms > 1.0):
            # Push back penalty
            penalty = np.sum(np.maximum(norms - 1.0, 0)**2) * 1000
            return -_score_fast(pts) + penalty
        return -_score_fast(pts)
    
    # Initial gradient estimate
    x0 = points.flatten()
    
    # Use L-BFGS-B with numerical gradients
    result = minimize(
        objective, x0, method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    if result.success or result.fun < objective(x0):
        return result.x.reshape(-1, 2)
    return points


def _refine_with_newton(points, max_iter=100):
    """Refine using active-set Newton method on binding constraints."""
    # This is a simplified version - find the most constrained pairs and adjust
    n = len(points)
    pts = points.copy()
    
    for it in range(max_iter):
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find the minimum distance pair
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_dist = dists[min_idx]
        
        # Check boundary
        norms = np.linalg.norm(pts, axis=1)
        min_boundary_idx = np.argmin(1.0 - norms)
        min_boundary = 1.0 - norms[min_boundary_idx]
        
        current_score = min(min_dist / 2.0, min_boundary)
        
        # If we've converged enough, break
        if min_dist / 2.0 > 0.05315 and min_boundary > 0.05315:
            break
        
        # Simple gradient ascent step on the score
        # Compute gradient of the score w.r.t. each point
        grad = np.zeros_like(pts)
        
        if min_dist / 2.0 <= min_boundary:
            # The limiting constraint is the pair
            i, j = min_idx
            if i > j:
                i, j = j, i
            direction = pts[i] - pts[j]
            if np.linalg.norm(direction) > 1e-10:
                direction /= np.linalg.norm(direction)
            # Move i and j apart
            step = 0.001
            grad[i] -= direction * step
            grad[j] += direction * step
        else:
            # The limiting constraint is the boundary
            i = min_boundary_idx
            direction = pts[i]
            if np.linalg.norm(direction) > 1e-10:
                direction /= np.linalg.norm(direction)
            # Move i inward
            step = 0.001
            grad[i] -= direction * step
        
        # Apply gradient step with damping
        alpha = 0.5
        new_pts = pts + alpha * grad
        
        # Check if it improved
        if _score_fast(new_pts) > current_score:
            pts = new_pts
        else:
            # Try smaller step
            alpha = 0.1
            new_pts = pts + alpha * grad
            if _score_fast(new_pts) > current_score:
                pts = new_pts
            else:
                break
    
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Start with hexagonal lattice
    points = _hex_lattice(N)
    best_points = points.copy()
    best_score = _score_fast(points)
    
    # Phase 1: Local optimization with L-BFGS-B
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        optimized = _local_optimize(points, max_iter=3000, timeout=deadline - start_time - 30)
        opt_score = _score_fast(optimized)
        if opt_score > best_score:
            best_points = optimized
            best_score = opt_score
    
    # Phase 2: Try multiple restarts with perturbations
    elapsed = time.time() - start_time
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() - start_time > deadline - 20:
            break
        
        # Perturb the best solution
        noise_scale = 0.01 * (1.0 - r / n_restarts)
        perturbed = best_points + np.random.randn(N, 2) * noise_scale
        
        # Project back inside the unit disk
        norms = np.linalg.norm(perturbed, axis=1)
        mask = norms > 0.99
        perturbed[mask] *= 0.99 / norms[mask]
        
        # Local optimize the perturbed version
        remaining_time = deadline - time.time() - 5
        if remaining_time < 5:
            break
        
        optimized = _local_optimize(perturbed, max_iter=1000, timeout=min(remaining_time, 15))
        opt_score = _score_fast(optimized)
        
        if opt_score > best_score:
            best_points = optimized
            best_score = opt_score
    
    # Phase 3: Final Newton refinement
    if time.time() - start_time < deadline - 5:
        refined = _refine_with_newton(best_points, max_iter=50)
        ref_score = _score_fast(refined)
        if ref_score > best_score:
            best_points = refined
            best_score = ref_score
    
    # Ensure validity: all points inside unit disk
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] *= 0.999 / norms[mask]
    
    return best_points

# EVOLVE-BLOCK-END
