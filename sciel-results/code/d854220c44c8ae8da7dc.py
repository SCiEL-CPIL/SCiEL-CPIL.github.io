"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 50
_DEADLINE = 115.0  # seconds, safety margin before 120s wall-clock kill


def _hex_grid_init(n):
    """Generate a hexagonal lattice initial packing, trimmed to n points."""
    best_pts = None
    best_score = 0.0
    for s in np.linspace(0.15, 0.35, 40):
        pts = []
        k = int(1.0 / s) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                if x * x + y * y <= (1.0 - s / 2.0) ** 2:
                    pts.append((x, y))
        if len(pts) >= n:
            pts_arr = np.array(pts)
            dists = np.hypot(pts_arr[:, 0], pts_arr[:, 1])
            idx = np.argsort(dists, kind='stable')[:n]
            candidate = pts_arr[idx]
            score = _score(candidate)
            if score > best_score:
                best_score = score
                best_pts = candidate
    if best_pts is not None:
        return best_pts
    return np.random.randn(n, 2) * 0.5


def _score(pts):
    """Compute the packing radius (minimum of min pair dist/2 and min distance to boundary)."""
    n = len(pts)
    if n == 0:
        return 0.0
    # Vectorized pairwise distances
    # Use broadcasting: (n, 1, 2) - (1, n, 2) -> (n, n, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Distance to boundary
    norms = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(1.0 - norms)
    
    return min(min_pair / 2.0, min_boundary)


def _relax(pts, max_iter=200, tol=1e-12):
    """
    Relax the packing using a force-based method.
    Returns improved pts.
    """
    pts = pts.copy()
    n = len(pts)
    r = _score(pts)
    if r <= 0:
        return pts
    
    # Learning rate
    lr = 0.01
    
    for _ in range(max_iter):
        # Compute distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        norms = np.hypot(pts[:, 0], pts[:, 1])
        
        # Current radius
        min_pair = np.min(dists)
        min_bound = np.min(1.0 - norms)
        current_r = min(min_pair / 2.0, min_bound)
        
        if current_r <= r * (1 + tol):
            break
            
        r = current_r
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        # We want to push apart pairs where dist < 2r
        # Force magnitude proportional to (2r - dist)
        # Direction is normalized diff
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            # dists is (n, n)
            # We only care about i < j
            # Create a mask for active constraints
            active = dists < 2.0 * r
            # For each active pair, calculate force
            # F_ij = (2r - d_ij) * (p_i - p_j) / d_ij
            # This is a vector.
            # We can compute this for all pairs and sum up.
            
            # To be efficient, we can just compute for all pairs and let the mask zero out inactive ones.
            # But we need to be careful with division by zero for inactive pairs.
            # Since active implies dist < 2r, and r > 0, dist > 0 is guaranteed for active pairs?
            # If two points coincide, dist=0. This is a singular case.
            # If dist < 2r and dist is very small, force is large.
            
            # Let's compute forces for all pairs
            # diff is (n, n, 2)
            # dists is (n, n)
            
            # Force on i from j:
            # f_ij = (2r - d_ij) * (p_i - p_j) / d_ij
            # We can write this as:
            # f_ij = (2r - d_ij) / d_ij * (p_i - p_j)
            
            # Let's compute the scalar coefficient
            # coeff_ij = (2r - d_ij) / d_ij
            # Set to 0 where not active or d_ij == 0
            
            coeff = np.zeros_like(dists)
            mask = (dists < 2.0 * r) & (dists > 1e-12)
            coeff[mask] = (2.0 * r - dists[mask]) / dists[mask]
            
            # Force on i from all j: sum_j coeff_ij * (p_i - p_j)
            # p_i is pts[i], p_j is pts[j]
            # diff[i, j] = p_i - p_j
            # So force_i = sum_j coeff[i, j] * diff[i, j]
            
            forces = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
            
            # Boundary repulsion
            # If 1 - |p_i| < r, push inwards
            # Force = (r - (1 - |p_i|)) * (-p_i / |p_i|)
            # Note: if p_i is at origin, direction is undefined. But if p_i is at origin, |p_i|=0, 1-0=1 > r usually.
            # If r is very large, maybe. But r < 0.5.
            
            bound_violation = r - (1.0 - norms)
            active_bound = bound_violation > 0
            # Direction: -p_i / |p_i|
            # If norm is 0, direction is 0.
            dir_bound = np.zeros_like(pts)
            mask_bound = norms > 1e-12
            dir_bound[mask_bound] = -pts[mask_bound] / norms[mask_bound, np.newaxis]
            
            forces[active_bound] += bound_violation[active_bound, np.newaxis] * dir_bound[active_bound]
            
            # Update positions
            pts += lr * forces
            
            # Clip to avoid running away
            # It's better to project back to unit disk if they go outside
            new_norms = np.hypot(pts[:, 0], pts[:, 1])
            outside = new_norms > 1.0
            if np.any(outside):
                pts[outside] = pts[outside] / new_norms[outside, np.newaxis] * 0.999

    return pts


def construct():
    t0 = time.time()
    n = N
    
    best_pts = None
    best_r = 0.0
    
    rng = np.random.RandomState(42)
    
    # 1. Hex grid
    pts0 = _hex_grid_init(n)
    r0 = _score(pts0)
    if r0 > best_r:
        best_r = r0
        best_pts = pts0.copy()
        
    # Relax hex grid
    if time.time() - t0 < _DEADLINE - 5:
        pts_relaxed = _relax(pts0, max_iter=100)
        r_relaxed = _score(pts_relaxed)
        if r_relaxed > best_r:
            best_r = r_relaxed
            best_pts = pts_relaxed.copy()
            
    # 2. Random starts
    for i in range(10):
        if time.time() - t0 > _DEADLINE - 10:
            break
            
        # Random points inside unit disk
        # Use rejection sampling or polar coords
        angles = rng.uniform(0, 2*np.pi, n)
        radii = np.sqrt(rng.uniform(0, 1, n))
        pts_rand = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        r_rand = _score(pts_rand)
        
        # Relax
        pts_relaxed = _relax(pts_rand, max_iter=50)
        r_relaxed = _score(pts_relaxed)
        
        if r_relaxed > best_r:
            best_r = r_relaxed
            best_pts = pts_relaxed.copy()
            
    # 3. Perturbed hex grid
    if time.time() - t0 < _DEADLINE - 5:
        for i in range(5):
            if time.time() - t0 > _DEADLINE - 5:
                break
            pts_pert = pts0 + rng.randn(n, 2) * 0.05
            # Project to unit disk
            norms = np.hypot(pts_pert[:, 0], pts_pert[:, 1])
            outside = norms > 1.0
            if np.any(outside):
                pts_pert[outside] = pts_pert[outside] / norms[outside, np.newaxis] * 0.999
                
            pts_relaxed = _relax(pts_pert, max_iter=100)
            r_relaxed = _score(pts_relaxed)
            
            if r_relaxed > best_r:
                best_r = r_relaxed
                best_pts = pts_relaxed.copy()
                
    # Final check
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_grid_init(n)
        
    return best_pts
