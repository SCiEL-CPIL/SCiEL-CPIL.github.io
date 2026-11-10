"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import time

DEADLINE = 115.0  # safety margin before 120s limit


def _score(points):
    """Compute the packing score: min(min pairwise dist/2, min boundary margin)."""
    if len(points) != N:
        return -1.0
    # Boundary check
    dists_center = np.linalg.norm(points, axis=1)
    boundary_margin = 1.0 - dists_center
    if np.any(boundary_margin < 0):
        return -1.0
    # Pairwise distances
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    r = min(min_pair / 2.0, np.min(boundary_margin))
    return r


def _gen_hex_lattice():
    """Generate a hexagonal lattice of points inside the unit disk, return N best."""
    # Try multiple spacings to find one that gives at least N points
    best_points = None
    for spacing in np.linspace(0.04, 0.12, 80):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            # Pick the N closest to center for better symmetry
            idx = np.argsort(radius, kind="stable")[:N]
            pts = points[idx]
            if best_points is None or _score(pts) > _score(best_points):
                best_points = pts.copy()
    return best_points


def _gen_ring_structure():
    """Generate a ring-based structure: center + concentric rings."""
    # 1 center + rings with increasing numbers of points
    # Ring i has n_i points, radius r_i
    # We want sum(n_i) + 1 = 300
    # Try various ring configurations
    best_points = None
    best_score = -1
    
    # Configuration: try to use rings with 6, 12, 18, 24, ... points
    # But adjust to hit exactly 300
    ring_counts = []
    remaining = N - 1  # one center point
    ring_idx = 0
    while remaining > 0:
        # Use 6 * (ring_idx + 1) points, but cap to not exceed remaining
        count = min(6 * (ring_idx + 1), remaining)
        ring_counts.append(count)
        remaining -= count
        ring_idx += 1
    
    # Total rings
    n_rings = len(ring_counts)
    
    # Radii: roughly evenly spaced
    radii = np.linspace(0.15, 0.95, n_rings)
    
    points = [np.array([0.0, 0.0])]  # center
    for r, count in zip(radii, ring_counts):
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        points.append(pts)
    
    points = np.vstack(points)
    if len(points) == N:
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    return best_points


def _local_optimize(points, max_iters=500, tol=1e-10):
    """Local optimization using gradient-free methods to maximize the min radius."""
    best_points = points.copy()
    best_score = _score(points)
    
    # Use Nelder-Mead style or simple iterative pairwise repulsion
    # Simple approach: iteratively push overlapping pairs apart, keep inside circle
    lr = 0.01
    for it in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        
        # Compute forces
        d = cdist(points, points)
        np.fill_diagonal(d, np.inf)
        
        # For each pair closer than 2*r, push apart
        # Use current score as target separation
        r_current = best_score
        target_sep = 2.0 * r_current + 0.001
        
        forces = np.zeros_like(points)
        for i in range(N):
            for j in range(i + 1, N):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < target_sep and dist > 1e-10:
                    # Push apart
                    f = (target_sep - dist) / dist * diff
                    forces[i] += f
                    forces[j] -= f
        
        # Boundary force: push inward if too close to boundary
        dists_center = np.linalg.norm(points, axis=1)
        boundary_r = 1.0 - r_current
        for i in range(N):
            if dists_center[i] > boundary_r:
                f = (dists_center[i] - boundary_r) * points[i] / (dists_center[i] + 1e-10)
                forces[i] -= f
        
        # Update
        points = points + lr * forces
        lr *= 0.999
        
        # Check score
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        
        if lr < 1e-12:
            break
    
    return best_points, best_score


def _refine_with_nm(points, max_time=20):
    """Use Nelder-Mead on a subset or full problem if time permits."""
    if time.time() > DEADLINE - 10:
        return points
    
    start_time = time.time()
    best_points = points.copy()
    best_score = _score(points)
    
    # Flatten
    x0 = points.flatten()
    
    def objective(x):
        if time.time() - start_time > max_time:
            return -best_score
        pts = x.reshape(N, 2)
        s = _score(pts)
        if s < 0:
            return -s  # Penalize invalid
        return -s
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10,
                                'adaptive': True})
        if time.time() < DEADLINE - 5:
            pts = res.x.reshape(N, 2)
            s = _score(pts)
            if s > best_score:
                best_points = pts.copy()
                best_score = s
    except Exception:
        pass
    
    return best_points


def construct():
    global DEADLINE
    start_time = time.time()
    DEADLINE = start_time + 115.0  # 150s limit, keep 35s margin
    
    # Generate initial candidates
    candidates = []
    
    # Hex lattice
    hex_pts = _gen_hex_lattice()
    if hex_pts is not None:
        candidates.append(hex_pts)
    
    # Ring structure
    ring_pts = _gen_ring_structure()
    if ring_pts is not None:
        candidates.append(ring_pts)
    
    # Pick best initial
    best_pts = max(candidates, key=lambda p: _score(p))
    best_score = _score(best_pts)
    
    # Local optimization
    if time.time() < DEADLINE - 30:
        best_pts, best_score = _local_optimize(best_pts, max_iters=200)
    
    # Try Nelder-Mead refinement if time permits
    if time.time() < DEADLINE - 15:
        best_pts = _refine_with_nm(best_pts, max_time=20)
        best_score = _score(best_pts)
    
    # Final safety check
    if time.time() > DEADLINE - 2:
        # Return whatever we have
        pass
    
    # Ensure validity
    final_score = _score(best_pts)
    if final_score < 0:
        # Fallback to hex lattice
        fallback = _gen_hex_lattice()
        if fallback is not None:
            return fallback
        # Last resort: simple grid
        spacing = 0.055
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
    
    return best_pts
# EVOLVE-BLOCK-END
