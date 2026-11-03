"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _ring_init(n):
    """Initialize n points in concentric rings inside unit circle."""
    if n == 1:
        return np.array([[0.0, 0.0]])
    
    points = []
    # Try different ring configurations
    # Ring 0: center
    # Ring k: k*6 points (hexagonal)
    
    best_points = None
    best_score = -1
    
    # Strategy: place points in rings
    # Ring 0: 1 point at center
    # Ring 1: 6 points
    # Ring 2: 12 points
    # Ring 3: 18 points
    # etc.
    
    # But we need exactly n points, so we may need to adjust
    # Let's try multiple ring spacing parameters
    
    for ring_spacing in np.linspace(0.15, 0.45, 30):
        pts = []
        ring = 0
        while len(pts) < n:
            if ring == 0:
                pts.append([0.0, 0.0])
            else:
                n_in_ring = 6 * ring
                # But we might not need all
                remaining = n - len(pts)
                n_in_ring = min(n_in_ring, remaining)
                if n_in_ring == 0:
                    break
                angle_offset = (ring % 2) * np.pi / (6 * ring) if ring > 0 else 0
                for i in range(n_in_ring):
                    angle = 2 * np.pi * i / n_in_ring + angle_offset
                    r = ring * ring_spacing
                    pts.append([r * np.cos(angle), r * np.sin(angle)])
            ring += 1
        
        if len(pts) < n:
            continue
        
        pts = np.array(pts[:n])
        
        # Check if all inside unit circle
        if np.any(np.hypot(pts[:, 0], pts[:, 1]) > 1.0):
            continue
        
        # Compute score
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    if best_points is None:
        # Fallback: hexagonal lattice
        return _hex_init(n)
    
    return best_points


def _hex_init(n):
    """Hexagonal lattice initialization."""
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
            return points[np.argsort(radius, kind="stable")[:n]]
        spacing *= 0.98


def _compute_score(pts):
    """Compute the packing radius (minimum of half-pairwise-distances and 1-norm)."""
    n = len(pts)
    if n < 2:
        return 1.0 - np.linalg.norm(pts[0])
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundary
    norms = np.linalg.norm(pts, axis=1)
    min_boundary = 1.0 - np.max(norms)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _push_optimize(pts, max_iter=5000, tol=1e-10):
    """Iterative push optimization: move each point away from its most constraining neighbor."""
    n = len(pts)
    pts = pts.copy()
    
    for it in range(max_iter):
        improved = False
        for i in range(n):
            # Find the most constraining "neighbor" for point i
            # This could be another point or the boundary
            
            # Distance to boundary
            norm_i = np.linalg.norm(pts[i])
            boundary_constraint = 1.0 - norm_i
            
            # Find closest other point
            diff = pts - pts[i]
            dists = np.sqrt(np.sum(diff**2, axis=1))
            dists[i] = np.inf
            j = np.argmin(dists)
            pair_constraint = dists[j] / 2.0
            
            # The bottleneck for point i
            if boundary_constraint <= pair_constraint:
                # Move away from center (toward boundary is wrong, we need to be inside)
                # Actually, if boundary is the constraint, we're too close to boundary
                # Move toward center
                if norm_i > 1e-12:
                    direction = -pts[i] / norm_i
                else:
                    direction = np.array([1.0, 0.0])
                # Move by a small amount
                step = boundary_constraint * 0.5
                pts[i] += direction * step
            else:
                # Move away from point j
                direction = pts[i] - pts[j]
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-12:
                    direction = direction / norm_dir
                else:
                    direction = np.array([1.0, 0.0])
                step = pair_constraint * 0.5
                pts[i] += direction * step
            
            # Ensure we don't go outside
            new_norm = np.linalg.norm(pts[i])
            if new_norm > 1.0:
                pts[i] = pts[i] / new_norm * 0.999
            
            improved = True
        
        if not improved:
            break
    
    return pts


def _soft_min_obj(pts, beta=50.0):
    """Soft-min approximation of the packing radius using log-sum-exp."""
    n = len(pts)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Boundary distances
    norms_sq = np.sum(pts**2, axis=1)
    boundary_dists = 1.0 - np.sqrt(norms_sq)
    
    # We want min of: dists[i,j]/2 for i<j, and (1 - |c_i|)
    # Soft min: -1/beta * log(sum(exp(-beta * x)))
    
    # Pairwise soft min
    upper_tri = np.triu_indices(n, k=1)
    pair_dists = np.sqrt(dists_sq[upper_tri]) / 2.0
    
    # Combine all constraints
    all_constraints = np.concatenate([pair_dists, boundary_dists])
    
    # Soft min
    min_val = np.min(all_constraints)
    exp_vals = np.exp(-beta * (all_constraints - min_val))
    soft_min = min_val - np.log(np.sum(exp_vals)) / beta
    
    return soft_min


def _gradient_ascent(pts, beta=50.0, lr=0.01, max_iter=2000):
    """Gradient ascent on soft-min objective."""
    n = len(pts)
    pts = pts.copy()
    
    for it in range(max_iter):
        # Compute gradient numerically
        grad = np.zeros_like(pts)
        eps = 1e-6
        
        for i in range(n):
            for d in range(2):
                pts_plus = pts.copy()
                pts_plus[i, d] += eps
                obj_plus = _soft_min_obj(pts_plus, beta)
                
                pts_minus = pts.copy()
                pts_minus[i, d] -= eps
                obj_minus = _soft_min_obj(pts_minus, beta)
                
                grad[i, d] = (obj_plus - obj_minus) / (2 * eps)
        
        # Update
        pts += lr * grad
        
        # Project back inside unit circle
        norms = np.linalg.norm(pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / norms[mask][:, np.newaxis] * 0.999
        
        # Decay learning rate
        lr *= 0.999
    
    return pts


def _polish(pts, max_iter=10000, tol=1e-12):
    """High-precision polish using direct constraint solving."""
    n = len(pts)
    pts = pts.copy()
    
    for it in range(max_iter):
        # Compute current score
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        norms = np.linalg.norm(pts, axis=1)
        boundary_dists = 1.0 - norms
        
        min_pair = np.min(dists) / 2.0
        min_boundary = np.min(boundary_dists)
        current_score = min(min_pair, min_boundary)
        
        # Find which constraint is active
        if min_pair <= min_boundary:
            # Find the pair achieving min
            i, j = np.unravel_index(np.argmin(dists), dists.shape)
            if i > j:
                i, j = j, i
            # Move i and j apart
            direction = pts[i] - pts[j]
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-15:
                direction = direction / norm_dir
            else:
                direction = np.array([1.0, 0.0])
            
            step = (min_pair - current_score) * 0.5 + 1e-8
            pts[i] += direction * step
            pts[j] -= direction * step
        else:
            # Find the point closest to boundary
            i = np.argmin(boundary_dists)
            # Move it toward center
            norm_i = np.linalg.norm(pts[i])
            if norm_i > 1e-15:
                direction = -pts[i] / norm_i
            else:
                direction = np.array([1.0, 0.0])
            step = (min_boundary - current_score) * 0.5 + 1e-8
            pts[i] += direction * step
        
        # Project inside
        norms = np.linalg.norm(pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / norms[mask][:, np.newaxis] * 0.999
        
        # Check convergence
        if it > 0 and abs(_compute_score(pts) - current_score) < tol:
            break
    
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Step 1: Initialize with ring structure
    pts = _ring_init(N)
    
    # Step 2: Push optimization
    pts = _push_optimize(pts, max_iter=2000)
    
    # Check time
    if time.time() > deadline - 30:
        return _polish(pts, max_iter=1000)
    
    # Step 3: Gradient ascent on soft-min
    pts = _gradient_ascent(pts, beta=50.0, lr=0.005, max_iter=500)
    
    # Check time
    if time.time() > deadline - 20:
        return _polish(pts, max_iter=500)
    
    # Step 4: Polish
    pts = _polish(pts, max_iter=5000)
    
    # Final check
    if time.time() > deadline - 5:
        return pts
    
    # One more polish if time allows
    pts = _polish(pts, max_iter=2000)
    
    return pts

# EVOLVE-BLOCK-END
