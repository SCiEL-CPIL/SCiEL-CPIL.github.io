"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Best known packing radius for 50 circles in a unit circle (Graham et al. 1998)
KNOWN_BEST_R = 0.1258254895304


def _score_of(pts):
    """Compute the objective: min of pairwise half-distances and boundary margins."""
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    # Set diagonal to inf
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    # Boundary margins
    radii = np.sqrt(np.sum(pts * pts, axis=1))
    min_bound = 1.0 - np.max(radii)
    return min(min_pair, min_bound)


def _initial_hexagonal(n):
    """Generate n points via hexagonal lattice within unit disk."""
    # Start with a fine hex grid
    spacing = 0.15
    points = []
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0 - spacing / 2.0:
                points.append([x, y])
    points = np.array(points)
    # Sort by distance from center and take first n
    dists = np.sqrt(np.sum(points * points, axis=1))
    idx = np.argsort(dists)
    return points[idx[:n]].copy()


def _initial_ring(n):
    """Generate n points in concentric rings."""
    # Approximate: rings with ~k points
    # For 50 points: center + rings of 6, 12, 18, 14 (total 51, take 50)
    rings = []
    remaining = n
    # Center
    rings.append([0.0, 0.0])
    remaining -= 1
    ring_idx = 1
    while remaining > 0:
        # Number of points in this ring
        k = min(6 * ring_idx, remaining)
        r = 0.3 * ring_idx  # approximate radius
        angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
        for a in angles:
            rings.append([r * np.cos(a), r * np.sin(a)])
        remaining -= k
        ring_idx += 1
    return np.array(rings[:n])


def _optimize(pts, max_iter=2000, tol=1e-10):
    """Optimize point positions to maximize the minimum clearance."""
    n = len(pts)
    pts = pts.astype(np.float64).copy()
    
    # We'll use a penalty-based approach with scipy minimize
    # Objective: minimize -min_clearance + penalties for violations
    
    def objective(x):
        p = x.reshape(n, 2)
        # Compute pairwise distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist) / 2.0
        
        # Boundary
        radii = np.sqrt(np.sum(p * p, axis=1))
        min_bound = 1.0 - np.max(radii)
        
        # The actual score is min(min_pair, min_bound)
        # We want to maximize this
        # Use a smooth approximation: if both positive, it's min
        # Penalty if either is negative
        score = min(min_pair, min_bound)
        
        # Add penalties for any violations
        # Pairwise: want all dist >= 2*r where r is current min
        # This is tricky; instead, use a barrier-like approach
        
        # Simpler: just return -score and let gradient handle it
        # But min is non-differentiable; use a soft-min
        # soft_min(a, b) = -logsumexp(-[a,b]) / 1  (for small scale)
        
        # Actually, let's just use the raw min and hope the optimizer handles it
        # Or use a differentiable approximation
        eps = 1e-6
        soft_min = -(np.log(np.exp(-min_pair / eps) + np.exp(-min_bound / eps))) * eps
        
        # Penalties for overlap and boundary
        # For pairwise: sum of max(0, 2*r_target - dist)^2
        # But we don't know r_target a priori
        
        # Alternative: maximize the minimum directly using a different formulation
        # Let's use the approach of minimizing the maximum violation
        
        # Violations: for each pair, violation = max(0, 2*r - dist_ij) where r is some target
        # This is circular. 
        
        # Better approach: use a sequential method
        # For now, return -soft_min as the objective to minimize
        return -soft_min
    
    # Gradient: hard to compute analytically for min, so use numerical
    # But that's slow. Let's try a different approach.
    
    # Alternative: use the fact that at optimum, the constraint is active
    # Let's use a simpler iterative approach: 
    # 1. Compute current min clearance r
    # 2. For each pair with dist < 2*r + eps, push apart
    # 3. For points near boundary, push inward
    # This is a simple relaxation method
    
    return pts


def _relax(pts, max_iter=1000):
    """Simple relaxation: push overlapping pairs apart and boundary-violating points inward."""
    n = len(pts)
    pts = pts.astype(np.float64).copy()
    
    for it in range(max_iter):
        # Compute current min clearance
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dist, np.inf)
        
        radii = np.sqrt(np.sum(pts * pts, axis=1))
        
        min_pair = np.min(dist) / 2.0
        min_bound = 1.0 - np.max(radii)
        r = min(min_pair, min_bound)
        
        if r <= 0:
            # Need to fix violations
            pass
        
        # Push apart overlapping pairs
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = dist[i, j]
                if d < 2 * r and d > 1e-15:
                    # Push apart
                    direction = (pts[i] - pts[j]) / d
                    push = (2 * r - d) / 2.0
                    pts[i] += push * direction
                    pts[j] -= push * direction
                    moved = True
                elif d <= 1e-15:
                    # Coincident, push randomly
                    angle = np.random.uniform(0, 2 * np.pi)
                    pts[i] += r * 0.1 * np.array([np.cos(angle), np.sin(angle)])
                    pts[j] -= r * 0.1 * np.array([np.cos(angle), np.sin(angle)])
                    moved = True
        
        # Push boundary-violating points inward
        for i in range(n):
            rad = radii[i]
            if rad > 1.0 - r and rad > 1e-15:
                direction = -pts[i] / rad
                push = rad - (1.0 - r)
                pts[i] += push * direction
                moved = True
        
        if not moved:
            break
    
    return pts


def _multi_start(n, n_starts=5, time_limit=120.0):
    """Try multiple initial configurations and relaxations."""
    start_time = time.time()
    best_pts = None
    best_score = -np.inf
    
    # Different initializations
    inits = []
    
    # 1. Hexagonal
    inits.append(_initial_hexagonal(n))
    
    # 2. Ring
    inits.append(_initial_ring(n))
    
    # 3. Random
    for _ in range(3):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.random.uniform(-0.1, 0.1, n)
        radii = np.sqrt(np.random.uniform(0, 1, n))
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(pts)
    
    for init_pts in inits:
        if time.time() - start_time > time_limit - 10:
            break
        
        # Relax
        pts = _relax(init_pts, max_iter=500)
        score = _score_of(pts)
        
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # Multi-start optimization
    pts, score = _multi_start(N, n_starts=5, time_limit=time_limit)
    
    if pts is None:
        # Fallback: return hexagonal
        pts = _initial_hexagonal(N)
    
    # Final check: ensure all points are inside and no overlaps
    # If any point is outside, clamp
    radii = np.sqrt(np.sum(pts * pts, axis=1))
    for i in range(len(pts)):
        if radii[i] > 1.0:
            pts[i] = pts[i] / radii[i] * 0.99
    
    return pts
# EVOLVE-BLOCK-END
