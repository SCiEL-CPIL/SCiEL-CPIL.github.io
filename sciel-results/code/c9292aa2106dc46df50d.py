"""Optimized packing of 300 equal circles in a unit circle."""
import numpy as np
import time

N = 300
DEADLINE_MARGIN = 5.0  # seconds before the 150s limit to stop
START_TIME = time.time()


def _time_left():
    """Return remaining seconds before deadline."""
    return 150.0 - (time.time() - START_TIME) - DEADLINE_MARGIN


def _compute_radius(pts):
    """Compute the maximum common radius for given centers."""
    # Distance from each center to origin
    dists_to_origin = np.linalg.norm(pts, axis=1)
    # Constraint: 1 - dist_to_origin
    r_container = 1.0 - dists_to_origin
    # Pairwise distances / 2
    # Use efficient computation: |pi - pj| = sqrt((pi-pj)^2)
    # For 300 points, this is 300x300 = 90000 pairs, manageable
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    r_pairs = dists / 2.0
    r = min(np.min(r_container), np.min(r_pairs))
    return r


def _soft_min_logsumexp(x, beta):
    """Soft-min approximation using log-sum-exp."""
    # -log(sum(exp(-beta * x))) / beta
    # For numerical stability:
    min_x = np.min(x)
    return -np.log(np.sum(np.exp(-beta * (x - min_x)))) / beta - min_x


def _negative_soft_radius(pts, beta):
    """Negative soft radius for minimization."""
    dists_to_origin = np.linalg.norm(pts, axis=1)
    r_container = 1.0 - dists_to_origin
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    r_pairs = dists / 2.0
    
    # Combine all constraints
    all_constraints = np.concatenate([r_container, r_pairs[r_pairs < np.inf].ravel()])
    return -_soft_min_logsumexp(all_constraints, beta)


def _hex_init():
    """Generate initial hexagonal lattice packing."""
    # Start with a hexagonal lattice
    spacing = 0.1
    best_pts = None
    best_r = 0
    
    for attempt in range(20):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        
        if len(points) >= N:
            # Select N points closest to center
            points = points[np.argsort(radius, kind="stable")[:N]]
            r = _compute_radius(points)
            if r > best_r:
                best_r = r
                best_pts = points.copy()
        
        spacing *= 0.98
    
    if best_pts is None:
        # Fallback: random points
        best_pts = np.random.randn(N, 2) * 0.9
        best_r = _compute_radius(best_pts)
    
    return best_pts, best_r


def _gradient_ascent(pts, beta, lr, steps):
    """Perform gradient ascent on soft radius."""
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    for step in range(steps):
        if _time_left() < 10:
            break
        
        # Compute gradient numerically
        grad = np.zeros_like(pts)
        eps = 1e-5
        base_val = _negative_soft_radius(pts, beta)
        
        for i in range(N):
            for d in range(2):
                orig = pts[i, d]
                pts[i, d] = orig + eps
                val_plus = _negative_soft_radius(pts, beta)
                pts[i, d] = orig - eps
                val_minus = _negative_soft_radius(pts, beta)
                pts[i, d] = orig
                grad[i, d] = (val_plus - val_minus) / (2 * eps)
        
        # Gradient ascent (we're minimizing negative soft radius)
        pts = pts - lr * grad
        
        # Project back to valid region (keep inside unit circle)
        dists = np.linalg.norm(pts, axis=1)
        mask = dists > 0.999
        if np.any(mask):
            pts[mask] = pts[mask] * (0.999 / dists[mask])
        
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    return best_pts, best_r


def _polish(pts, iterations=1000):
    """High-precision polish using very sharp soft-min."""
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    # Use progressively sharper beta
    for beta in [100, 500, 1000, 5000, 10000]:
        if _time_left() < 20:
            break
        
        lr = 1e-4 / beta
        for it in range(iterations):
            if _time_left() < 10:
                break
            
            # Compute gradient numerically (batched for speed)
            grad = np.zeros_like(pts)
            eps = 1e-6
            base_val = _negative_soft_radius(pts, beta)
            
            for i in range(N):
                for d in range(2):
                    orig = pts[i, d]
                    pts[i, d] = orig + eps
                    val_plus = _negative_soft_radius(pts, beta)
                    pts[i, d] = orig - eps
                    val_minus = _negative_soft_radius(pts, beta)
                    pts[i, d] = orig
                    grad[i, d] = (val_plus - val_minus) / (2 * eps)
            
            # Gradient ascent
            pts = pts - lr * grad
            
            # Project back to valid region
            dists = np.linalg.norm(pts, axis=1)
            mask = dists > 0.9999
            if np.any(mask):
                pts[mask] = pts[mask] * (0.9999 / dists[mask])
            
            r = _compute_radius(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
    
    return best_pts, best_r


def construct():
    """Construct centers of N equal circles inside the unit circle."""
    global START_TIME
    START_TIME = time.time()
    
    # Phase 1: Initialize with hexagonal lattice
    pts, r = _hex_init()
    
    # Phase 2: Gradient ascent with moderate beta
    if _time_left() > 60:
        for beta in [10, 50, 100]:
            if _time_left() < 30:
                break
            pts, r = _gradient_ascent(pts, beta, lr=1e-3 / beta, steps=200)
    
    # Phase 3: Polish with sharp soft-min
    if _time_left() > 30:
        pts, r = _polish(pts, iterations=500)
    
    # Final check: ensure validity
    dists_to_origin = np.linalg.norm(pts, axis=1)
    if np.any(dists_to_origin >= 1.0):
        # Push points inside
        pts = pts * (0.999 / np.maximum(dists_to_origin, 1e-10))
    
    return pts
