"""Packing 50 equal circles in a unit circle - force-based optimization."""
import time
import numpy as np

N = 50
_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 0.0
    if N < 2:
        return r_cont
    # Pairwise distances
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dmat = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(dmat, np.inf)
    min_pair = np.min(dmat)
    r_pair = min_pair / 2.0
    return min(r_cont, r_pair)


def _hex_lattice_init(n):
    """Generate points on a hexagonal lattice inside the unit disk, pick best n."""
    best_pts = None
    best_score = -1.0
    for spacing in np.linspace(0.15, 0.35, 40):
        k = int(1.0 / spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x, y, r))
        if len(rows) < n:
            continue
        rows.sort(key=lambda t: t[2])
        pts = np.array([(x, y) for x, y, _ in rows[:n]])
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    if best_pts is None:
        # Fallback: random
        best_pts = np.random.randn(n, 2) * 0.8
    return best_pts


def _force_optimize(pts, max_iters=5000):
    """Force-based optimization to improve the packing."""
    pts = pts.astype(np.float64).copy()
    n = len(pts)
    
    # Compute initial score
    best_score = _score(pts)
    best_pts = pts.copy()
    
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        
        # Compute pairwise distances
        dx = pts[:, 0:1] - pts[0:1, 0]
        dy = pts[:, 1:2] - pts[0:1, 1]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        np.fill_diagonal(dist, np.inf)
        
        # Find minimum distance
        min_dist = np.min(dist)
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Repulsion between circles
        for i in range(n):
            for j in range(i + 1, n):
                d = dist[i, j]
                if d < min_dist * 1.1:  # Only interact if close
                    # Strong repulsion
                    force_mag = 10.0 / (d + 1e-10)
                    fx = (pts[i, 0] - pts[j, 0]) / (d + 1e-10) * force_mag
                    fy = (pts[i, 1] - pts[j, 1]) / (d + 1e-10) * force_mag
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Attraction to boundary (push circles outward)
        for i in range(n):
            r = np.hypot(pts[i, 0], pts[i, 1])
            if r > 1e-10:
                # Push toward boundary
                force_mag = 0.5 * (1.0 - r)
                forces[i, 0] += pts[i, 0] / r * force_mag
                forces[i, 1] += pts[i, 1] / r * force_mag
        
        # Update positions with adaptive step size
        step_size = 0.01 * (1.0 - it / max_iters) + 0.001
        pts += step_size * forces
        
        # Keep points inside container
        r = np.hypot(pts[:, 0], pts[:, 1])
        mask = r > 1.0
        if np.any(mask):
            pts[mask] *= (1.0 - 1e-6) / r[mask, None]
        
        # Compute new score
        new_score = _score(pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = pts.copy()
    
    return best_pts, best_score


def construct():
    """Main entry point: construct the packing."""
    # Start from hexagonal lattice
    pts = _hex_lattice_init(N)
    
    # Run force-based optimization
    pts, score = _force_optimize(pts)
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        return pts
    
    # Try a few more optimization passes if time permits
    for _ in range(3):
        if time.time() > _DEADLINE - 10.0:
            break
        pts, new_score = _force_optimize(pts, max_iters=2000)
        if new_score <= score:
            break
        score = new_score
    
    return pts
