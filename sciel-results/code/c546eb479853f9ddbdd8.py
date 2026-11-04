"""Pack 300 equal circles in a unit circle using a hexagonal-lattice start + spatial-hash force relaxation."""
import time
import numpy as np

N = 300
DEADLINE = 130.0  # seconds; leave margin before the 150 s kill


def _hex_start(n):
    """Return n points on a hexagonal lattice clipped to the unit disk (centred at origin)."""
    # Target density: area per circle ~ pi/r^2. For N circles in unit disk, r ~ 0.053.
    # Hexagonal spacing s = 2r gives packing fraction ~ 0.907.
    # We'll build a lattice with spacing s and clip to radius 1 - s/2.
    s = 2.0 / (np.sqrt(n) * 0.95)  # slight over-estimate to ensure enough points
    # Build grid
    k = int(1.0 / s) + 3
    pts = []
    for j in range(-k, k + 1):
        y = j * s * np.sqrt(3.0) / 2.0
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j & 1)) * s
            if x * x + y * y <= (1.0 - s / 2.0) ** 2:
                pts.append((x, y))
    pts = np.array(pts, dtype=np.float64)
    if len(pts) >= n:
        # Pick the n closest to origin for a compact start
        r2 = np.sum(pts ** 2, axis=1)
        idx = np.argsort(r2, kind="stable")[:n]
        return pts[idx]
    # Fallback: shrink spacing
    while len(pts) < n:
        s *= 0.97
        k = int(1.0 / s) + 3
        pts = []
        for j in range(-k, k + 1):
            y = j * s * np.sqrt(3.0) / 2.0
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j & 1)) * s
                if x * x + y * y <= (1.0 - s / 2.0) ** 2:
                    pts.append((x, y))
        pts = np.array(pts, dtype=np.float64)
        r2 = np.sum(pts ** 2, axis=1)
        idx = np.argsort(r2, kind="stable")[:n]
        return pts[idx]
    return pts[:n]


def _score(pts):
    """Compute the packing radius for a configuration."""
    # Min distance between any pair / 2
    # Use scipy's cdist if available, else numpy
    try:
        from scipy.spatial.distance import cdist
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d)
    except Exception:
        # Fallback: O(N^2)
        min_pair = np.inf
        for i in range(len(pts)):
            diff = pts[i + 1:] - pts[i]
            d = np.sqrt(np.sum(diff ** 2, axis=1))
            if len(d) > 0:
                min_pair = min(min_pair, np.min(d))
    # Min distance to boundary
    r_norm = np.sqrt(np.sum(pts ** 2, axis=1))
    min_boundary = np.min(1.0 - r_norm)
    return 0.5 * min(min_pair, 2.0 * min_boundary)


def _relax(pts, max_steps=50000, dt_initial=0.01, dt_min=1e-6, cooldown=0.9999):
    """
    Force-directed relaxation with spatial hashing for neighbor search.
    Forces: repulsion between close circles, attraction to fill space, boundary constraint.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    dt = dt_initial
    best_score = _score(pts)
    best_pts = pts.copy()
    
    # Cell size for spatial hash: use initial spacing estimate
    s = 2.0 / (np.sqrt(n) * 0.95)
    cell_size = s * 1.5
    
    t0 = time.time()
    
    for step in range(max_steps):
        if time.time() - t0 > DEADLINE - 10:
            break
        
        # Build spatial hash
        cell_map = {}
        for i in range(n):
            cx = int(pts[i, 0] / cell_size)
            cy = int(pts[i, 1] / cell_size)
            key = (cx, cy)
            if key not in cell_map:
                cell_map[key] = []
            cell_map[key].append(i)
        
        # Compute forces
        forces = np.zeros((n, 2), dtype=np.float64)
        
        for i in range(n):
            cx = int(pts[i, 0] / cell_size)
            cy = int(pts[i, 1] / cell_size)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    key = (cx + dx, cy + dy)
                    if key not in cell_map:
                        continue
                    for j in cell_map[key]:
                        if j <= i:
                            continue
                        diff = pts[i] - pts[j]
                        dist = np.sqrt(np.sum(diff ** 2))
                        if dist < 1e-12:
                            # Random nudge for coincident points
                            diff = np.random.randn(2) * 1e-6
                            dist = np.sqrt(np.sum(diff ** 2))
                        # Repulsive force: strong when close
                        if dist < s * 2.0:
                            f = 1.0 / (dist ** 2 + 1e-8)
                            forces[i] += f * diff / dist
                            forces[j] -= f * diff / dist
        
        # Boundary force: push points inward if too close to boundary
        r_norm = np.sqrt(np.sum(pts ** 2, axis=1, keepdims=True))
        r_norm_safe = np.where(r_norm < 1e-12, 1e-12, r_norm)
        margin = s * 0.5
        for i in range(n):
            if r_norm[i, 0] > 1.0 - margin:
                # Push toward center
                push = (r_norm[i, 0] - (1.0 - margin)) * 5.0
                forces[i] -= push * pts[i] / r_norm_safe[i, 0]
        
        # Apply forces with damping
        pts += dt * forces
        
        # Clamp to unit disk
        r_norm = np.sqrt(np.sum(pts ** 2, axis=1, keepdims=True))
        r_norm_safe = np.where(r_norm < 1e-12, 1e-12, r_norm)
        too_far = r_norm > (1.0 - 1e-12)
        if np.any(too_far):
            pts[too_far] *= (1.0 - 1e-12) / r_norm_safe[too_far, 0]
        
        # Decay step size
        dt *= cooldown
        
        # Check score periodically
        if step % 500 == 0:
            sc = _score(pts)
            if sc > best_score:
                best_score = sc
                best_pts = pts.copy()
        
        # Early stop if no improvement for a while and dt is small
        if dt < dt_min and step > 10000:
            break
    
    # Final check
    sc = _score(pts)
    if sc > best_score:
        best_score = sc
        best_pts = pts.copy()
    
    return best_pts, best_score


def construct():
    t0 = time.time()
    
    # Start with hexagonal lattice
    pts = _hex_start(N)
    
    # Run relaxation
    pts, score = _relax(pts)
    
    # If we have time, try a few more rounds with smaller dt
    if time.time() - t0 < DEADLINE - 20:
        pts, score2 = _relax(pts, max_steps=30000, dt_initial=0.005, cooldown=0.99995)
        if score2 > score:
            pts, score = pts, score2
    
    # Final safety: ensure all points are strictly inside
    r_norm = np.sqrt(np.sum(pts ** 2, axis=1))
    mask = r_norm < 1.0
    if not np.all(mask):
        pts = pts[mask]
        if len(pts) < N:
            # Pad with points near origin if needed (shouldn't happen)
            while len(pts) < N:
                pts = np.vstack([pts, np.array([0.0, 0.0])])
    
    return pts[:N]
