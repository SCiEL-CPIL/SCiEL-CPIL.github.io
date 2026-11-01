"""Circle packing: 50 equal circles in unit disk. Optimize via Nelder-Mead + force relaxation."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def _compute_score(pts):
    """Return the packing radius (min of pairwise/2 and boundary distances)."""
    n = len(pts)
    if n < 2:
        return 0.0
    # boundary distances
    boundary = 1.0 - np.linalg.norm(pts, axis=1)
    if np.any(boundary < -1e-12):
        return 0.0  # invalid: outside container
    # pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_pair, np.min(boundary))

def _objective(x, n):
    """Objective to minimize: negative of the packing radius, with penalty for invalid configs."""
    pts = x.reshape(n, 2)
    # boundary
    boundary = 1.0 - np.linalg.norm(pts, axis=1)
    # pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # penalty for going outside or overlapping
    penalty = 0.0
    if np.any(boundary < 0):
        penalty += 10.0 * np.sum(np.clip(-boundary, 0, None))
    if min_pair < 0:
        penalty += 10.0 * (-min_pair)
    r = min(min_pair, np.min(boundary))
    return -r + penalty

def _hex_lattice_init(n):
    """Generate a hexagonal lattice point cloud and pick n points closest to center."""
    best_pts = None
    for spacing in np.linspace(0.12, 0.25, 40):
        k = int(2.0 / spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        pts = np.array(rows)
        if len(pts) >= n:
            r = np.linalg.norm(pts, axis=1)
            idx = np.argsort(r, kind="stable")[:n]
            best_pts = pts[idx]
            break
    if best_pts is None:
        # fallback: random
        best_pts = _random_init(n)
    return best_pts

def _random_init(n):
    """Random points in unit disk."""
    theta = np.random.uniform(0, 2 * np.pi, n)
    rho = np.sqrt(np.random.uniform(0, 1, n))
    return np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])

def _nelder_mead_optimize(pts, n, max_time):
    """Optimize using Nelder-Mead. Returns improved points."""
    x0 = pts.flatten()
    best_x = x0.copy()
    best_f = _objective(x0, n)
    
    # Nelder-Mead with adaptive settings
    options = {
        'maxiter': 2000,
        'xatol': 1e-8,
        'fatol': 1e-10,
        'adaptive': True,
    }
    
    try:
        result = minimize(_objective, x0, args=(n,), method='Nelder-Mead', options=options)
        if result.fun < best_f:
            best_x = result.x
            best_f = result.fun
    except Exception:
        pass
    
    return best_x.reshape(n, 2)

def _force_relax(pts, n, steps=5000, lr=0.01):
    """Simple force-directed relaxation to push circles apart and keep inside disk."""
    pts = pts.copy()
    for step in range(steps):
        if time.time() > max(_DEADLINE - 10, 0):
            break
        forces = np.zeros_like(pts)
        # repulsive forces between circles
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    d = np.random.randn(2) * 1e-10
                    dist = np.linalg.norm(d)
                if dist < 0.3:  # only interact when close
                    f = d / dist * (0.3 - dist)
                    forces[i] += f
                    forces[j] -= f
        # boundary force: push inward if too close to edge
        for i in range(n):
            r = np.linalg.norm(pts[i])
            if r > 0.8:
                f = -pts[i] / r * (r - 0.8)
                forces[i] += f
        # centering force to prevent drift
        forces -= 0.001 * pts
        pts += lr * forces
        # clip to unit disk
        r = np.linalg.norm(pts, axis=1)
        mask = r > 0.99
        if np.any(mask):
            pts[mask] *= 0.99 / r[mask]
    return pts

def construct():
    t_start = time.time()
    deadline = t_start + 135.0  # 15s safety margin
    
    # Initialize with hex lattice
    pts = _hex_lattice_init(N)
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    # Force relaxation first to spread out
    if time.time() < deadline - 30:
        pts = _force_relax(pts, N, steps=3000, lr=0.005)
        s = _compute_score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Nelder-Mead optimization
    if time.time() < deadline - 10:
        pts = _nelder_mead_optimize(best_pts, N, max_time=deadline)
        s = _compute_score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Another round of force relaxation
    if time.time() < deadline - 5:
        pts = _force_relax(best_pts, N, steps=2000, lr=0.003)
        s = _compute_score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Final Nelder-Mead polish
    if time.time() < deadline - 2:
        pts = _nelder_mead_optimize(best_pts, N, max_time=deadline)
        s = _compute_score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Ensure validity
    r = np.linalg.norm(best_pts, axis=1)
    if np.any(r > 1.0):
        best_pts[r > 1.0] *= 1.0 / r[r > 1.0]
    
    return best_pts
