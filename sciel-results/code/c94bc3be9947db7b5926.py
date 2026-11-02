"""Pack 300 equal circles in a unit circle using multi-stage optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300
# Deadline guard: finish well before the 150s wall-clock limit.
_DEADLINE = 140.0


def _score(pts):
    """Return the achievable common radius for a set of centres."""
    if pts.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Boundary term
    r_boundary = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
    if r_boundary <= 0:
        return 0.0
    # Pairwise distances (vectorised chunked to keep memory bounded)
    min_dist = np.inf
    chunk = 64
    for i in range(0, N, chunk):
        a = pts[i:i + chunk]
        # distances from a to all points
        d = np.linalg.norm(a[:, None, :] - pts[None, :, :], axis=2)
        # zero out self
        idx = np.arange(a.shape[0])
        d[np.arange(a.shape[0]), (i + idx) % N] = np.inf
        # but need correct indexing: row j of d corresponds to a[j], column k to pts[k]
        # self indices are i+j
        for j in range(a.shape[0]):
            d[j, i + j] = np.inf
        min_dist = min(min_dist, d.min())
    return min(min_dist / 2.0, r_boundary)


def _initial_hex():
    """Generate a hexagonal lattice packing as starting point."""
    # Use a hexagonal grid with spacing tuned for 300 points
    # Approximate number of points in a hex grid of radius R with spacing s: ~pi*R^2/(s^2*sqrt(3)/2)
    # We want ~300 points in unit circle, so s^2 ~ 2*pi/(300*sqrt(3)) ≈ 0.0121, s ≈ 0.11
    spacing = 0.11
    points = []
    # Generate hex lattice points
    s = spacing
    # Range to cover unit circle
    max_k = int(1.0 / (s * np.sqrt(3) / 2)) + 2
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2.0
            if x * x + y * y <= (1.0 - s / 2) ** 2:
                points.append((x, y))
    points = np.array(points)
    if len(points) >= N:
        # Sort by distance from origin, take innermost N
        r = np.hypot(points[:, 0], points[:, 1])
        order = np.argsort(r, kind='stable')
        return points[order[:N]]
    # If not enough, adjust spacing
    for _ in range(20):
        spacing *= 0.97
        s = spacing
        max_k = int(1.0 / (s * np.sqrt(3) / 2)) + 2
        points = []
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2.0
                if x * x + y * y <= (1.0 - s / 2) ** 2:
                    points.append((x, y))
        points = np.array(points)
        if len(points) >= N:
            r = np.hypot(points[:, 0], pts[:, 1] if False else points[:, 1])
            r = np.hypot(points[:, 0], points[:, 1])
            order = np.argsort(r, kind='stable')
            return points[order[:N]]
    # Fallback: random
    return np.random.randn(N, 2) * 0.5


def _local_search(pts, max_iters=200, lr=0.01, deadline=None):
    """Simple gradient-free local search using pairwise repulsion."""
    pts = pts.copy()
    best_score = _score(pts)
    best_pts = pts.copy()
    t0 = time.time()
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        # Compute forces
        forces = np.zeros_like(pts)
        # Boundary force
        r = np.hypot(pts[:, 0], pts[:, 1])
        for i in range(N):
            if r[i] > 0.99:
                forces[i] -= 0.5 * (r[i] - 0.99) * pts[i] / max(r[i], 1e-10)
        
        # Pairwise repulsion (chunked)
        chunk = 32
        for i in range(0, N, chunk):
            a = pts[i:i+chunk]
            d = np.linalg.norm(a[:, None, :] - pts[None, :, :], axis=2)
            # Self distances
            for j in range(a.shape[0]):
                d[j, i+j] = np.inf
            # Only repel if too close
            min_d = 0.1
            mask = d < min_d
            if np.any(mask):
                for j in range(a.shape[0]):
                    # Sum forces from too-close neighbors
                    f = np.zeros(2)
                    for k in range(N):
                        if d[j, k] < min_d and d[j, k] > 1e-10:
                            direction = a[j] - pts[k]
                            f += direction / d[j, k] * (min_d - d[j, k])
                    forces[i+j] += f
        
        # Update
        pts = pts + lr * forces
        # Keep within bounds
        r = np.hypot(pts[:, 0], pts[:, 1])
        for i in range(N):
            if r[i] > 0.999:
                pts[i] *= 0.999 / r[i]
        
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_pts = pts.copy()
    
    return best_pts, best_score


def construct():
    t_start = time.time()
    deadline = t_start + _DEADLINE - 5.0  # safety margin
    
    # Stage 1: Initial hexagonal packing
    pts = _initial_hex()
    current_score = _score(pts)
    
    # Stage 2: Local search to improve
    if time.time() < deadline - 30:
        pts, current_score = _local_search(pts, max_iters=100, lr=0.005, deadline=deadline - 30)
    
    # Stage 3: Multi-start with random perturbations
    best_pts = pts.copy()
    best_score = current_score
    
    if time.time() < deadline - 10:
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            # Perturb the best solution
            perturbed = best_pts + np.random.randn(N, 2) * 0.02
            # Project back inside
            r = np.hypot(perturbed[:, 0], perturbed[:, 1])
            for i in range(N):
                if r[i] > 0.999:
                    perturbed[i] *= 0.999 / r[i]
            
            sc = _score(perturbed)
            if sc > best_score:
                best_score = sc
                best_pts = perturbed.copy()
            
            # Local search from perturbed
            if time.time() < deadline - 10:
                local_pts, local_score = _local_search(perturbed, max_iters=50, lr=0.003, deadline=deadline - 10)
                if local_score > best_score:
                    best_score = local_score
                    best_pts = local_pts.copy()
    
    return best_pts


# Fix the bug in _initial_hex (there was a typo)
def _initial_hex():
    """Generate a hexagonal lattice packing as starting point."""
    spacing = 0.11
    points = []
    s = spacing
    max_k = int(1.0 / (s * np.sqrt(3) / 2)) + 2
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2.0
            if x * x + y * y <= (1.0 - s / 2) ** 2:
                points.append((x, y))
    points = np.array(points)
    if len(points) >= N:
        r = np.hypot(points[:, 0], points[:, 1])
        order = np.argsort(r, kind='stable')
        return points[order[:N]]
    for _ in range(20):
        spacing *= 0.97
        s = spacing
        max_k = int(1.0 / (s * np.sqrt(3) / 2)) + 2
        points = []
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2.0
                if x * x + y * y <= (1.0 - s / 2) ** 2:
                    points.append((x, y))
        points = np.array(points)
        if len(points) >= N:
            r = np.hypot(points[:, 0], points[:, 1])
            order = np.argsort(r, kind='stable')
            return points[order[:N]]
    return np.random.randn(N, 2) * 0.5
