import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
import time

N = 32

def _voronoi_inradii(centers, tol=1e-12):
    """For each center, compute the maximum radius such that the circle stays
    inside [0,1]^2 and does not overlap any other circle (using Voronoi cells)."""
    n = len(centers)
    radii = np.zeros(n)
    # Add boundary "points" far outside to make Voronoi cells bounded
    # We use a trick: add 4 points far away so that each cell is bounded
    # Actually, for inradius we need distance to nearest other center / 2 AND distance to boundary
    # Simpler: for each circle, the max radius is min(
    #   distance to each boundary,
    #   distance to each other center / 2
    # )
    # But that's O(n^2). For n=32 that's fine.
    for i in range(n):
        cx, cy = centers[i]
        # Distance to boundaries
        d_bound = min(cx, 1 - cx, cy, 1 - cy)
        # Distance to other centers
        min_d_center = np.inf
        for j in range(n):
            if i == j:
                continue
            dx = centers[j, 0] - cx
            dy = centers[j, 1] - cy
            d = np.sqrt(dx * dx + dy * dy)
            if d < min_d_center:
                min_d_center = d
        radii[i] = min(d_bound, min_d_center / 2.0)
    return radii

def _sum_radii(centers):
    radii = _voronoi_inradii(centers)
    return np.sum(radii)

def _objective(centers):
    return -_sum_radii(centers)

def _is_valid(centers, radii, tol=1e-9):
    n = len(radii)
    if np.any(radii < -tol):
        return False
    for i in range(n):
        if centers[i, 0] - radii[i] < -tol:
            return False
        if 1 - centers[i, 0] - radii[i] < -tol:
            return False
        if centers[i, 1] - radii[i] < -tol:
            return False
        if 1 - centers[i, 1] - radii[i] < -tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[j, 0] - centers[i, 0]
            dy = centers[j, 1] - centers[i, 1]
            d = np.sqrt(dx * dx + dy * dy)
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _grid_init(rows, cols, rng):
    centers = np.zeros((N, 2))
    for i in range(N):
        row = i // cols
        col = i % cols
        centers[i, 0] = (col + 0.5) / cols
        centers[i, 1] = (row + 0.5) / rows
    return centers

def _hex_init(rng):
    """Hexagonal packing: 4 rows of 8, or 8 rows of 4, etc."""
    # Try 4 rows of 8
    rows = 4
    cols = 8
    centers = np.zeros((N, 2))
    for i in range(N):
        row = i // cols
        col = i % cols
        offset = 0.5 / cols if row % 2 == 1 else 0.0
        centers[i, 0] = (col + 0.5) / cols + offset
        centers[i, 1] = (row + 0.5) / rows
    # Clip to [0,1]
    centers[:, 0] = np.clip(centers[:, 0], 0, 1)
    centers[:, 1] = np.clip(centers[:, 1], 0, 1)
    return centers

def _random_init(rng):
    return rng.uniform(0.05, 0.95, size=(N, 2))

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    rng = np.random.RandomState(42)
    
    best_centers = None
    best_score = -np.inf
    
    # Initial configurations to try
    inits = []
    inits.append(_grid_init(6, 6, rng))       # 6x6 grid
    inits.append(_grid_init(4, 8, rng))       # 4x8 grid
    inits.append(_grid_init(8, 4, rng))       # 8x4 grid
    inits.append(_hex_init(rng))              # Hexagonal
    inits.append(_random_init(rng))           # Random
    inits.append(_random_init(rng))           # Random 2
    inits.append(_random_init(rng))           # Random 3
    
    for init_idx, centers in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Nelder-Mead on centers only
        try:
            result = minimize(
                _objective,
                centers,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
        except Exception:
            continue
        
        centers_opt = result.x.reshape(N, 2)
        radii = _voronoi_inradii(centers_opt)
        
        if _is_valid(centers_opt, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_centers = centers_opt.copy()
        
        # Try a few perturbations of the best found so far
        if best_centers is not None and time.time() < deadline - 10:
            for _ in range(3):
                if time.time() > deadline:
                    break
                pert = best_centers + rng.uniform(-0.02, 0.02, size=(N, 2))
                pert[:, 0] = np.clip(pert[:, 0], 0, 1)
                pert[:, 1] = np.clip(pert[:, 1], 0, 1)
                
                try:
                    res2 = minimize(
                        _objective,
                        pert,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 2000,
                            'xatol': 1e-10,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
                except Exception:
                    continue
                
                c2 = res2.x.reshape(N, 2)
                r2 = _voronoi_inradii(c2)
                if _is_valid(c2, r2):
                    s2 = np.sum(r2)
                    if s2 > best_score:
                        best_score = s2
                        best_centers = c2.copy()
    
    if best_centers is not None:
        radii = _voronoi_inradii(best_centers)
        # Final validity check
        if _is_valid(best_centers, radii):
            return np.column_stack([best_centers, radii])
    
    # Fallback: 6x6 grid
    centers = _grid_init(6, 6, rng)
    radii = _voronoi_inradii(centers)
    return np.column_stack([centers, radii])
