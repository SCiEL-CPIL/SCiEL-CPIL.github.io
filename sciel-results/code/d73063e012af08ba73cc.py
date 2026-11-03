"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay, cKDTree

N = 26

# EVOLVE-BLOCK-START

def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0)."""
    return np.logaddexp(0, z / eps) * eps

def _voronoi_radii(centers, eps=1e-6):
    """
    Compute radii from center positions using Voronoi cell distances.
    radius_i = min(distance to boundary, distance to nearest Voronoi edge / 2)
    This ensures non-overlap by construction.
    """
    n = len(centers)
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Distance to boundary
    r_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    
    # Distance to nearest other center (for Voronoi approximation)
    # Use Delaunay neighbors for efficiency
    r_voronoi = np.full(n, np.inf)
    
    try:
        tri = Delaunay(centers)
        edge_set = set()
        for simplex in tri.simplices:
            for k in range(3):
                a, b = simplex[k], simplex[(k+1) % 3]
                edge_set.add((min(a, b), max(a, b)))
        
        for i, j in edge_set:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            # In Voronoi, the shared edge is at half the distance
            half_dist = dist / 2.0
            r_voronoi[i] = min(r_voronoi[i], half_dist)
            r_voronoi[j] = min(r_voronoi[j], half_dist)
    except Exception:
        # Fallback: use all pairs
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                half_dist = dist / 2.0
                r_voronoi[i] = min(r_voronoi[i], half_dist)
                r_voronoi[j] = min(r_voronoi[j], half_dist)
    
    # Final radius is minimum of boundary and Voronoi constraints
    radii = np.minimum(r_boundary, r_voronoi)
    radii = np.maximum(radii, 1e-8)  # Ensure positive
    
    return radii

def _objective_voronoi(centers, eps=1e-6):
    """
    Objective: maximize sum of radii derived from Voronoi cells.
    Only optimize center positions (2N = 52 dimensions).
    """
    radii = _voronoi_radii(centers, eps)
    return -np.sum(radii)

def _objective_voronoi_grad(centers, eps=1e-6):
    """
    Compute objective and gradient for Voronoi-based optimization.
    Uses finite differences for gradient since analytic gradient of min() is complex.
    """
    f0 = _objective_voronoi(centers, eps)
    
    grad = np.zeros_like(centers)
    h = 1e-7
    
    for i in range(len(centers)):
        for d in range(2):
            centers_plus = centers.copy()
            centers_plus[i, d] += h
            f_plus = _objective_voronoi(centers_plus, eps)
            grad[i, d] = (f_plus - f0) / h
    
    return f0, grad

def _validate(xyz, tol=1e-7):
    """Check validity."""
    n = xyz.shape[0]
    x, y, r = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _fix_boundaries(xyz):
    """Project circles inside unit square."""
    xyz = xyz.copy()
    for i in range(N):
        r = max(0, xyz[i, 2])
        xyz[i, 0] = np.clip(xyz[i, 0], r, 1 - r) if r < 0.5 else 0.5
        xyz[i, 1] = np.clip(xyz[i, 1], r, 1 - r) if r < 0.5 else 0.5
        xyz[i, 2] = r
    return xyz

def _initial_hex():
    """Hexagonal packing initial config."""
    cols = 5
    rows = 6
    positions = []
    r = 0.09
    for row in range(rows):
        for col in range(cols):
            if row % 2 == 1:
                x = (col + 0.5) / cols + 0.5 / cols
            else:
                x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            if x > 1 or y > 1:
                continue
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.5])
    return np.array(positions[:N])

def _initial_grid():
    """Grid initial config."""
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    r = 0.5 / side * 0.9
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.5])
    return np.array(positions[:N])

def _initial_corner_weighted(seed=0):
    """Random config with corner preference."""
    rng = np.random.default_rng(seed)
    positions = []
    corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92],
               [0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
    for cx, cy in corners:
        positions.append([cx, cy, 0.08])
    for _ in range(N - len(positions)):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        positions.append([x, y, rng.uniform(0.06, 0.10)])
    return np.array(positions[:N])

def _initial_ring(seed=0):
    """Ring-shaped initial config."""
    rng = np.random.default_rng(seed)
    positions = [[0.5, 0.5, 0.08]]
    for ring, n_circles, radius in [(1, 6, 0.10), (2, 10, 0.08), (3, 9, 0.07)]:
        for k in range(n_circles):
            angle = 2 * np.pi * k / n_circles + rng.uniform(-0.1, 0.1)
            dist = ring * 0.14
            x = 0.5 + dist * np.cos(angle)
            y = 0.5 + dist * np.sin(angle)
            positions.append([x, y, radius])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 0.06])
    return np.array(positions[:N])

def _optimize_voronoi(centers0, deadline):
    """
    Optimize center positions using Voronoi-derived radii.
    This reduces search space to 52 dimensions and ensures non-overlap.
    """
    centers = centers0.copy()
    best_centers = centers.copy()
    best_score = -_objective_voronoi(centers)
    
    bounds = [(0.01, 0.99)] * (2 * N)
    
    # Phase 1: L-BFGS-B with finite difference gradients
    remaining = deadline - time.time()
    if remaining > 5:
        try:
            res = minimize(_objective_voronoi, centers, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            if -res.fun > best_score:
                best_centers = res.x
                best_score = -res.fun
                centers = res.x
        except Exception:
            pass
    
    # Phase 2: More iterations with tighter tolerance
    remaining = deadline - time.time()
    if remaining > 3:
        try:
            res = minimize(_objective_voronoi, centers, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-12})
            if -res.fun > best_score:
                best_centers = res.x
                best_score = -res.fun
                centers = res.x
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead for local refinement (no gradient needed)
    remaining = deadline - time.time()
    if remaining > 2:
        try:
            res = minimize(_objective_voronoi, centers, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
            if -res.fun > best_score:
                best_centers = res.x
                best_score = -res.fun
                centers = res.x
        except Exception:
            pass
    
    # Compute final radii
    radii = _voronoi_radii(best_centers)
    xyz = np.column_stack([best_centers[:, 0], best_centers[:, 1], radii])
    
    # Ensure validity
    if not _validate(xyz):
        xyz_fix = xyz.copy()
        for _ in range(100):
            if _validate(xyz_fix):
                break
            xyz_fix[:, 2] *= 0.98
            xyz_fix = _fix_boundaries(xyz_fix)
        xyz = xyz_fix
    
    return xyz, np.sum(xyz[:, 2])

def _optimize_single(xyz0, deadline):
    """
    Optimize from a single initial config using both Voronoi and penalty methods.
    """
    # Extract centers from initial config
    centers0 = xyz0[:, :2].copy()
    
    # Method 1: Voronoi-based optimization (primary)
    xyz_voronoi, score_voronoi = _optimize_voronoi(centers0, deadline)
    
    # Method 2: Traditional penalty-based optimization (fallback/comparison)
    remaining = deadline - time.time()
    if remaining > 5:
        params = xyz0.flatten().copy()
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
            bounds.append((0, 1))
            bounds.append((0, 0.5))
        
        try:
            res = minimize(_objective_vectorized, params, method='L-BFGS-B',
                          bounds=bounds, args=(50.0, 1e-5),
                          options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10})
            xyz_penalty = res.x.reshape(N, 3)
            if not _validate(xyz_penalty):
                xyz_penalty = _fix_boundaries(xyz_penalty)
                if not _validate(xyz_penalty):
                    xyz_penalty = xyz0
            score_penalty = np.sum(xyz_penalty[:, 2])
        except Exception:
            xyz_penalty = xyz0
            score_penalty = np.sum(xyz0[:, 2])
    else:
        xyz_penalty = xyz0
        score_penalty = np.sum(xyz0[:, 2])
    
    # Choose best result
    if score_voronoi >= score_penalty:
        return xyz_voronoi, score_voronoi
    else:
        return xyz_penalty, score_penalty

def _objective_vectorized(params, penalty_weight, smooth_eps=1e-4):
    """
    Vectorized objective: minimize -sum(r) + smooth penalty for violations.
    params: flat array [x0,y0,r0, x1,y1,r1, ...]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    
    penalty += np.sum(_softplus(lx, smooth_eps) ** 2)
    penalty += np.sum(_softplus(ux, smooth_eps) ** 2)
    penalty += np.sum(_softplus(ly, smooth_eps) ** 2)
    penalty += np.sum(_softplus(uy, smooth_eps) ** 2)
    penalty += np.sum(_softplus(-r, smooth_eps) ** 2)
    
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    penalty += np.sum(_softplus(overlap, smooth_eps) ** 2)
    
    return -np.sum(r) + penalty_weight * penalty

def construct():
    global_start = time.time()
    deadline = global_start + 140.0  # 10s safety margin before 150s kill
    
    best_xyz = _initial_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Generate diverse initial configurations
    inits = []
    inits.append(_initial_grid())
    inits.append(_initial_hex())
    inits.append(_initial_corner_weighted(0))
    inits.append(_initial_corner_weighted(1))
    inits.append(_initial_ring(0))
    inits.append(_initial_ring(1))
    
    # Random inits
    rng = np.random.default_rng(123)
    for s in range(4):
        x = rng.uniform(0.05, 0.95, N)
        y = rng.uniform(0.05, 0.95, N)
        r = rng.uniform(0.05, 0.12, N)
        cfg = np.column_stack([x, y, r])
        cfg = _fix_boundaries(cfg)
        inits.append(cfg)
    
    # Optimize from each
    for xyz0 in inits:
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
        time_budget = deadline - 2.0
        xyz_opt, score = _optimize_single(xyz0, time_budget)
        
        if not _validate(xyz_opt):
            xyz_fix = xyz_opt.copy()
            for _ in range(50):
                if _validate(xyz_fix):
                    break
                xyz_fix[:, 2] *= 0.97
                xyz_fix = _fix_boundaries(xyz_fix)
            xyz_opt = xyz_fix
        
        if not _validate(xyz_opt):
            continue
        
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
    
    # Final validation
    if not _validate(best_xyz):
        best_xyz = _initial_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
