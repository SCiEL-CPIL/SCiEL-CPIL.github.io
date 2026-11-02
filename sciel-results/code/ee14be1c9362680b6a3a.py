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

def _objective_vectorized(params, penalty_weight, smooth_eps=1e-4):
    """
    Vectorized objective: minimize -sum(r) + smooth penalty for violations.
    params: flat array [x0,y0,r0, x1,y1,r1, ...]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary violations (smooth)
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    
    penalty += np.sum(_softplus(lx, smooth_eps)**2)
    penalty += np.sum(_softplus(ux, smooth_eps)**2)
    penalty += np.sum(_softplus(ly, smooth_eps)**2)
    penalty += np.sum(_softplus(uy, smooth_eps)**2)
    
    # Negative radius
    penalty += np.sum(_softplus(-r, smooth_eps)**2)
    
    # Pairwise overlaps (vectorized)
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
    return -np.sum(r) + penalty_weight * penalty

def _get_delaunay_edges(centers):
    """Get Delaunay edges from centers."""
    try:
        tri = Delaunay(centers)
        edge_set = set()
        for simplex in tri.simplices:
            for k in range(3):
                a, b = simplex[k], simplex[(k+1)%3]
                edge_set.add((min(a,b), max(a,b)))
        return list(edge_set)
    except Exception:
        idx_i, idx_j = np.triu_indices(N, k=1)
        return list(zip(idx_i, idx_j))

def _objective_delaunay(params, penalty_weight, smooth_eps=1e-4):
    """
    Objective using Delaunay edges for overlap checking (faster).
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary violations
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    penalty += np.sum(_softplus(lx, smooth_eps)**2)
    penalty += np.sum(_softplus(ux, smooth_eps)**2)
    penalty += np.sum(_softplus(ly, smooth_eps)**2)
    penalty += np.sum(_softplus(uy, smooth_eps)**2)
    penalty += np.sum(_softplus(-r, smooth_eps)**2)
    
    # Delaunay edges for overlap
    centers = np.column_stack([x, y])
    edges = _get_delaunay_edges(centers)
    edge_set = set(edges)
    
    for i, j in edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
        overlap = r[i] + r[j] - dist
        if overlap > 0:
            penalty += _softplus(overlap, smooth_eps)**2
    
    # Also check nearest neighbors not in Delaunay
    try:
        tree = cKDTree(centers)
        for i in range(N):
            dists, indices = tree.query(centers[i], k=min(N, 8))
            for j in indices[1:]:
                if j == i:
                    continue
                if (min(i,j), max(i,j)) in edge_set:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    penalty += _softplus(overlap, smooth_eps)**2
    except Exception:
        pass
    
    return -np.sum(r) + penalty_weight * penalty

def _validate(xyz, tol=1e-7):
    """Check validity."""
    n = xyz.shape[0]
    x, y, r = xyz[:,0], xyz[:,1], xyz[:,2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i]-x[j]
            dy = y[i]-y[j]
            dist = np.sqrt(dx*dx+dy*dy)
            if dist < r[i]+r[j]-tol:
                return False
    return True

def _fix_boundaries(xyz):
    """Project circles inside unit square."""
    xyz = xyz.copy()
    for i in range(N):
        r = max(0, xyz[i,2])
        xyz[i,0] = np.clip(xyz[i,0], r, 1-r) if r < 0.5 else 0.5
        xyz[i,1] = np.clip(xyz[i,1], r, 1-r) if r < 0.5 else 0.5
        xyz[i,2] = r
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
                x = (col + 0.5) / cols + 0.5/cols
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
        positions.append([0.5, 0.5, r*0.5])
    return np.array(positions[:N])

def _initial_grid():
    """Grid initial config."""
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    r = 0.5/side * 0.9
    for i in range(side):
        for j in range(side):
            x = (i+0.5)/side
            y = (j+0.5)/side
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r*0.5])
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
            angle = 2*np.pi*k/n_circles + rng.uniform(-0.1, 0.1)
            dist = ring * 0.14
            x = 0.5 + dist*np.cos(angle)
            y = 0.5 + dist*np.sin(angle)
            positions.append([x, y, radius])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([rng.uniform(0.1,0.9), rng.uniform(0.1,0.9), 0.06])
    return np.array(positions[:N])

def _physics_relax(xyz, steps=2000, dt=0.001, deadline=None):
    """
    Physics-based relaxation: circles repel each other and are pushed to fill space.
    Uses a simple force-directed approach with gravity toward center of mass.
    """
    xyz = xyz.copy().astype(float)
    n = N
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()
    
    # Force constants
    repulsion_k = 50.0
    attraction_k = 2.0
    damping = 0.95
    
    vx = np.zeros(n)
    vy = np.zeros(n)
    
    for step in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Repulsion between overlapping circles
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    # Repulsive force proportional to overlap
                    overlap = min_dist - dist
                    force = repulsion_k * overlap
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
        
        # Boundary forces
        for i in range(n):
            if x[i] < r[i]:
                fx[i] += repulsion_k * (r[i] - x[i])
            if x[i] > 1 - r[i]:
                fx[i] -= repulsion_k * (x[i] - (1 - r[i]))
            if y[i] < r[i]:
                fy[i] += repulsion_k * (r[i] - y[i])
            if y[i] > 1 - r[i]:
                fy[i] -= repulsion_k * (y[i] - (1 - r[i]))
        
        # Attraction toward center of mass (to pack tightly)
        cx = np.mean(x)
        cy = np.mean(y)
        for i in range(n):
            fx[i] += attraction_k * (cx - x[i])
            fy[i] += attraction_k * (cy - y[i])
        
        # Update velocities and positions
        vx = vx * damping + fx * dt
        vy = vy * damping + fy * dt
        x += vx * dt
        y += vy * dt
        
        # Project back to valid region
        for i in range(n):
            r_i = r[i]
            x[i] = np.clip(x[i], r_i, 1 - r_i) if r_i < 0.5 else 0.5
            y[i] = np.clip(y[i], r_i, 1 - r_i) if r_i < 0.5 else 0.5
    
    xyz[:, 0] = x
    xyz[:, 1] = y
    return xyz

def _optimize_single(xyz0, deadline):
    """Optimize from a single initial config with adaptive penalty."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    # Phase 0: Physics relaxation
    remaining = deadline - time.time()
    if remaining > 5:
        xyz_phys = _physics_relax(xyz0, steps=1000, dt=0.001, deadline=deadline - 2)
        if _validate(xyz_phys):
            params = xyz_phys.flatten()
            best_params = params.copy()
            best_score = np.sum(params[2::3])
    
    # Phase 1: Delaunay-based with moderate penalty
    penalty = 10.0
    smooth_eps = 1e-4
    remaining = deadline - time.time()
    if remaining > 3:
        try:
            res = minimize(_objective_delaunay, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, smooth_eps),
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective_delaunay(params, penalty, smooth_eps):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 2: Increase penalty, full pairwise
    remaining = deadline - time.time()
    if remaining > 3:
        penalty = 50.0
        smooth_eps = 1e-5
        try:
            res = minimize(_objective_vectorized, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, smooth_eps),
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective_vectorized(params, penalty, smooth_eps):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 3: Very high penalty for fine-tuning
    remaining = deadline - time.time()
    if remaining > 2:
        penalty = 200.0
        smooth_eps = 1e-6
        try:
            res = minimize(_objective_vectorized, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, smooth_eps),
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective_vectorized(params, penalty, smooth_eps):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 4: Even higher penalty
    remaining = deadline - time.time()
    if remaining > 1.5:
        penalty = 1000.0
        smooth_eps = 1e-7
        try:
            res = minimize(_objective_vectorized, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, smooth_eps),
                          options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective_vectorized(params, penalty, smooth_eps):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 5: Final high-precision polish
    remaining = deadline - time.time()
    if remaining > 1:
        penalty = 5000.0
        smooth_eps = 1e-8
        try:
            res = minimize(_objective_vectorized, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, smooth_eps),
                          options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < _objective_vectorized(params, penalty, smooth_eps):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    return best_params.reshape(N, 3), best_score

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
    for s in range(3):
        x = rng.uniform(0.05, 0.95, N)
        y = rng.uniform(0.05, 0.95, N)
        r = rng.uniform(0.05, 0.12, N)
        cfg = np.column_stack([x, y, r])
        cfg = _fix_boundaries(cfg)
        inits.append(cfg)
    
    # Optimize from each
    for xyz0 in inits:
        remaining = deadline - time.time()
        if remaining < 8.0:
            break
        time_budget = deadline - 2.0
        xyz_opt, score = _optimize_single(xyz0, time_budget)
        
        # Fix if invalid
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
