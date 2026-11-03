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

def _softplus_grad(z, eps=1e-4):
    """Gradient of softplus."""
    return 1.0 / (1.0 + np.exp(-z / eps))

def _objective_and_grad(params, penalty_weight, smooth_eps=1e-4):
    """
    Objective and analytical gradient.
    params: flat array [x0,y0,r0, x1,y1,r1, ...]
    Returns (value, gradient).
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    n = len(r)
    
    val = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] -= 1.0
    
    # Boundary violations
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    
    for viol, idx, sign in [(lx, slice(0, None, 3), -1), (ux, slice(0, None, 3), 1), (ly, slice(1, None, 3), -1), (uy, slice(1, None, 3), 1)]:
        sp = _softplus(viol, smooth_eps)
        spg = _softplus_grad(viol, smooth_eps)
        val += np.sum(sp**2)
        # d/dparam of sp^2 = 2*sp*spg*d(viol)/dparam
        grad[idx] += 2 * penalty_weight * sp * spg * sign
        grad[2::3] += 2 * penalty_weight * sp * spg * 1.0
    
    # Negative radius
    neg_r = -r
    sp = _softplus(neg_r, smooth_eps)
    spg = _softplus_grad(neg_r, smooth_eps)
    val += np.sum(sp**2)
    grad[2::3] += 2 * penalty_weight * sp * spg * (-1)
    
    # Pairwise overlaps
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    
    sp = _softplus(overlap, smooth_eps)
    spg = _softplus_grad(overlap, smooth_eps)
    val += np.sum(sp**2)
    
    # Gradient contributions for overlaps
    # d(overlap)/dx_i = -dx/dist, d(overlap)/dx_j = dx/dist
    # d(overlap)/dy_i = -dy/dist, d(overlap)/dy_j = dy/dist
    # d(overlap)/dr_i = 1, d(overlap)/dr_j = 1
    
    factor = 2 * penalty_weight * sp * spg
    inv_dist = 1.0 / dist
    
    # For each pair (i, j):
    # grad_x[i] += factor * (-dx/dist)
    # grad_x[j] += factor * (dx/dist)
    # grad_y[i] += factor * (-dy/dist)
    # grad_y[j] += factor * (dy/dist)
    # grad_r[i] += factor * 1
    # grad_r[j] += factor * 1
    
    gx_i = factor * (-dx * inv_dist)
    gx_j = factor * (dx * inv_dist)
    gy_i = factor * (-dy * inv_dist)
    gy_j = factor * (dy * inv_dist)
    gr = factor * 1.0
    
    np.add.at(grad[0::3], idx_i, gx_i)
    np.add.at(grad[0::3], idx_j, gx_j)
    np.add.at(grad[1::3], idx_i, gy_i)
    np.add.at(grad[1::3], idx_j, gy_j)
    np.add.at(grad[2::3], idx_i, gr)
    np.add.at(grad[2::3], idx_j, gr)
    
    return val, grad

def _objective_only(params, penalty_weight, smooth_eps=1e-4):
    """Objective value only."""
    val, _ = _objective_and_grad(params, penalty_weight, smooth_eps)
    return val

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
        if r < 0.5:
            xyz[i,0] = np.clip(xyz[i,0], r, 1-r)
            xyz[i,1] = np.clip(xyz[i,1], r, 1-r)
        else:
            xyz[i,0] = 0.5
            xyz[i,1] = 0.5
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

def _physics_relax(xyz, steps=1500, dt=0.001, deadline=None):
    """
    Physics-based relaxation: circles repel each other and are pushed to fill space.
    """
    xyz = xyz.copy().astype(float)
    n = N
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()
    
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
        
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    force = repulsion_k * overlap
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
        
        for i in range(n):
            if x[i] < r[i]:
                fx[i] += repulsion_k * (r[i] - x[i])
            if x[i] > 1 - r[i]:
                fx[i] -= repulsion_k * (x[i] - (1 - r[i]))
            if y[i] < r[i]:
                fy[i] += repulsion_k * (r[i] - y[i])
            if y[i] > 1 - r[i]:
                fy[i] -= repulsion_k * (y[i] - (1 - r[i]))
        
        cx = np.mean(x)
        cy = np.mean(y)
        for i in range(n):
            fx[i] += attraction_k * (cx - x[i])
            fy[i] += attraction_k * (cy - y[i])
        
        vx = vx * damping + fx * dt
        vy = vy * damping + fy * dt
        x += vx * dt
        y += vy * dt
        
        for i in range(n):
            r_i = r[i]
            if r_i < 0.5:
                x[i] = np.clip(x[i], r_i, 1 - r_i)
                y[i] = np.clip(y[i], r_i, 1 - r_i)
            else:
                x[i] = 0.5
                y[i] = 0.5
    
    xyz[:, 0] = x
    xyz[:, 1] = y
    return xyz

def _optimize_single(xyz0, deadline):
    """Optimize from a single initial config with adaptive penalty and analytical gradients."""
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
        xyz_phys = _physics_relax(xyz0, steps=1200, dt=0.001, deadline=deadline - 2)
        if _validate(xyz_phys):
            params = xyz_phys.flatten()
            best_params = params.copy()
            best_score = np.sum(params[2::3])
    
    # Optimization phases with increasing penalty and tighter smoothing
    phases = [
        (10.0, 1e-4, 300, 1e-14, 1e-12),
        (50.0, 1e-5, 300, 1e-14, 1e-12),
        (200.0, 1e-6, 250, 1e-14, 1e-12),
        (1000.0, 1e-7, 200, 1e-14, 1e-12),
        (5000.0, 1e-8, 200, 1e-15, 1e-13),
        (20000.0, 1e-9, 150, 1e-15, 1e-13),
    ]
    
    for penalty, smooth_eps, maxiter, ftol, gtol in phases:
        remaining = deadline - time.time()
        if remaining < 1.5:
            break
        
        try:
            res = minimize(_objective_only, params, jac=lambda p: _objective_and_grad(p, penalty, smooth_eps)[1],
                          method='L-BFGS-B',
                          bounds=bounds, args=(penalty, smooth_eps),
                          options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
            
            if res.fun < _objective_only(params, penalty, smooth_eps):
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
