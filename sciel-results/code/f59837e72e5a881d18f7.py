"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
_GLOBAL_START = None
_GLOBAL_DEADLINE = None

# EVOLVE-BLOCK-START

def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0) using log-sum-exp."""
    return np.logaddexp(0, z / eps) * eps

def _softplus_grad(z, eps=1e-4):
    """Gradient of softplus: sigmoid(z/eps)."""
    z_scaled = z / eps
    # Numerically stable sigmoid
    pos = z_scaled > 0
    result = np.zeros_like(z_scaled)
    result[pos] = 1.0 / (1.0 + np.exp(-z_scaled[pos]))
    exp_z = np.exp(z_scaled[~pos])
    result[~pos] = exp_z / (1.0 + exp_z)
    return result

def _objective(params, penalty_weight, smooth_eps=1e-4):
    """
    Vectorized objective for N=26 circles.
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
    
    # Full pairwise overlaps (N=26 is small enough for full pairwise)
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
    return -np.sum(r) + penalty_weight * penalty

def _objective_grad(params, penalty_weight, smooth_eps=1e-4):
    """
    Analytic gradient of the objective.
    Returns (value, gradient) where gradient has same shape as params.
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    n = len(r)
    
    grad = np.zeros_like(params)
    
    # Boundary violations
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    
    # d(softplus^2)/dz = 2*softplus * sigmoid(z/eps)
    sp_lx = _softplus(lx, smooth_eps)
    sp_ux = _softplus(ux, smooth_eps)
    sp_ly = _softplus(ly, smooth_eps)
    sp_uy = _softplus(uy, smooth_eps)
    
    g_lx = 2 * sp_lx * _softplus_grad(lx, smooth_eps)
    g_ux = 2 * sp_ux * _softplus_grad(ux, smooth_eps)
    g_ly = 2 * sp_ly * _softplus_grad(ly, smooth_eps)
    g_uy = 2 * sp_uy * _softplus_grad(uy, smooth_eps)
    
    # d/dx: from lx (r-x): -g_lx; from ux (x+r-1): +g_ux
    grad[0::3] += penalty_weight * (g_ux - g_lx)
    # d/dy: from ly: -g_ly; from uy: +g_uy
    grad[1::3] += penalty_weight * (g_uy - g_ly)
    # d/dr: from lx: +g_lx; from ux: +g_ux; from ly: +g_ly; from uy: +g_uy
    grad[2::3] += penalty_weight * (g_lx + g_ux + g_ly + g_uy)
    
    # Negative radius
    nr = -r
    sp_nr = _softplus(nr, smooth_eps)
    g_nr = 2 * sp_nr * _softplus_grad(nr, smooth_eps)
    grad[2::3] += penalty_weight * (-g_nr)  # d/dr of softplus(-r)^2 = -g_nr
    
    # Objective: -sum(r)
    grad[2::3] -= 1.0
    
    # Pairwise overlaps
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    
    sp_ov = _softplus(overlap, smooth_eps)
    g_ov = 2 * sp_ov * _softplus_grad(overlap, smooth_eps)
    
    # d/dx_i of overlap = -dx/dist, d/dx_j = +dx/dist
    # d/dy_i of overlap = -dy/dist, d/dy_j = +dy/dist
    # d/dr_i = 1, d/dr_j = 1
    
    inv_dist = 1.0 / dist
    
    for k in range(len(idx_i)):
        i, j = idx_i[k], idx_j[k]
        gk = penalty_weight * g_ov[k]
        grad[3*i]   += gk * (-dx[k] * inv_dist[k])
        grad[3*j]   += gk * ( dx[k] * inv_dist[k])
        grad[3*i+1] += gk * (-dy[k] * inv_dist[k])
        grad[3*j+1] += gk * ( dy[k] * inv_dist[k])
        grad[3*i+2] += gk
        grad[3*j+2] += gk
    
    # Compute value
    penalty = (np.sum(sp_lx**2) + np.sum(sp_ux**2) + np.sum(sp_ly**2) + 
               np.sum(sp_uy**2) + np.sum(sp_nr**2) + np.sum(sp_ov**2))
    value = -np.sum(r) + penalty_weight * penalty
    
    return value, grad

def _physics_relax(xyz, n_steps=500, dt=0.001, k_repel=50.0, k_attract=5.0):
    """
    Physics-based relaxation: repulsive forces for overlaps, 
    attractive forces to encourage larger radii.
    """
    xyz = xyz.copy()
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    
    for step in range(n_steps):
        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)
        
        # Repulsive forces between overlapping circles
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(max(dist_sq, 1e-12))
                min_dist = r[i] + r[j]
                
                if dist < min_dist:
                    # Strong repulsion
                    overlap = min_dist - dist
                    force = k_repel * overlap
                    if dist > 1e-10:
                        fx[i] += force * dx / dist
                        fy[i] += force * dy / dist
                        fx[j] -= force * dx / dist
                        fy[j] -= force * dy / dist
                    # Encourage radius reduction when overlapping
                    fr[i] -= k_repel * 0.5 * overlap
                    fr[j] -= k_repel * 0.5 * overlap
        
        # Boundary forces
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += k_repel * (r[i] - x[i])
            elif x[i] > 1 - r[i]:
                fx[i] -= k_repel * (x[i] - (1 - r[i]))
            if y[i] < r[i]:
                fy[i] += k_repel * (r[i] - y[i])
            elif y[i] > 1 - r[i]:
                fy[i] -= k_repel * (y[i] - (1 - r[i]))
            if r[i] < 0:
                fr[i] += k_repel * (-r[i])
        
        # Update positions
        x += dt * fx
        y += dt * fy
        r += dt * fr
        
        # Clamp to valid range
        for i in range(N):
            r[i] = max(0.001, min(r[i], 0.45))
            x[i] = np.clip(x[i], r[i], 1 - r[i]) if r[i] < 0.5 else 0.5
            y[i] = np.clip(y[i], r[i], 1 - r[i]) if r[i] < 0.5 else 0.5
        
        xyz[:, 0] = x
        xyz[:, 1] = y
        xyz[:, 2] = r
    
    return xyz

def _enforce_validity(xyz):
    """Ensure the configuration is valid by shrinking radii if needed."""
    xyz = xyz.copy()
    for _ in range(100):
        if _validate(xyz):
            break
        # Find minimum violation
        x, y, r = xyz[:,0], xyz[:,1], xyz[:,2]
        min_scale = 1.0
        for i in range(N):
            if x[i] - r[i] < 0:
                min_scale = min(min_scale, x[i] / (r[i] + 1e-10) if r[i] > 0 else 1.0)
            if x[i] + r[i] > 1:
                min_scale = min(min_scale, (1 - x[i]) / (r[i] + 1e-10) if r[i] > 0 else 1.0)
            if y[i] - r[i] < 0:
                min_scale = min(min_scale, y[i] / (r[i] + 1e-10) if r[i] > 0 else 1.0)
            if y[i] + r[i] > 1:
                min_scale = min(min_scale, (1 - y[i]) / (r[i] + 1e-10) if r[i] > 0 else 1.0)
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j]:
                    needed = (r[i] + r[j]) / max(dist, 1e-10)
                    min_scale = min(min_scale, 1.0 / needed * 0.99)
        min_scale = max(0.5, min_scale * 0.99)
        xyz[:, 2] *= min_scale
        xyz = _fix_boundaries(xyz)
    return xyz

def _polish_analytic(xyz, penalty=100000, smooth_eps=1e-8, maxiter=200):
    """Final high-precision polish using analytic gradients."""
    params = xyz.flatten().copy()
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    try:
        res = minimize(_objective, params, method='L-BFGS-B',
                      bounds=bounds, args=(penalty, smooth_eps),
                      jac=lambda p: _objective_grad(p, penalty, smooth_eps)[1],
                      options={'maxiter': maxiter, 'ftol': 1e-18, 'gtol': 1e-16})
        if res.success or res.fun < _objective(params, penalty, smooth_eps):
            return res.x.reshape(N, 3)
    except Exception:
        pass
    return xyz

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

def _remaining_time():
    """Get remaining time before deadline."""
    if _GLOBAL_DEADLINE is None:
        return 140.0
    return max(0.0, _GLOBAL_DEADLINE - time.time())

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

def _initial_corner_cluster():
    """
    Specific initialization placing larger circles in corners and edges,
    which is a strong heuristic for this problem.
    """
    positions = []
    # 4 corners
    positions.append([0.1, 0.1, 0.12])
    positions.append([0.9, 0.1, 0.12])
    positions.append([0.1, 0.9, 0.12])
    positions.append([0.9, 0.9, 0.12])
    
    # 4 edge midpoints
    positions.append([0.5, 0.08, 0.10])
    positions.append([0.5, 0.92, 0.10])
    positions.append([0.08, 0.5, 0.10])
    positions.append([0.92, 0.5, 0.10])
    
    # Center
    positions.append([0.5, 0.5, 0.12])
    
    # Fill remaining with smaller circles in gaps
    rng = np.random.default_rng(123)
    while len(positions) < N:
        x = rng.uniform(0.15, 0.85)
        y = rng.uniform(0.15, 0.85)
        # Smaller radius to fit in gaps
        r = rng.uniform(0.05, 0.08)
        positions.append([x, y, r])
        
    return np.array(positions[:N])

def _get_contact_edges(params, tol=1e-3):
    """Get pairs of circles that are close (potential contacts)."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    n = len(r)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] + tol:
                edges.append((i, j))
    return edges

def _delaunay_objective(params, penalty_weight, smooth_eps=1e-4):
    """
    Objective using only Delaunay contact edges for speed in final phases.
    Falls back to full pairwise if contact set is too large.
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
    
    # Use contact edges for overlap penalty
    edges = _get_contact_edges(params, tol=1e-2)
    if len(edges) < 100:  # Use contact edges if manageable
        if len(edges) > 0:
            idx_i = np.array([e[0] for e in edges])
            idx_j = np.array([e[1] for e in edges])
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(np.maximum(dist_sq, 1e-12))
            overlap = r[idx_i] + r[idx_j] - dist
            penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    else:
        # Fall back to full pairwise
        n = len(r)
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
    return -np.sum(r) + penalty_weight * penalty

def _optimize_single(xyz0):
    """Optimize from a single initial config with adaptive penalty and analytic gradients."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    # Phase 1: Low penalty with analytic gradient
    penalty = 5.0
    remaining = _remaining_time()
    if remaining > 8:
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-3),
                          jac=lambda p: _objective_grad(p, penalty, 1e-3)[1],
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective(params, penalty, 1e-3):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 2: Medium penalty
    remaining = _remaining_time()
    if remaining > 6:
        penalty = 50.0
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4),
                          jac=lambda p: _objective_grad(p, penalty, 1e-4)[1],
                          options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective(params, penalty, 1e-4):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 3: High penalty
    remaining = _remaining_time()
    if remaining > 4:
        penalty = 500.0
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-5),
                          jac=lambda p: _objective_grad(p, penalty, 1e-5)[1],
                          options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < _objective(params, penalty, 1e-5):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 4: Very high penalty, tight tolerance
    remaining = _remaining_time()
    if remaining > 2:
        penalty = 5000.0
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-6),
                          jac=lambda p: _objective_grad(p, penalty, 1e-6)[1],
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < _objective(params, penalty, 1e-6):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 5: Ultra-high penalty for final polish
    remaining = _remaining_time()
    if remaining > 1.5:
        penalty = 50000.0
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-7),
                          jac=lambda p: _objective_grad(p, penalty, 1e-7)[1],
                          options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < _objective(params, penalty, 1e-7):
                params = res.x
                full_obj = _objective(params, penalty, 1e-7)
                if full_obj < _objective(best_params, penalty, 1e-7):
                    best_params = params.copy()
                    best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 6: Physics relaxation for final adjustments
    remaining = _remaining_time()
    if remaining > 1.0:
        xyz_temp = best_params.reshape(N, 3)
        xyz_phys = _physics_relax(xyz_temp, n_steps=300, dt=0.002, k_repel=30.0, k_attract=3.0)
        if _validate(xyz_phys):
            phys_score = np.sum(xyz_phys[:, 2])
            if phys_score > best_score:
                best_score = phys_score
                best_params = xyz_phys.flatten().copy()
        else:
            xyz_phys = _enforce_validity(xyz_phys)
            if _validate(xyz_phys):
                phys_score = np.sum(xyz_phys[:, 2])
                if phys_score > best_score:
                    best_score = phys_score
                    best_params = xyz_phys.flatten().copy()
    
    return best_params.reshape(N, 3), best_score

def construct():
    global _GLOBAL_START, _GLOBAL_DEADLINE
    _GLOBAL_START = time.time()
    _GLOBAL_DEADLINE = _GLOBAL_START + 145.0  # 5s safety margin before 150s kill
    
    best_xyz = _initial_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Generate diverse initial configurations
    inits = [
        _initial_grid(),
        _initial_hex(),
        _initial_corner_weighted(0),
        _initial_corner_weighted(1),
        _initial_ring(0),
        _initial_ring(1),
        _initial_corner_cluster()
    ]
    
    # Optimize from each
    for xyz0 in inits:
        remaining = _remaining_time()
        if remaining < 12.0:
            break
        xyz_opt, score = _optimize_single(xyz0)
        
        if not _validate(xyz_opt):
            xyz_fix = xyz_opt.copy()
            for _ in range(50):
                if _validate(xyz_fix):
                    break
                xyz_fix[:, 2] *= 0.98
                xyz_fix = _fix_boundaries(xyz_fix)
            xyz_opt = xyz_fix
        
        if not _validate(xyz_opt):
            continue
        
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
    
    # Local refinement from the best solution found
    remaining = _remaining_time()
    if remaining > 12.0:
        rng = np.random.default_rng(42)
        # Try multiple perturbation scales
        for scale in [0.003, 0.005, 0.008, 0.01, 0.015, 0.02]:
            if remaining < 4.0:
                break
            # Perturb the best solution
            perturbed = best_xyz.copy()
            noise = rng.normal(0, scale, perturbed.shape)
            # Do not perturb radii too much
            noise[:, 2] *= 0.1
            perturbed += noise
            perturbed = _fix_boundaries(perturbed)
            
            xyz_opt, score = _optimize_single(perturbed)
            
            if not _validate(xyz_opt):
                xyz_fix = xyz_opt.copy()
                for _ in range(50):
                    if _validate(xyz_fix):
                        break
                    xyz_fix[:, 2] *= 0.98
                    xyz_fix = _fix_boundaries(xyz_fix)
                xyz_opt = xyz_fix
            
            if not _validate(xyz_opt):
                continue
            
            score = np.sum(xyz_opt[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_opt
            remaining = _remaining_time()
    
    # Final high-precision polish: run a few more optimization rounds with very tight tolerances
    remaining = _remaining_time()
    if remaining > 3.0:
        params = best_xyz.flatten().copy()
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
            bounds.append((0, 1))
            bounds.append((0, 0.5))
        
        for penalty, eps, maxiter in [(100000, 1e-8, 200), (200000, 1e-9, 200), (500000, 1e-10, 150)]:
            remaining = _remaining_time()
            if remaining < 1.0:
                break
            try:
                res = minimize(_objective, params, method='L-BFGS-B',
                              bounds=bounds, args=(penalty, eps),
                              jac=lambda p: _objective_grad(p, penalty, eps)[1],
                              options={'maxiter': maxiter, 'ftol': 1e-18, 'gtol': 1e-16})
                # Verify improvement with full objective
                if res.fun < _objective(params, penalty, eps):
                    new_params = res.x
                    if _objective(new_params, penalty, eps) < _objective(params, penalty, eps):
                        params = new_params
                        # Check validity before accepting
                        candidate = params.reshape(N, 3)
                        if _validate(candidate):
                            new_score = np.sum(candidate[:, 2])
                            if new_score > best_score:
                                best_score = new_score
                                best_xyz = candidate
            except Exception:
                pass
        
        # Final physics relaxation
        remaining = _remaining_time()
        if remaining > 1.5:
            xyz_phys = _physics_relax(best_xyz, n_steps=500, dt=0.001, k_repel=40.0, k_attract=5.0)
            if _validate(xyz_phys):
                phys_score = np.sum(xyz_phys[:, 2])
                if phys_score > best_score:
                    best_score = phys_score
                    best_xyz = xyz_phys
            else:
                xyz_phys = _enforce_validity(xyz_phys)
                if _validate(xyz_phys):
                    phys_score = np.sum(xyz_phys[:, 2])
                    if phys_score > best_score:
                        best_score = phys_score
                        best_xyz = xyz_phys
    
    # Final validation
    if not _validate(best_xyz):
        best_xyz = _initial_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
