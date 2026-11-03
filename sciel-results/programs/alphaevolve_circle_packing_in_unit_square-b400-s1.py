"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

_start_time = time.time()
_TIME_LIMIT = 145  # seconds, safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0)."""
    return np.logaddexp(0, z / eps) * eps


def _sigmoid(z, eps=1e-4):
    """Smooth derivative of softplus."""
    return 1.0 / (1.0 + np.exp(-z / eps))


def _validate(xyz, tol=1e-7):
    """Check if configuration is valid: circles inside unit square and non-overlapping."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    if np.any(r < -tol):
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


def _enforce_validity(params):
    """
    Enforce all constraints by shrinking radii and clipping positions.
    Returns a valid configuration.
    """
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)
    
    # Clip radii to fit within walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    
    # Iteratively resolve overlaps
    for _ in range(300):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 0
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            # Shrink the smaller circle preferentially
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
        # Re-clip to walls
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    
    return p


def _objective_grad(params, penalty_weight, smooth_eps=1e-4):
    """
    Objective and analytic gradient for soft-plus penalty method.
    Returns (value, gradient).
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    val = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0  # d(-sum r)/dr

    # Boundary violations
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0

    # Left x boundary
    sp_lx = _softplus(lx, smooth_eps)
    sg_lx = _sigmoid(lx, smooth_eps)
    grad[0::3] += -2.0 * penalty_weight * sp_lx * sg_lx
    grad[2::3] += 2.0 * penalty_weight * sp_lx * sg_lx

    # Right x boundary
    sp_ux = _softplus(ux, smooth_eps)
    sg_ux = _sigmoid(ux, smooth_eps)
    grad[0::3] += 2.0 * penalty_weight * sp_ux * sg_ux
    grad[2::3] += 2.0 * penalty_weight * sp_ux * sg_ux

    # Left y boundary
    sp_ly = _softplus(ly, smooth_eps)
    sg_ly = _sigmoid(ly, smooth_eps)
    grad[1::3] += -2.0 * penalty_weight * sp_ly * sg_ly
    grad[2::3] += 2.0 * penalty_weight * sp_ly * sg_ly

    # Right y boundary
    sp_uy = _softplus(uy, smooth_eps)
    sg_uy = _sigmoid(uy, smooth_eps)
    grad[1::3] += 2.0 * penalty_weight * sp_uy * sg_uy
    grad[2::3] += 2.0 * penalty_weight * sp_uy * sg_uy

    # Negative radius
    sp_nr = _softplus(-r, smooth_eps)
    sg_nr = _sigmoid(-r, smooth_eps)
    grad[2::3] += -2.0 * penalty_weight * sp_nr * sg_nr

    val += penalty_weight * (np.sum(sp_lx**2) + np.sum(sp_ux**2) +
                             np.sum(sp_ly**2) + np.sum(sp_uy**2) +
                             np.sum(sp_nr**2))

    # Pairwise overlaps (vectorized)
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    sp_ov = _softplus(overlap, smooth_eps)
    sg_ov = _sigmoid(overlap, smooth_eps)

    factor = 2.0 * penalty_weight * sp_ov * sg_ov
    inv_dist = 1.0 / dist

    np.add.at(grad[0::3], idx_i, -factor * dx * inv_dist)
    np.add.at(grad[0::3], idx_j, factor * dx * inv_dist)
    np.add.at(grad[1::3], idx_i, -factor * dy * inv_dist)
    np.add.at(grad[1::3], idx_j, factor * dy * inv_dist)
    np.add.at(grad[2::3], idx_i, factor)
    np.add.at(grad[2::3], idx_j, factor)

    val += penalty_weight * np.sum(sp_ov**2)

    return val, grad


def _polish_analytic(xyz, time_budget=5.0):
    """
    High-precision polish using analytic gradient L-BFGS-B.
    """
    start_time = time.time()
    params = xyz.flatten().copy()
    best_params = params.copy()
    best_val, _ = _objective_grad(best_params, 1000.0)
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 0.5))
    
    for penalty_weight in [1000.0, 5000.0, 20000.0]:
        if time.time() - start_time > time_budget:
            break
        
        def obj(p):
            v, _ = _objective_grad(p, penalty_weight)
            return v
        
        def grad(p):
            _, g = _objective_grad(p, penalty_weight)
            return g
        
        try:
            result = minimize(obj, best_params, jac=grad, method='L-BFGS-B', 
                             bounds=bounds,
                             options={'maxiter': 2000, 'ftol': 1e-14, 
                                      'gtol': 1e-12, 'maxfun': 10000})
            obj_val, _ = _objective_grad(result.x, penalty_weight)
            if obj_val < best_val:
                best_params = result.x
                best_val = obj_val
        except Exception:
            pass
    
    return best_params.reshape(N, 3)


def _physics_relax(xyz, time_limit=3.0, grow_rate=0.001):
    """
    Force-directed physics relaxation that also grows radii over time.
    """
    start_time = time.time()
    xyz = xyz.copy().astype(float)
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()
    
    best_score = np.sum(r)
    best_xyz = xyz.copy()
    
    dt = 0.005
    
    for iteration in range(5000):
        if time.time() - start_time > time_limit:
            break
        
        # Grow radii slowly
        r *= (1.0 + grow_rate)
        
        # Clip radii to fit within walls
        wall_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, wall_max)
        r = np.maximum(r, 1e-6)
        
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Pairwise repulsion for overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(max(dist_sq, 1e-12))
                
                # Repulsion if overlapping
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    # Strong repulsion
                    force = 50.0 * overlap
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
        
        # Boundary forces
        for i in range(N):
            # Left wall
            if x[i] < r[i]:
                fx[i] += 50.0 * (r[i] - x[i])
            # Right wall
            if x[i] > 1 - r[i]:
                fx[i] -= 50.0 * (x[i] - (1 - r[i]))
            # Bottom wall
            if y[i] < r[i]:
                fy[i] += 50.0 * (r[i] - y[i])
            # Top wall
            if y[i] > 1 - r[i]:
                fy[i] -= 50.0 * (y[i] - (1 - r[i]))
        
        # Update positions with damping
        damping = 0.5
        x += dt * fx * damping
        y += dt * fy * damping
        
        # Clip positions
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        
        # Update xyz
        xyz[:, 0] = x
        xyz[:, 1] = y
        xyz[:, 2] = r
        
        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_xyz = xyz.copy()
    
    return best_xyz


def construct():
    """
    Pack 26 circles in the unit square, maximizing the sum of radii.
    Uses multi-start strategy with physics relaxation and analytic polishing.
    """
    start_time = time.time()
    deadline = _TIME_LIMIT - 5.0  # safety margin
    
    best_result = None
    best_score = -np.inf
    
    # Multiple initial configurations
    configs = []
    
    # 1. 5x5 grid + 1 center
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y])
    positions.append([0.5, 0.5])
    positions = np.array(positions, dtype=np.float64)
    radii = np.ones(N, dtype=np.float64) * 0.08
    xyz = np.column_stack([positions, radii])
    configs.append(xyz)
    
    # 2. Hexagonal grid
    positions = []
    row_height = 0.17
    for row in range(6):
        y = 0.08 + row * row_height
        if row % 2 == 0:
            for col in range(5):
                x = 0.1 + col * 0.2
                positions.append([x, y])
        else:
            for col in range(5):
                x = 0.2 + col * 0.2
                positions.append([x, y])
        if len(positions) >= N:
            break
    positions = positions[:N]
    positions = np.array(positions, dtype=np.float64)
    radii = np.ones(N, dtype=np.float64) * 0.08
    xyz = np.column_stack([positions, radii])
    configs.append(xyz)
    
    # 3. Random initialization
    np.random.seed(42)
    positions = np.random.rand(N, 2) * 0.8 + 0.1
    radii = np.ones(N, dtype=np.float64) * 0.05
    xyz = np.column_stack([positions, radii])
    configs.append(xyz)
    
    # 4. Another random with different seed
    np.random.seed(123)
    positions = np.random.rand(N, 2) * 0.8 + 0.1
    radii = np.ones(N, dtype=np.float64) * 0.05
    xyz = np.column_stack([positions, radii])
    configs.append(xyz)
    
    # Process each configuration
    for config_idx, xyz in enumerate(configs):
        if _elapsed() > deadline:
            break
        
        # Physics relaxation
        xyz_relaxed = _physics_relax(xyz, time_limit=2.0)
        score_relaxed = np.sum(xyz_relaxed[:, 2])
        
        # Polish with analytic gradient
        xyz_polished = _polish_analytic(xyz_relaxed, time_budget=3.0)
        
        # Enforce validity
        xyz_final = _enforce_validity(xyz_polished)
        score_final = np.sum(xyz_final[:, 2])
        
        if score_final > best_score:
            best_score = score_final
            best_result = xyz_final
    
    # If no valid result, use the first config as fallback
    if best_result is None:
        best_result = _enforce_validity(configs[0])
    
    # Final validation and cleanup
    for i in range(N):
        best_result[i, 0] = np.clip(best_result[i, 0], 0, 1)
        best_result[i, 1] = np.clip(best_result[i, 1], 0, 1)
        max_r = min(best_result[i, 0], best_result[i, 1], 
                   1 - best_result[i, 0], 1 - best_result[i, 1])
        best_result[i, 2] = min(best_result[i, 2], max_r)
        best_result[i, 2] = max(best_result[i, 2], 1e-6)
    
    return best_result
# EVOLVE-BLOCK-END
