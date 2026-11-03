"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _softplus(z, beta=1.0):
    """Numerically stable softplus: log(1+exp(beta*z))/beta."""
    z = np.asarray(z, dtype=float)
    return np.where(z > 0, z + np.log1p(np.exp(-beta * z)) / beta,
                    np.log1p(np.exp(beta * z)) / beta)

def _softplus_grad(z, beta=1.0):
    """Gradient of softplus w.r.t. z."""
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-beta * z))

def _objective_grad(params, beta=1.0):
    """
    Smooth objective: -sum(r) + softplus penalties for all constraint violations.
    Returns (value, gradient).
    params: flat array [x0,y0,r0, x1,y1,r1, ...] of length 3*N
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    val = -np.sum(r)
    grad = np.zeros_like(params)
    
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r, r >= 0
    # Violation if (r - x) > 0, (x + r - 1) > 0, etc.
    
    # x >= r  => violation = r - x
    v1 = r - x
    val += np.sum(_softplus(v1, beta))
    g1 = _softplus_grad(v1, beta)
    grad[2::3] += g1       # d/d r_i of (r_i - x_i) = 1
    grad[0::3] -= g1       # d/d x_i of (r_i - x_i) = -1
    
    # x <= 1 - r  => violation = x + r - 1
    v2 = x + r - 1.0
    val += np.sum(_softplus(v2, beta))
    g2 = _softplus_grad(v2, beta)
    grad[2::3] += g2       # d/d r_i of (x_i + r_i - 1) = 1
    grad[0::3] += g2       # d/d x_i of (x_i + r_i - 1) = 1
    
    # y >= r  => violation = r - y
    v3 = r - y
    val += np.sum(_softplus(v3, beta))
    g3 = _softplus_grad(v3, beta)
    grad[2::3] += g3
    grad[1::3] -= g3
    
    # y <= 1 - r  => violation = y + r - 1
    v4 = y + r - 1.0
    val += np.sum(_softplus(v4, beta))
    g4 = _softplus_grad(v4, beta)
    grad[2::3] += g4
    grad[1::3] += g4
    
    # r >= 0  => violation = -r
    v5 = -r
    val += np.sum(_softplus(v5, beta))
    g5 = _softplus_grad(v5, beta)
    grad[2::3] -= g5
    
    # Pairwise non-overlap: r_i + r_j <= dist_ij
    # violation = r_i + r_j - dist_ij
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist = np.sqrt(dx*dx + dy*dy + 1e-12)
    v6 = r[idx_i] + r[idx_j] - dist
    
    val += np.sum(_softplus(v6, beta))
    g6 = _softplus_grad(v6, beta)
    
    # d/d x_i: g6 * (-dx/dist) for each pair where i is idx_i
    # d/d x_j: g6 * (dx/dist) for each pair where j is idx_j
    # d/d r_i: g6 * 1
    # d/d r_j: g6 * 1
    
    nx = np.zeros(n)
    ny = np.zeros(n)
    nr = np.zeros(n)
    
    inv_dist = g6 / dist
    nx[idx_i] -= inv_dist * dx
    nx[idx_j] += inv_dist * dx
    ny[idx_i] -= inv_dist * dy
    ny[idx_j] += inv_dist * dy
    nr[idx_i] += g6
    nr[idx_j] += g6
    
    grad[0::3] += nx
    grad[1::3] += ny
    grad[2::3] += nr
    
    return val, grad

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

def _enforce_validity(xyz, max_iter=200):
    """Iteratively fix violations by adjusting positions and radii."""
    xyz = xyz.copy()
    for _ in range(max_iter):
        x, y, r = xyz[:,0], xyz[:,1], xyz[:,2]
        
        # Clip radii to be non-negative and not too large
        r = np.maximum(r, 1e-8)
        r = np.minimum(r, 0.5)
        
        # Project into square
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        
        # Check for overlaps and resolve
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist - 1e-10:
                    # Push apart
                    if dist < 1e-10:
                        # Random small displacement
                        angle = np.random.uniform(0, 2*np.pi)
                        push = (min_dist - dist) * 0.5
                        x[i] += push * np.cos(angle)
                        y[i] += push * np.sin(angle)
                        x[j] -= push * np.cos(angle)
                        y[j] -= push * np.sin(angle)
                    else:
                        push = (min_dist - dist) * 0.5 + 1e-8
                        dx_norm = dx / dist
                        dy_norm = dy / dist
                        x[i] += push * dx_norm
                        y[i] += push * dy_norm
                        x[j] -= push * dx_norm
                        y[j] -= push * dy_norm
                    changed = True
        
        xyz[:,0] = np.clip(x, 0, 1)
        xyz[:,1] = np.clip(y, 0, 1)
        xyz[:,2] = r
        
        if not changed:
            break
    
    # Final check: if still overlapping, shrink radii
    if not _validate(xyz):
        for scale in np.linspace(0.95, 0.5, 20):
            xyz_test = xyz.copy()
            xyz_test[:,2] *= scale
            if _validate(xyz_test):
                xyz = xyz_test
                break
    
    return xyz

def _physics_relax(xyz, deadline, iterations=500):
    """
    Physics-based relaxation: apply forces to push overlapping circles apart
    and attract them to fill space, then increase radii.
    """
    xyz = xyz.copy()
    x, y, r = xyz[:,0].copy(), xyz[:,1].copy(), xyz[:,2].copy()
    
    best_r = r.copy()
    best_score = np.sum(r)
    
    dt = 0.01
    for it in range(iterations):
        if time.time() > deadline:
            break
        
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Repulsion between overlapping circles
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    if dist < 1e-10:
                        angle = np.random.uniform(0, 2*np.pi)
                        f = 10.0
                        fx[i] += f * np.cos(angle)
                        fy[i] += f * np.sin(angle)
                        fx[j] -= f * np.cos(angle)
                        fy[j] -= f * np.sin(angle)
                    else:
                        f = 5.0 * (min_dist - dist) / min_dist
                        fx[i] += f * dx / dist
                        fy[i] += f * dy / dist
                        fx[j] -= f * dx / dist
                        fy[j] -= f * dy / dist
        
        # Boundary forces
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += 10.0 * (r[i] - x[i])
            if x[i] > 1 - r[i]:
                fx[i] -= 10.0 * (x[i] - (1 - r[i]))
            if y[i] < r[i]:
                fy[i] += 10.0 * (r[i] - y[i])
            if y[i] > 1 - r[i]:
                fy[i] -= 10.0 * (y[i] - (1 - r[i]))
        
        # Update positions
        x += dt * fx
        y += dt * fy
        
        # Project into square
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        
        # Try to increase radii slightly
        if it % 10 == 5:
            new_r = r * 1.002
            new_r = np.minimum(new_r, 0.5)
            # Check if still valid
            test_xyz = np.column_stack([x, y, new_r])
            if _validate(test_xyz):
                r = new_r
                best_r = r.copy()
                best_score = np.sum(r)
        
        xyz = np.column_stack([x, y, r])
    
    return xyz, best_score

def _polish_analytic(xyz, deadline):
    """
    Final high-precision polish using L-BFGS-B with very tight tolerances
    and smooth softplus objective.
    """
    params = xyz.flatten().copy()
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    # Use progressively smaller beta for sharper constraint enforcement
    for beta in [1.0, 5.0, 20.0, 50.0]:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _objective_grad,
                params,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                args=(beta,),
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            if res.x is not None:
                params = res.x
        except Exception:
            pass
    
    xyz_opt = params.reshape(N, 3)
    return xyz_opt

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

def _optimize_single(xyz0, deadline):
    """Optimize from a single initial config."""
    xyz = xyz0.copy()
    
    # Phase 1: Physics relaxation to get a valid starting point
    remaining = deadline - time.time()
    if remaining > 2:
        xyz_phys, _ = _physics_relax(xyz, deadline - 1.0)
        if _validate(xyz_phys):
            xyz = xyz_phys
    
    # Phase 2: Smooth gradient-based optimization
    remaining = deadline - time.time()
    if remaining > 3:
        xyz_opt = _polish_analytic(xyz, deadline - 0.5)
        if _validate(xyz_opt):
            xyz = xyz_opt
        else:
            xyz = _enforce_validity(xyz_opt)
    
    # Phase 3: Another physics + polish cycle
    remaining = deadline - time.time()
    if remaining > 3:
        xyz_phys, _ = _physics_relax(xyz, deadline - 1.0)
        if _validate(xyz_phys):
            xyz = xyz_phys
        xyz_opt = _polish_analytic(xyz, deadline - 0.5)
        if _validate(xyz_opt):
            xyz = xyz_opt
        else:
            xyz = _enforce_validity(xyz_opt)
    
    return xyz, np.sum(xyz[:, 2])

def construct():
    global_start = time.time()
    deadline = global_start + 145.0  # 5s safety margin before 150s kill
    
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
    
    rng = np.random.default_rng(123)
    for s in range(4):
        x = rng.uniform(0.05, 0.95, N)
        y = rng.uniform(0.05, 0.95, N)
        r = rng.uniform(0.05, 0.12, N)
        cfg = np.column_stack([x, y, r])
        cfg = _enforce_validity(cfg)
        inits.append(cfg)
    
    # Optimize from each
    for xyz0 in inits:
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
        time_budget = deadline - 1.0
        xyz_opt, score = _optimize_single(xyz0, time_budget)
        
        if not _validate(xyz_opt):
            xyz_opt = _enforce_validity(xyz_opt)
        
        if not _validate(xyz_opt):
            continue
        
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
    
    # Final validation
    if not _validate(best_xyz):
        best_xyz = _enforce_validity(best_xyz)
    if not _validate(best_xyz):
        best_xyz = _initial_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
