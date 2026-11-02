"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

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

def _resolve_overlaps(xyz, max_iter=100):
    """Iteratively resolve overlaps by moving circles apart and shrinking radii."""
    xyz = xyz.copy()
    for _ in range(max_iter):
        moved = False
        for i in range(N):
            for j in range(i+1, N):
                dx = xyz[j,0] - xyz[i,0]
                dy = xyz[j,1] - xyz[i,1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = xyz[i,2] + xyz[j,2]
                if dist < min_dist - 1e-9:
                    overlap = min_dist - dist
                    if dist < 1e-9:
                        angle = np.random.uniform(0, 2*np.pi)
                        dx, dy = np.cos(angle), np.sin(angle)
                        dist = 1e-9
                    else:
                        dx, dy = dx/dist, dy/dist
                    push = overlap * 0.5
                    xyz[i,0] -= dx * push
                    xyz[i,1] -= dy * push
                    xyz[j,0] += dx * push
                    xyz[j,1] += dy * push
                    # Shrink both radii slightly
                    xyz[i,2] *= 0.995
                    xyz[j,2] *= 0.995
                    moved = True
        xyz = _fix_boundaries(xyz)
        if not moved:
            break
    return xyz

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

def _simulated_annealing(xyz0, deadline, seed=0):
    """Simulated annealing for circle packing."""
    rng = np.random.default_rng(seed)
    current = xyz0.copy()
    best = current.copy()
    best_score = np.sum(current[:, 2])
    
    T0 = 0.05
    T_end = 0.0001
    alpha = 0.995
    T = T0
    step_size = 0.03
    
    while time.time() < deadline:
        # Generate perturbation
        xyz_new = current.copy()
        
        # Move 1-3 circles
        n_move = rng.integers(1, 4)
        idx = rng.choice(N, size=n_move, replace=False)
        for i in idx:
            xyz_new[i, 0] += rng.normal(0, step_size)
            xyz_new[i, 1] += rng.normal(0, step_size)
            # Occasionally adjust radius
            if rng.random() < 0.3:
                xyz_new[i, 2] += rng.normal(0, step_size * 0.5)
        
        # Fix boundaries and resolve overlaps
        xyz_new = _fix_boundaries(xyz_new)
        xyz_new = _resolve_overlaps(xyz_new, max_iter=20)
        
        if not _validate(xyz_new):
            # Reject invalid
            T *= alpha
            step_size *= 0.999
            continue
        
        score_new = np.sum(xyz_new[:, 2])
        score_cur = np.sum(current[:, 2])
        delta = score_new - score_cur
        
        # Accept if better or with probability exp(-delta/T)
        if delta > 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current = xyz_new
            if score_new > best_score:
                best_score = score_new
                best = xyz_new.copy()
        
        T *= alpha
        step_size *= 0.9995
        
        if T < T_end:
            break
    
    return best, best_score

def _gradient_refinement(xyz0, deadline):
    """Gradient-based refinement using L-BFGS-B with penalty method."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    def objective(params, penalty_weight, smooth_eps=1e-4):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        penalty = 0.0
        lx = r - x
        ux = x + r - 1.0
        ly = r - y
        uy = y + r - 1.0
        
        penalty += np.sum(np.maximum(lx, 0)**2)
        penalty += np.sum(np.maximum(ux, 0)**2)
        penalty += np.sum(np.maximum(ly, 0)**2)
        penalty += np.sum(np.maximum(uy, 0)**2)
        penalty += np.sum(np.maximum(-r, 0)**2)
        
        n = len(r)
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(np.maximum(overlap, 0)**2)
        
        return -np.sum(r) + penalty_weight * penalty
    
    # Progressive penalty
    for penalty in [1.0, 10.0, 100.0, 1000.0]:
        remaining = deadline - time.time()
        if remaining < 2:
            break
        try:
            res = minimize(objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4),
                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < objective(params, penalty, 1e-4):
                params = res.x
                if _validate(params.reshape(N, 3)):
                    best_params = params.copy()
                    best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    return best_params.reshape(N, 3), best_score

def construct():
    global_start = time.time()
    deadline = global_start + 145.0
    
    best_xyz = _initial_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Phase 1: Simulated annealing from multiple starts
    inits = [
        _initial_grid(),
        _initial_hex(),
        _initial_corner_weighted(0),
        _initial_corner_weighted(1),
    ]
    
    for idx, xyz0 in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 15.0:
            break
        sa_deadline = deadline - 5.0
        xyz_sa, score_sa = _simulated_annealing(xyz0, sa_deadline, seed=idx*100)
        
        if score_sa > best_score:
            best_score = score_sa
            best_xyz = xyz_sa
    
    # Phase 2: Gradient refinement from best SA result
    remaining = deadline - time.time()
    if remaining > 10.0:
        grad_deadline = deadline - 3.0
        xyz_grad, score_grad = _gradient_refinement(best_xyz, grad_deadline)
        
        if score_grad > best_score:
            best_score = score_grad
            best_xyz = xyz_grad
    
    # Phase 3: More SA refinement from gradient result
    remaining = deadline - time.time()
    if remaining > 10.0:
        sa_deadline2 = deadline - 3.0
        xyz_sa2, score_sa2 = _simulated_annealing(best_xyz, sa_deadline2, seed=999)
        
        if score_sa2 > best_score:
            best_score = score_sa2
            best_xyz = xyz_sa2
    
    # Final validation
    if not _validate(best_xyz):
        best_xyz = _fix_boundaries(best_xyz)
        best_xyz = _resolve_overlaps(best_xyz)
        if not _validate(best_xyz):
            best_xyz = _initial_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
