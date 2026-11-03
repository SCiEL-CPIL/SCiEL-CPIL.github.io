"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
_GLOBAL_START = None
_GLOBAL_DEADLINE = None

# EVOLVE-BLOCK-START

def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0) using log-sum-exp."""
    return np.logaddexp(0, z / eps) * eps

def _softplus_grad(z, eps=1e-4):
    """Derivative of softplus: sigmoid(z/eps)."""
    return 1.0 / (1.0 + np.exp(-z / eps))

def _get_contact_edges(x, y, r, margin=0.02):
    """
    Use Delaunay triangulation to find candidate contact pairs.
    Returns list of (i, j) pairs that are close enough to potentially overlap.
    """
    n = len(x)
    points = np.column_stack([x, y])
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a+1, 3):
                    i, j = simplex[a], simplex[b]
                    edges.add((min(i,j), max(i,j)))
        # Also add boundary-adjacent pairs (circles near walls)
        # And ensure all pairs within a certain distance are included
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] + margin:
                    edges.add((i, j))
        return list(edges)
    except Exception:
        # Fallback: all pairs
        return [(i, j) for i in range(n) for j in range(i+1, n)]

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
    
    # Pairwise overlaps using contact edges from Delaunay
    edges = _get_contact_edges(x, y, r)
    if len(edges) > 0:
        idx_i = np.array([e[0] for e in edges])
        idx_j = np.array([e[1] for e in edges])
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
    return -np.sum(r) + penalty_weight * penalty

def _objective_grad(params, penalty_weight, smooth_eps=1e-4):
    """
    Analytic gradient of the objective for L-BFGS-B.
    """
    x = params[0::3].copy()
    y = params[1::3].copy()
    r = params[2::3].copy()
    
    n = len(r)
    grad = np.zeros(3*n)
    
    # Objective gradient: d(-sum(r))/d(r_i) = -1
    grad[2::3] += -1.0
    
    # Boundary penalty gradients
    # lx = r - x
    slx = _softplus(r - x, smooth_eps)
    glx = _softplus_grad(r - x, smooth_eps)
    # d/dx_i: -2*slx*glx ; d/dr_i: 2*slx*glx
    grad[0::3] += penalty_weight * (-2*slx*glx)
    grad[2::3] += penalty_weight * (2*slx*glx)
    
    # ux = x + r - 1
    sux = _softplus(x + r - 1.0, smooth_eps)
    gux = _softplus_grad(x + r - 1.0, smooth_eps)
    grad[0::3] += penalty_weight * (2*sux*gux)
    grad[2::3] += penalty_weight * (2*sux*gux)
    
    # ly = r - y
    sly = _softplus(r - y, smooth_eps)
    gly = _softplus_grad(r - y, smooth_eps)
    grad[1::3] += penalty_weight * (-2*sly*gly)
    grad[2::3] += penalty_weight * (2*sly*gly)
    
    # uy = y + r - 1
    suy = _softplus(y + r - 1.0, smooth_eps)
    guy = _softplus_grad(y + r - 1.0, smooth_eps)
    grad[1::3] += penalty_weight * (2*suy*guy)
    grad[2::3] += penalty_weight * (2*suy*guy)
    
    # Negative radius penalty
    sneg = _softplus(-r, smooth_eps)
    gneg = _softplus_grad(-r, smooth_eps)
    grad[2::3] += penalty_weight * (-2*sneg*gneg)
    
    # Pairwise overlap gradients using contact edges
    edges = _get_contact_edges(x, y, r)
    if len(edges) > 0:
        idx_i = np.array([e[0] for e in edges])
        idx_j = np.array([e[1] for e in edges])
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        s = _softplus(overlap, smooth_eps)
        g = _softplus_grad(overlap, smooth_eps)
        
        # d(overlap)/dx_i = -dx/dist, d(overlap)/dx_j = dx/dist
        # d(overlap)/dy_i = -dy/dist, d(overlap)/dy_j = dy/dist
        # d(overlap)/dr_i = 1, d(overlap)/dr_j = 1
        
        factor = 2 * s * g * penalty_weight
        inv_dist = 1.0 / dist
        
        for k in range(len(idx_i)):
            i, j = idx_i[k], idx_j[k]
            f = factor[k]
            # dx contribution
            grad[i*3] += f * (-dx[k] * inv_dist[k])
            grad[j*3] += f * (dx[k] * inv_dist[k])
            # dy contribution
            grad[i*3+1] += f * (-dy[k] * inv_dist[k])
            grad[j*3+1] += f * (dy[k] * inv_dist[k])
            # dr contribution
            grad[i*3+2] += f * 1.0
            grad[j*3+2] += f * 1.0
    
    return grad

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

def _pt_mc_search(initial_states, temps, steps, rng):
    """
    Parallel Tempering Monte Carlo search.
    """
    n_replicas = len(temps)
    states = [s.copy() for s in initial_states]
    energies = []
    
    for i, s in enumerate(states):
        x, y, r = s[:, 0], s[:, 1], s[:, 2]
        
        lx = np.maximum(0, r - x)
        ux = np.maximum(0, x + r - 1)
        ly = np.maximum(0, r - y)
        uy = np.maximum(0, y + r - 1)
        bound_pen = np.sum(lx**2 + ux**2 + ly**2 + uy**2)
        
        n = len(r)
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = np.maximum(0, r[idx_i] + r[idx_j] - dist)
        overlap_pen = np.sum(overlap**2)
        
        neg_r_pen = np.sum(np.maximum(0, -r)**2)
        
        penalty = bound_pen + overlap_pen + neg_r_pen
        energy = -np.sum(r) + 100.0 * penalty
        energies.append(energy)
        
    best_energy = min(energies)
    best_state = states[energies.index(best_energy)].copy()
    
    for step in range(steps):
        if _remaining_time() < 5.0:
            break
            
        i = rng.integers(0, n_replicas)
        
        new_state = states[i].copy()
        
        scale_pos = 0.01 * temps[i]
        new_state[:, 0] += rng.normal(0, scale_pos, N)
        new_state[:, 1] += rng.normal(0, scale_pos, N)
        
        scale_r = 0.005 * temps[i]
        new_state[:, 2] += rng.normal(0, scale_r, N)
        
        new_state[:, 2] = np.maximum(new_state[:, 2], 0.01)
        new_state[:, 0] = np.clip(new_state[:, 0], -0.1, 1.1)
        new_state[:, 1] = np.clip(new_state[:, 1], -0.1, 1.1)
        
        x, y, r = new_state[:, 0], new_state[:, 1], new_state[:, 2]
        
        lx = np.maximum(0, r - x)
        ux = np.maximum(0, x + r - 1)
        ly = np.maximum(0, r - y)
        uy = np.maximum(0, y + r - 1)
        bound_pen = np.sum(lx**2 + ux**2 + ly**2 + uy**2)
        
        n = len(r)
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = np.maximum(0, r[idx_i] + r[idx_j] - dist)
        overlap_pen = np.sum(overlap**2)
        
        neg_r_pen = np.sum(np.maximum(0, -r)**2)
        
        penalty = bound_pen + overlap_pen + neg_r_pen
        new_energy = -np.sum(r) + 100.0 * penalty
        
        delta_e = new_energy - energies[i]
        if delta_e < 0 or rng.random() < np.exp(-delta_e / temps[i]):
            states[i] = new_state
            energies[i] = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_state = new_state.copy()
        
        if n_replicas > 1:
            j = rng.integers(0, n_replicas - 1)
            k = j + 1
            
            inv_t_j = 1.0 / temps[j]
            inv_t_k = 1.0 / temps[k]
            
            delta_swap = (energies[j] - energies[k]) * (inv_t_k - inv_t_j)
            
            if delta_swap < 0 or rng.random() < np.exp(-delta_swap):
                states[j], states[k] = states[k], states[j]
                energies[j], energies[k] = energies[k], energies[j]
                
                if energies[j] < best_energy:
                    best_energy = energies[j]
                    best_state = states[j].copy()
                if energies[k] < best_energy:
                    best_energy = energies[k]
                    best_state = states[k].copy()
    
    return best_state

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
    Specific initialization placing larger circles in corners and edges.
    """
    positions = []
    positions.append([0.1, 0.1, 0.12])
    positions.append([0.9, 0.1, 0.12])
    positions.append([0.1, 0.9, 0.12])
    positions.append([0.9, 0.9, 0.12])
    
    positions.append([0.5, 0.08, 0.10])
    positions.append([0.5, 0.92, 0.10])
    positions.append([0.08, 0.5, 0.10])
    positions.append([0.92, 0.5, 0.10])
    
    positions.append([0.5, 0.5, 0.12])
    
    rng = np.random.default_rng(123)
    while len(positions) < N:
        x = rng.uniform(0.15, 0.85)
        y = rng.uniform(0.15, 0.85)
        r = rng.uniform(0.05, 0.08)
        positions.append([x, y, r])
        
    return np.array(positions[:N])

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
    
    phases = [
        (5.0, 1e-3, 300, 1e-12, 1e-10, 10),
        (50.0, 1e-4, 400, 1e-14, 1e-12, 8),
        (500.0, 1e-5, 400, 1e-15, 1e-13, 5),
        (5000.0, 1e-6, 300, 1e-15, 1e-14, 2),
        (50000.0, 1e-7, 200, 1e-16, 1e-15, 1),
    ]
    
    for penalty, eps, maxiter, ftol, gtol, min_time in phases:
        remaining = _remaining_time()
        if remaining < min_time:
            continue
        try:
            res = minimize(_objective, params, jac=_objective_grad,
                          method='L-BFGS-B', bounds=bounds,
                          args=(penalty, eps),
                          options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
            if res.fun < _objective(params, penalty, eps):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    return best_params.reshape(N, 3), best_score

def construct():
    global _GLOBAL_START, _GLOBAL_DEADLINE
    _GLOBAL_START = time.time()
    _GLOBAL_DEADLINE = _GLOBAL_START + 145.0
    
    best_xyz = _initial_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    inits = [
        _initial_grid(),
        _initial_hex(),
        _initial_corner_weighted(0),
        _initial_corner_weighted(1),
        _initial_ring(0),
        _initial_ring(1),
        _initial_corner_cluster()
    ]
    
    for xyz0 in inits:
        remaining = _remaining_time()
        if remaining < 15.0:
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
    if remaining > 15.0:
        rng = np.random.default_rng(42)
        for scale in [0.005, 0.01, 0.02, 0.03]:
            if remaining < 5.0:
                break
            perturbed = best_xyz.copy()
            noise = rng.normal(0, scale, perturbed.shape)
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
    
    if not _validate(best_xyz):
        best_xyz = _initial_grid()
    
    # Parallel Tempering Monte Carlo refinement
    remaining = _remaining_time()
    if remaining > 20.0:
        rng = np.random.default_rng(99)
        replicas = [best_xyz]
        perturbed = best_xyz.copy()
        perturbed += rng.normal(0, 0.01, perturbed.shape)
        perturbed = _fix_boundaries(perturbed)
        replicas.append(perturbed)
        
        temps = [1.0, 2.0, 4.0]
        
        if len(replicas) < 3:
            rep3 = best_xyz.copy()
            rep3 += rng.normal(0, 0.02, rep3.shape)
            rep3 = _fix_boundaries(rep3)
            replicas.append(rep3)
            
        replicas = replicas[:3]
        
        steps = 50000
        
        pt_result = _pt_mc_search(replicas, temps, steps, rng)
        
        if _validate(pt_result):
            pt_score = np.sum(pt_result[:, 2])
            if pt_score > best_score:
                best_xyz = pt_result
                best_score = pt_score
    
    return best_xyz

# EVOLVE-BLOCK-END
