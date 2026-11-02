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

def _compute_energy(xyz, penalty_weight=100.0):
    """Compute energy of a configuration."""
    x, y, r = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    
    # Boundary violations
    lx = np.maximum(0, r - x)
    ux = np.maximum(0, x + r - 1)
    ly = np.maximum(0, r - y)
    uy = np.maximum(0, y + r - 1)
    bound_pen = np.sum(lx**2 + ux**2 + ly**2 + uy**2)
    
    # Overlaps
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = np.maximum(0, r[idx_i] + r[idx_j] - dist)
    overlap_pen = np.sum(overlap**2)
    
    # Negative radius
    neg_r_pen = np.sum(np.maximum(0, -r)**2)
    
    penalty = bound_pen + overlap_pen + neg_r_pen
    return -np.sum(r) + penalty_weight * penalty

def _get_contact_edges(xyz, tol=1e-4):
    """Get Delaunay contact edges (pairs of circles that are close)."""
    points = xyz[:, :2]
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a,b), max(a,b)))
        return list(edges)
    except Exception:
        return []

def _delaunay_objective(params, penalty_weight, smooth_eps=1e-4, contact_edges=None):
    """
    Objective using only contact edges from Delaunay triangulation.
    This is more efficient and focuses optimization on relevant constraints.
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
    
    # Contact edge overlaps only
    if contact_edges is not None and len(contact_edges) > 0:
        idx_i = np.array([e[0] for e in contact_edges])
        idx_j = np.array([e[1] for e in contact_edges])
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    else:
        # Fallback to full pairwise
        n = len(r)
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
    return -np.sum(r) + penalty_weight * penalty

def _pt_mc_search(initial_states, temps, max_steps, rng):
    """
    Parallel Tempering Monte Carlo search with structure-aware moves.
    initial_states: list of arrays (N, 3)
    temps: list of temperatures (floats)
    max_steps: maximum number of MC steps
    rng: numpy random generator
    """
    n_replicas = len(temps)
    states = [s.copy() for s in initial_states]
    energies = [_compute_energy(s) for s in states]
    
    best_energy = min(energies)
    best_state = states[energies.index(best_energy)].copy()
    
    # Precompute index pairs for overlap checks
    idx_i, idx_j = np.triu_indices(N, k=1)
    
    for step in range(max_steps):
        if _remaining_time() < 5.0:
            break
        
        # Randomly pick a replica to move
        i = rng.integers(0, n_replicas)
        T = temps[i]
        
        new_state = states[i].copy()
        
        # Choose move type based on temperature
        move_type = rng.integers(0, 4)
        
        if move_type == 0:
            # Random position jitter (exploration)
            scale = 0.008 * T
            new_state[:, 0] += rng.normal(0, scale, N)
            new_state[:, 1] += rng.normal(0, scale, N)
        elif move_type == 1:
            # Random radius change
            idx = rng.integers(0, N)
            delta_r = rng.normal(0, 0.005 * T)
            new_state[idx, 2] = max(0.01, new_state[idx, 2] + delta_r)
        elif move_type == 2:
            # Push apart overlapping circles
            x, y, r = new_state[:, 0], new_state[:, 1], new_state[:, 2]
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(np.maximum(dist_sq, 1e-12))
            overlaps = r[idx_i] + r[idx_j] - dist
            # Find overlapping pairs
            overlap_mask = overlaps > 0
            if np.any(overlap_mask):
                # Pick a random overlapping pair
                overlap_indices = np.where(overlap_mask)[0]
                pick = overlap_indices[rng.integers(0, len(overlap_indices))]
                ii, jj = idx_i[pick], idx_j[pick]
                d = dist[pick]
                if d > 1e-9:
                    # Push apart along the line connecting centers
                    push = overlaps[pick] * 0.5 * rng.uniform(0.5, 1.5)
                    ux, uy = dx[pick]/d, dy[pick]/d
                    new_state[ii, 0] += ux * push
                    new_state[ii, 1] += uy * push
                    new_state[jj, 0] -= ux * push
                    new_state[jj, 1] -= uy * push
        else:
            # Grow a circle if it has room
            idx = rng.integers(0, N)
            x_i, y_i, r_i = new_state[idx, 0], new_state[idx, 1], new_state[idx, 2]
            # Check max possible radius
            max_r = min(x_i, 1-x_i, y_i, 1-y_i)
            # Check distance to other circles
            other_r = np.delete(new_state[:, 2], idx)
            other_x = np.delete(new_state[:, 0], idx)
            other_y = np.delete(new_state[:, 1], idx)
            d2 = (other_x - x_i)**2 + (other_y - y_i)**2
            d = np.sqrt(np.maximum(d2, 1e-12))
            max_r = min(max_r, np.min(d - other_r) if len(other_r) > 0 else max_r)
            max_r = max(0.01, max_r)
            if r_i < max_r:
                new_r = min(max_r, r_i + rng.uniform(0.001, 0.01) * T)
                new_state[idx, 2] = new_r
        
        # Clip positions and radii
        new_state[:, 0] = np.clip(new_state[:, 0], -0.05, 1.05)
        new_state[:, 1] = np.clip(new_state[:, 1], -0.05, 1.05)
        new_state[:, 2] = np.maximum(new_state[:, 2], 0.005)
        
        # Calculate new energy
        new_energy = _compute_energy(new_state)
        
        # Metropolis acceptance
        delta_e = new_energy - energies[i]
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            states[i] = new_state
            energies[i] = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_state = new_state.copy()
        
        # Attempt swap with adjacent replica
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

def _optimize_single(xyz0):
    """Optimize from a single initial config with adaptive penalty."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    # Phase 1: Low penalty, focus on layout expansion
    penalty = 5.0
    remaining = _remaining_time()
    if remaining > 10:
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-3),
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective(params, penalty, 1e-3):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 2: Medium penalty
    remaining = _remaining_time()
    if remaining > 8:
        penalty = 50.0
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4),
                          options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective(params, penalty, 1e-4):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 3: High penalty
    remaining = _remaining_time()
    if remaining > 5:
        penalty = 500.0
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-5),
                          options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < _objective(params, penalty, 1e-5):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 4: Delaunay-based optimization with high penalty
    remaining = _remaining_time()
    if remaining > 4:
        penalty = 1000.0
        xyz_current = params.reshape(N, 3)
        contact_edges = _get_contact_edges(xyz_current)
        try:
            res = minimize(_delaunay_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-5, contact_edges),
                          options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < _delaunay_objective(params, penalty, 1e-5, contact_edges):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 5: Very high penalty, tight tolerance
    remaining = _remaining_time()
    if remaining > 2:
        penalty = 5000.0
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-6),
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < _objective(params, penalty, 1e-6):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 6: Ultra-high penalty for final polish
    remaining = _remaining_time()
    if remaining > 1:
        penalty = 50000.0
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-7),
                          options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < _objective(params, penalty, 1e-7):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    return best_params.reshape(N, 3), best_score

def _delaunay_polish(xyz, max_iters=10):
    """
    Iterative Delaunay-based polish: recompute contact edges each iteration
    and optimize with high penalty on the reduced constraint set.
    """
    params = xyz.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    for it in range(max_iters):
        if _remaining_time() < 2.0:
            break
        
        xyz_current = params.reshape(N, 3)
        contact_edges = _get_contact_edges(xyz_current)
        
        # Use high penalty with Delaunay edges
        penalty = 5000.0
        smooth_eps = 1e-6
        
        try:
            res = minimize(_delaunay_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, smooth_eps, contact_edges),
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-14})
            
            new_score = np.sum(res.x[2::3])
            if new_score > best_score and _validate(res.x.reshape(N, 3)):
                params = res.x
                best_params = params.copy()
                best_score = new_score
            elif _validate(res.x.reshape(N, 3)):
                # Even if score didn't improve, use the valid result for next iteration
                params = res.x
        except Exception:
            break
    
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
        # Try multiple perturbation scales
        for scale in [0.005, 0.01, 0.02, 0.03]:
            if remaining < 5.0:
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
    
    # Final validation
    if not _validate(best_xyz):
        best_xyz = _initial_grid()
    
    # Parallel Tempering Monte Carlo refinement
    remaining = _remaining_time()
    if remaining > 25.0:
        rng = np.random.default_rng(99)
        # Use best_xyz and several perturbed versions as replicas
        replicas = [best_xyz]
        for scale in [0.005, 0.01, 0.015, 0.02]:
            perturbed = best_xyz.copy()
            perturbed += rng.normal(0, scale, perturbed.shape)
            perturbed = _fix_boundaries(perturbed)
            replicas.append(perturbed)
        
        # Temperatures: geometric progression from low to high
        temps = [0.5, 1.0, 2.0, 4.0, 8.0]
        
        # Ensure we have matching number of replicas and temps
        n_rep = min(len(replicas), len(temps))
        replicas = replicas[:n_rep]
        temps = temps[:n_rep]
        
        # Run PT-MC with time-based step limit
        max_steps = int(remaining * 800)  # ~800 steps per second
        max_steps = max(10000, min(max_steps, 200000))
        
        pt_result = _pt_mc_search(replicas, temps, max_steps, rng)
        
        if _validate(pt_result):
            pt_score = np.sum(pt_result[:, 2])
            if pt_score > best_score:
                best_xyz = pt_result
                best_score = pt_score
    
    # Delaunay-based iterative polish
    remaining = _remaining_time()
    if remaining > 10.0 and _validate(best_xyz):
        polished, polished_score = _delaunay_polish(best_xyz, max_iters=8)
        if _validate(polished) and polished_score > best_score:
            best_xyz = polished
            best_score = polished_score
    
    # Final high-precision polish pass
    remaining = _remaining_time()
    if remaining > 3.0 and _validate(best_xyz):
        params = best_xyz.flatten().copy()
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
            bounds.append((0, 1))
            bounds.append((0, 0.5))
        
        # Very high penalty, very tight tolerance
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(100000.0, 1e-8),
                          options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 20})
            if res.success and _validate(res.x.reshape(N, 3)):
                polish_score = np.sum(res.x[2::3])
                if polish_score > best_score:
                    best_xyz = res.x.reshape(N, 3)
                    best_score = polish_score
        except Exception:
            pass
    
    return best_xyz

# EVOLVE-BLOCK-END
