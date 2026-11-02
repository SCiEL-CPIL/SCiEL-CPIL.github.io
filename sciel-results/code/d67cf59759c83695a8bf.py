"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

def _validate(xyz):
    """Check if configuration is valid: circles inside unit square and non-overlapping."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -1e-7) or np.any(x + r > 1 + 1e-7):
        return False
    if np.any(y - r < -1e-7) or np.any(y + r > 1 + 1e-7):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-7:
                return False
    return True

def _get_delaunay_edges(centers):
    """Get Delaunay triangulation edges for the given centers."""
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                if a > b:
                    a, b = b, a
                edges.add((a, b))
        return list(edges)
    except Exception:
        # Fallback to all pairs if Delaunay fails
        return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _objective_delaunay(params, edges_cache):
    """
    Objective using Delaunay-based constraints.
    params: [x_0, y_0, r_0, ..., x_25, y_25, r_25]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary penalties
    for i in range(N):
        if x[i] - r[i] < 0:
            penalty += (x[i] - r[i]) ** 2
        if x[i] + r[i] > 1:
            penalty += (x[i] + r[i] - 1) ** 2
        if y[i] - r[i] < 0:
            penalty += (y[i] - r[i]) ** 2
        if y[i] + r[i] > 1:
            penalty += (y[i] + r[i] - 1) ** 2
        if r[i] < 0:
            penalty += r[i] ** 2
    
    # Get Delaunay edges
    centers = np.column_stack([x, y])
    edges = _get_delaunay_edges(centers)
    edges_cache['edges'] = edges
    edge_set = set(edges)
    
    # Overlap penalties only for Delaunay neighbors (and a few extra close pairs)
    for i, j in edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = r[i] + r[j] - dist
        if overlap > 0:
            penalty += overlap ** 2
    
    # Also check a few nearest neighbors not in Delaunay to be safe
    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) in edge_set or (j, i) in edge_set:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            max_r = max(r[i], r[j])
            if dist_sq < (4 * max_r * max_r):
                dist = np.sqrt(dist_sq)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    penalty += overlap ** 2
    
    return -np.sum(r) + 50.0 * penalty

def _objective_full(params):
    """Full pairwise objective for final polish."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    for i in range(N):
        if x[i] - r[i] < 0:
            penalty += (x[i] - r[i]) ** 2
        if x[i] + r[i] > 1:
            penalty += (x[i] + r[i] - 1) ** 2
        if y[i] - r[i] < 0:
            penalty += (y[i] - r[i]) ** 2
        if y[i] + r[i] > 1:
            penalty += (y[i] + r[i] - 1) ** 2
        if r[i] < 0:
            penalty += r[i] ** 2
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                penalty += overlap ** 2
    
    return -np.sum(r) + 100.0 * penalty

def _initial_config_grid():
    """Simple grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.9
            cells.append([x, y, r])
    return np.array(cells[:N])

def _initial_config_hex():
    """Hexagonal-like initial configuration."""
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            if j % 2 == 1:
                x = (i + 0.5) / side + 0.1 / side
            else:
                x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95
            positions.append([x, y, r])
    
    positions.append([0.5, 0.5, 0.5 / side * 0.5])
    return np.array(positions[:N])

def _initial_config_force_directed(seed):
    """Generate initial config using a simple force-directed relaxation."""
    np.random.seed(seed)
    x = np.random.uniform(0.1, 0.9, N)
    y = np.random.uniform(0.1, 0.9, N)
    r = np.full(N, 0.1)
    
    # Simple relaxation: push apart overlaps
    for _ in range(100):
        moved = False
        for i in range(N):
            fx, fy = 0, 0
            for j in range(N):
                if i == j: continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] and dist > 1e-6:
                    # Repulsive force
                    f = (r[i] + r[j] - dist) / dist
                    fx += f * dx
                    fy += f * dy
            # Boundary forces
            if x[i] < r[i]: fx += (r[i] - x[i]) * 10
            if x[i] > 1 - r[i]: fx -= (x[i] - (1 - r[i])) * 10
            if y[i] < r[i]: fy += (r[i] - y[i]) * 10
            if y[i] > 1 - r[i]: fy -= (y[i] - (1 - r[i])) * 10
            
            x[i] += 0.01 * fx
            y[i] += 0.01 * fy
            x[i] = np.clip(x[i], 0, 1)
            y[i] = np.clip(y[i], 0, 1)
            if np.abs(fx) > 1e-4 or np.abs(fy) > 1e-4:
                moved = True
        if not moved:
            break
            
    # Clip to valid bounds
    for i in range(N):
        x[i] = np.clip(x[i], r[i], 1 - r[i])
        y[i] = np.clip(y[i], r[i], 1 - r[i])
        
    return np.column_stack([x, y, r])

def _optimize_delaunay(xyz, time_budget=30.0):
    """
    Optimize using Delaunay-based constraints, then polish with full pairwise.
    """
    start_time = time.time()
    params = xyz.flatten()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    edges_cache = {'edges': None}
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    # Phase 1: Delaunay-based optimization
    try:
        result = minimize(_objective_delaunay, best_params, method='L-BFGS-B',
                          bounds=bounds, args=(edges_cache,),
                          options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.fun < _objective_delaunay(best_params, edges_cache):
            best_params = result.x
            best_score = np.sum(best_params[2::3])
    except Exception:
        pass
    
    if time.time() - start_time > time_budget:
        return best_params.reshape(N, 3)
    
    # Phase 2: Polish with full pairwise constraints
    try:
        result = minimize(_objective_full, best_params, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.fun < _objective_full(best_params):
            best_params = result.x
            best_score = np.sum(best_params[2::3])
    except Exception:
        pass
    
    return best_params.reshape(N, 3)

def _fix_config(xyz):
    """Try to fix an invalid configuration by reducing radii and adjusting positions."""
    xyz_fixed = xyz.copy()
    for _ in range(100):
        if _validate(xyz_fixed):
            break
        xyz_fixed[:, 2] *= 0.95
        for i in range(N):
            x, y, r = xyz_fixed[i]
            xyz_fixed[i, 0] = np.clip(x, r, 1 - r)
            xyz_fixed[i, 1] = np.clip(y, r, 1 - r)
    return xyz_fixed

def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of their radii.
    """
    global_start = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    best_xyz = _initial_config_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Generate initial configurations
    initial_configs = []
    initial_configs.append(_initial_config_grid())
    initial_configs.append(_initial_config_hex())
    
    # Add several force-directed initializations with different seeds
    for seed in [1, 2, 3, 4, 5]:
        initial_configs.append(_initial_config_force_directed(seed))
    
    # Add a few random ones
    np.random.seed(42)
    for trial in range(3):
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.05, 0.15, N)
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        initial_configs.append(np.column_stack([x, y, r]))
        
    # Optimize from each initial configuration
    for idx, xyz0 in enumerate(initial_configs):
        elapsed = time.time() - global_start
        remaining = deadline - elapsed
        if remaining < 5.0:
            break
        
        # Allocate time budget
        time_budget = min(20.0, remaining - 2.0)
        if time_budget < 2.0:
            break
            
        xyz_opt = _optimize_delaunay(xyz0, time_budget=time_budget)
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
            
    # Final polish on the best configuration
    elapsed = time.time() - global_start
    if elapsed < deadline - 3.0:
        remaining = deadline - elapsed - 1.0
        if remaining > 1.0:
            # Jitter and re-optimize to escape local minima
            np.random.seed(123)
            jittered = best_xyz + np.random.normal(0, 0.01, best_xyz.shape)
            # Ensure jittered is roughly valid
            for i in range(N):
                jittered[i, 0] = np.clip(jittered[i, 0], 0, 1)
                jittered[i, 1] = np.clip(jittered[i, 1], 0, 1)
                jittered[i, 2] = np.clip(jittered[i, 2], 0.01, 0.5)
                
            xyz_polished = _optimize_delaunay(jittered, time_budget=remaining)
            score = np.sum(xyz_polished[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_polished
                
    # Final check
    if not _validate(best_xyz):
        best_xyz = _fix_config(best_xyz)
    
    if not _validate(best_xyz):
        best_xyz = _initial_config_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
