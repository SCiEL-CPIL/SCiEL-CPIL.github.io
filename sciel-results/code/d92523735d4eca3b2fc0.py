"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay, cKDTree

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
        # Ensure points are unique enough for Delaunay
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
        # Fallback to cKDTree with a reasonable distance if Delaunay fails
        try:
            tree = cKDTree(centers)
            pairs = tree.query_pairs(0.5, output_type='set')
            return list(pairs)
        except Exception:
            return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _smooth_max(x, beta=100.0):
    """
    Smooth approximation of max(0, x).
    Uses log-sum-exp trick: log(exp(x/beta) + exp(0)) * beta
    """
    # To avoid overflow, we can use a simpler smooth max if beta is large
    # Or just use a polynomial approximation if performance is a concern, but log-sum-exp is robust
    # For small N and moderate iterations, this is fine.
    # Alternative: x/2 + sqrt(x^2 + 1/beta^2)/2  (smooth max)
    # Let's use the quadratic smooth max for speed and stability
    return 0.5 * (x + np.sqrt(x*x + 1.0/(beta*beta)))

def _objective_delaunay(params, edges_cache, beta=50.0):
    """
    Objective using Delaunay-based constraints with smooth overlap penalty.
    params: [x_0, y_0, r_0, ..., x_25, y_25, r_25]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary penalties (smoothed)
    # We want to penalize going outside [0,1]
    # dist_to_left = x - r, dist_to_right = 1 - (x + r)
    # We penalize if these are negative.
    # Smooth penalty: smooth_max(-dist, beta)
    
    # Left boundary: x - r < 0
    # Right boundary: x + r > 1 => 1 - x - r < 0
    # Bottom: y - r < 0
    # Top: y + r > 1 => 1 - y - r < 0
    # Radius: r < 0
    
    # Using smooth max for boundaries
    b_left = - (x - r)
    b_right = - (1 - x - r)
    b_bottom = - (y - r)
    b_top = - (1 - y - r)
    b_rad = -r
    
    # Sum of smooth max
    penalty += np.sum(_smooth_max(b_left, beta))
    penalty += np.sum(_smooth_max(b_right, beta))
    penalty += np.sum(_smooth_max(b_bottom, beta))
    penalty += np.sum(_smooth_max(b_top, beta))
    penalty += np.sum(_smooth_max(b_rad, beta))
    
    # Get Delaunay edges
    centers = np.column_stack([x, y])
    edges = _get_delaunay_edges(centers)
    edges_cache['edges'] = edges
    
    # Overlap penalties for Delaunay neighbors
    # overlap = r_i + r_j - dist
    # We penalize if overlap > 0
    # smooth_max(overlap, beta)
    
    for i, j in edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = r[i] + r[j] - dist
        penalty += _smooth_max(overlap, beta)
        
    # Also check a few nearest neighbors not in Delaunay to be safe
    # This handles cases where Delaunay might miss a close pair due to numerical issues or topology changes
    # We use a cKDTree query for a small radius to catch any missed close pairs
    try:
        tree = cKDTree(centers)
        # Query pairs within a distance that could cause overlap
        # Max possible overlap distance is 2*max_r
        max_r = np.max(r)
        query_dist = 2.0 * max_r + 0.01
        close_pairs = tree.query_pairs(query_dist, output_type='set')
        
        for i, j in close_pairs:
            # Skip if already in Delaunay edges to avoid double counting (or just add, penalty is convex)
            # To be safe and smooth, we can just add all close pairs. 
            # However, adding too many pairs makes it dense. 
            # Let's only add if not in Delaunay edges.
            if (min(i,j), max(i,j)) not in edges_cache['edges_set']:
                 dx = x[i] - x[j]
                 dy = y[i] - y[j]
                 dist = np.sqrt(dx * dx + dy * dy)
                 overlap = r[i] + r[j] - dist
                 penalty += _smooth_max(overlap, beta)
    except Exception:
        pass
    
    return -np.sum(r) + 50.0 * penalty

def _objective_full(params, beta=100.0):
    """Full pairwise objective for final polish with smooth penalty."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary penalties
    b_left = - (x - r)
    b_right = - (1 - x - r)
    b_bottom = - (y - r)
    b_top = - (1 - y - r)
    b_rad = -r
    
    penalty += np.sum(_smooth_max(b_left, beta))
    penalty += np.sum(_smooth_max(b_right, beta))
    penalty += np.sum(_smooth_max(b_bottom, beta))
    penalty += np.sum(_smooth_max(b_top, beta))
    penalty += np.sum(_smooth_max(b_rad, beta))
    
    # All pairs
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            penalty += _smooth_max(overlap, beta)
    
    return -np.sum(r) + 100.0 * penalty

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

def _optimize_delaunay(xyz, time_budget=30.0):
    """
    Optimize using Delaunay-based constraints, then polish with full pairwise.
    """
    start_time = time.time()
    params = xyz.flatten()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    edges_cache = {'edges': [], 'edges_set': set()}
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
        
    # Pre-compute edges set for the objective to avoid re-computation overhead in the set check
    # The objective function will update edges_cache['edges'] and we should update the set there too.
    # Let's modify the objective to update the set as well.
    # Wait, the objective is a function, it can't easily modify the set in the cache dict without passing it.
    # Let's just pass the cache dict and let the objective update it.
    
    # Phase 1: Delaunay-based optimization
    try:
        result = minimize(_objective_delaunay, best_params, method='L-BFGS-B',
                          bounds=bounds, args=(edges_cache, 50.0),
                          options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.fun < _objective_delaunay(best_params, edges_cache, 50.0):
            best_params = result.x
            best_score = np.sum(best_params[2::3])
    except Exception:
        pass
    
    if time.time() - start_time > time_budget:
        return best_params.reshape(N, 3)
    
    # Phase 2: Polish with full pairwise constraints
    try:
        result = minimize(_objective_full, best_params, method='L-BFGS-B',
                          bounds=bounds, args=(100.0,),
                          options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.fun < _objective_full(best_params, 100.0):
            best_params = result.x
            best_score = np.sum(best_params[2::3])
    except Exception:
        pass
    
    # Phase 3: High precision polish
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    if remaining > 1.0:
        try:
            result = minimize(_objective_full, best_params, method='L-BFGS-B',
                              bounds=bounds, args=(200.0,),
                              options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            if result.fun < _objective_full(best_params, 200.0):
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
    
    # 1. Simple grid
    initial_configs.append(_initial_config_grid())
    
    # 2. Hexagonal-like
    initial_configs.append(_initial_config_hex())
    
    # 3. Random configurations
    np.random.seed(42)
    for trial in range(4):
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.06, 0.14, N)
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        initial_configs.append(np.column_stack([x, y, r]))
    
    # 4. Concentric rings pattern
    positions = []
    positions.append([0.5, 0.5, 0.10])
    for k in range(8):
        angle = 2 * np.pi * k / 8
        x = 0.5 + 0.12 * 1.8 * np.cos(angle)
        y = 0.5 + 0.12 * 1.8 * np.sin(angle)
        positions.append([x, y, 0.12 * 0.9])
    for k in range(10):
        angle = 2 * np.pi * k / 10 + np.pi / 10
        x = 0.5 + 0.10 * 3.5 * np.cos(angle)
        y = 0.5 + 0.10 * 3.5 * np.sin(angle)
        positions.append([x, y, 0.10 * 0.9])
    for k in range(7):
        angle = 2 * np.pi * k / 7
        x = 0.5 + 0.08 * 4.8 * np.cos(angle)
        y = 0.5 + 0.08 * 4.8 * np.sin(angle)
        positions.append([x, y, 0.08 * 0.9])
    
    initial_configs.append(np.array(positions[:N]))
    
    # Optimize from each initial configuration
    for idx, xyz0 in enumerate(initial_configs):
        elapsed = time.time() - global_start
        remaining = deadline - elapsed
        if remaining < 5.0:
            break
        
        # Allocate more time to earlier configs
        time_budget = min(20.0, remaining * 0.8 - 1.0)
        if time_budget < 2.0:
            break
        
        xyz_opt = _optimize_delaunay(xyz0, time_budget=time_budget)
        
        # Fix if needed
        if not _validate(xyz_opt):
            xyz_opt = _fix_config(xyz_opt)
        
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
    
    # Final validation and fix
    if not _validate(best_xyz):
        best_xyz = _fix_config(best_xyz)
    
    if not _validate(best_xyz):
        best_xyz = _initial_config_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
