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

def _get_delaunay_edges(pts):
    """Get Delaunay triangulation edges for proximity-based constraint reduction."""
    try:
        delaunay = Delaunay(pts)
        edges = set()
        for simplex in delaunay.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
        return list(edges)
    except Exception:
        # Fallback to all pairs
        n = len(pts)
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _objective(params):
    """
    params: array of length 3*n: [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    Objective: minimize -sum(r_i) + penalty for violations
    Uses Delaunay edges for constraint reduction and smooth log-sum-exp penalty.
    """
    n = N
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Penalty for going out of bounds (smooth)
    penalty = 0.0
    for i in range(n):
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
    
    # Get Delaunay edges for proximity constraints
    pts = np.column_stack([x, y])
    edges = _get_delaunay_edges(pts)
    
    # Smooth penalty for overlap using log-sum-exp
    if len(edges) > 0:
        overlaps = []
        for i, j in edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                overlaps.append(overlap)
        
        if overlaps:
            overlaps_arr = np.array(overlaps)
            # Smooth log-sum-exp: beta * log(sum(exp(overlap/beta)))
            beta = 0.01
            max_overlap = np.max(overlaps_arr)
            if max_overlap > 1e-10:
                lse = beta * (np.log(np.sum(np.exp((overlaps_arr - max_overlap) / beta))) + max_overlap / beta)
                penalty += lse ** 2
    
    return -np.sum(r) + 100.0 * penalty

def _initial_config_hex():
    """
    Generate an initial configuration using a hexagonal-like arrangement.
    We'll use a grid with slight perturbations.
    """
    # Use a 5x5 grid (25 circles) plus one more
    # Actually, let's use a smarter initial layout
    # Start with a 5x5 grid
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            # Offset odd rows for hexagonal packing
            if j % 2 == 1:
                x = (i + 0.5) / side + 0.1 / side
            else:
                x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95
            positions.append([x, y, r])
    
    # Add the 26th circle - try to place it in a gap
    # For now, just add another at a random-ish position
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

def _optimize_from(xyz, time_budget=30.0):
    """
    Optimize from a given initial configuration.
    Returns the best valid configuration found.
    Uses Delaunay-based constraint reduction and smooth penalties.
    """
    start_time = time.time()
    params = xyz.flatten()
    
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    # Use L-BFGS-B with smooth log-sum-exp penalty
    def obj(p):
        n = N
        x = p[0::3]
        y = p[1::3]
        r = p[2::3]
        
        penalty = 0.0
        for i in range(n):
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
        
        pts = np.column_stack([x, y])
        edges = _get_delaunay_edges(pts)
        
        # Smooth penalty using log-sum-exp
        if len(edges) > 0:
            overlaps = []
            for i, j in edges:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    overlaps.append(overlap)
            
            if overlaps:
                overlaps_arr = np.array(overlaps)
                beta = 0.01
                max_overlap = np.max(overlaps_arr)
                if max_overlap > 1e-10:
                    lse = beta * (np.log(np.sum(np.exp((overlaps_arr - max_overlap) / beta))) + max_overlap / beta)
                    penalty += lse ** 2
        
        return -np.sum(r) + 100.0 * penalty
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((0, 0.5))  # r
    
    # First pass: moderate penalty weight
    try:
        result = minimize(obj, best_params, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
        if result.fun < obj(best_params):
            best_params = result.x
            best_score = np.sum(best_params[2::3])
    except Exception:
        pass
    
    if time.time() - start_time > time_budget * 0.6:
        pass
    else:
        # Second pass: higher precision polish
        def obj_polish(p):
            n = N
            x = p[0::3]
            y = p[1::3]
            r = p[2::3]
            
            penalty = 0.0
            for i in range(n):
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
            
            pts = np.column_stack([x, y])
            edges = _get_delaunay_edges(pts)
            
            # Smaller beta for tighter penalty
            if len(edges) > 0:
                overlaps = []
                for i, j in edges:
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    overlap = r[i] + r[j] - dist
                    if overlap > 0:
                        overlaps.append(overlap)
                
                if overlaps:
                    overlaps_arr = np.array(overlaps)
                    beta = 0.005
                    max_overlap = np.max(overlaps_arr)
                    if max_overlap > 1e-10:
                        lse = beta * (np.log(np.sum(np.exp((overlaps_arr - max_overlap) / beta))) + max_overlap / beta)
                        penalty += lse ** 2
            
            return -np.sum(r) + 500.0 * penalty
        
        try:
            result2 = minimize(obj_polish, best_params, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if result2.fun < obj_polish(best_params):
                best_params = result2.x
                best_score = np.sum(best_params[2::3])
        except Exception:
            pass
    
    # Final validation
    xyz_best = best_params.reshape(N, 3)
    if _validate(xyz_best):
        return xyz_best
    
    # If invalid, try to fix by reducing radii
    xyz_fixed = xyz_best.copy()
    for _ in range(100):
        if _validate(xyz_fixed):
            break
        # Reduce all radii slightly
        xyz_fixed[:, 2] *= 0.95
        # Also adjust positions to be inside bounds
        for i in range(N):
            x, y, r = xyz_fixed[i]
            xyz_fixed[i, 0] = np.clip(x, r, 1 - r)
            xyz_fixed[i, 1] = np.clip(y, r, 1 - r)
    
    if _validate(xyz_fixed):
        return xyz_fixed
    
    # Fallback: return a simple valid grid
    return _initial_config_grid()

def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of radii.
    """
    global_start = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    best_xyz = _initial_config_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Try multiple initial configurations
    initial_configs = []
    
    # 1. Simple grid
    initial_configs.append(_initial_config_grid())
    
    # 2. Hexagonal-like
    initial_configs.append(_initial_config_hex())
    
    # 3. Random configurations
    np.random.seed(42)
    for trial in range(5):
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.05, 0.15, N)
        # Ensure inside bounds
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        initial_configs.append(np.column_stack([x, y, r]))
    
    # 4. Concentric rings pattern
    center = np.array([0.5, 0.5])
    ring1_r = 0.12
    ring1_n = 8
    ring2_r = 0.10
    ring2_n = 10
    ring3_r = 0.08
    ring3_n = 8
    
    positions = []
    # Center circle
    positions.append([0.5, 0.5, 0.10])
    # Ring 1
    for k in range(ring1_n):
        angle = 2 * np.pi * k / ring1_n
        x = 0.5 + ring1_r * 1.8 * np.cos(angle)
        y = 0.5 + ring1_r * 1.8 * np.sin(angle)
        positions.append([x, y, ring1_r * 0.9])
    # Ring 2
    for k in range(ring2_n):
        angle = 2 * np.pi * k / ring2_n + np.pi / ring2_n
        x = 0.5 + ring2_r * 3.5 * np.cos(angle)
        y = 0.5 + ring2_r * 3.5 * np.sin(angle)
        positions.append([x, y, ring2_r * 0.9])
    # Ring 3
    for k in range(ring3_n):
        angle = 2 * np.pi * k / ring3_n
        x = 0.5 + ring3_r * 4.8 * np.cos(angle)
        y = 0.5 + ring3_r * 4.8 * np.sin(angle)
        positions.append([x, y, ring3_r * 0.9])
    
    initial_configs.append(np.array(positions[:N]))
    
    # Optimize from each initial configuration
    for idx, xyz0 in enumerate(initial_configs):
        elapsed = time.time() - global_start
        remaining = deadline - elapsed
        if remaining < 5.0:
            break
        
        # Give more time to earlier configs, less to later ones
        if idx < 3:
            time_budget = min(30.0, remaining - 2.0)
        else:
            time_budget = min(20.0, remaining - 2.0)
        if time_budget < 2.0:
            break
        
        xyz_opt = _optimize_from(xyz0, time_budget=time_budget)
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
    
    # Final check
    if not _validate(best_xyz):
        # Try to fix
        best_xyz = best_xyz.copy()
        for _ in range(200):
            if _validate(best_xyz):
                break
            best_xyz[:, 2] *= 0.95
            for i in range(N):
                x, y, r = best_xyz[i]
                best_xyz[i, 0] = np.clip(x, r, 1 - r)
                best_xyz[i, 1] = np.clip(y, r, 1 - r)
    
    # If still invalid, use the simple grid
    if not _validate(best_xyz):
        best_xyz = _initial_config_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
