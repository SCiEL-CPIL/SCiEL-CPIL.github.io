"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True

def _hex_init():
    """
    Create a hexagonal lattice packing of N circles.
    This is a very dense initial configuration, close to optimal for large N.
    We scale it to fit within [0,1]^2.
    """
    # Generate hex lattice points
    # Distance between centers in hex lattice is 2*r
    # Let's generate a large enough grid and pick the first N
    # Hex lattice vectors:
    # v1 = (1, 0) * s
    # v2 = (0.5, sqrt(3)/2) * s
    # where s is the distance between centers.
    
    # We want to fit N=26 circles.
    # A hexagonal packing of 26 circles fits well in a roughly 5x5 or 6x4 block.
    # Let's generate points in a bounding box and scale.
    
    points = []
    # Generate enough points
    max_coord = 10
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            x = i + 0.5 * j
            y = (np.sqrt(3) / 2) * j
            points.append([x, y])
            
    points = np.array(points)
    
    # Sort by distance from origin to get a compact cluster
    dists = np.linalg.norm(points, axis=1)
    sorted_indices = np.argsort(dists)
    points = points[sorted_indices]
    
    # Take first N points
    points = points[:N]
    
    # Center the cluster
    points -= np.mean(points, axis=0)
    
    # Scale to fit in [0,1]^2 with some margin
    # Find bounding box
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # We want the distance between adjacent centers to be 2*r.
    # In our lattice, distance is 1. So if we scale by S, distance is S.
    # Then r = S/2.
    # The total width of the packing is roughly width * S + 2*r = width * S + S = S * (width + 1).
    # Similarly for height.
    # We need S * (width + 1) <= 1 and S * (height + 1) <= 1.
    # So S <= 1 / (width + 1) and S <= 1 / (height + 1).
    
    S = min(1.0 / (width + 1.0), 1.0 / (height + 1.0))
    
    # Scale points
    points *= S
    
    # Shift to center in [0,1]^2
    points += 0.5
    
    # Calculate radii
    r = S / 2.0
    
    # Create output array
    xy = points
    r_arr = np.full(N, r)
    
    return xy, r_arr

def _objective(v, n, deadline):
    """
    Objective function for optimization.
    Minimizes -sum(r) + penalty for violations.
    """
    if time.time() > deadline:
        return 1e10
        
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]
    
    # Wall violations
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = np.maximum(0.0, r - wall_min)
    
    # Pairwise violations
    total_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            viol = r[i] + r[j] - dist
            if viol > 0:
                total_viol += viol * viol
                
    wall_pen = np.sum(wall_viol * wall_viol)
    
    return -np.sum(r) + 100.0 * (wall_pen + total_viol)

def _gradient_polish(xy, r, deadline):
    """
    High-precision polish using L-BFGS-B.
    """
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])
    
    bounds = []
    for i in range(n):
        bounds.append((0.0001, 0.9999))
    for i in range(n):
        bounds.append((0.0001, 0.9999))
    for i in range(n):
        bounds.append((0.0001, 0.4999))
        
    def obj(v):
        return _objective(v, n, deadline)
        
    try:
        result = minimize(obj, v0, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8,
                                   'maxfun': 50000})
    except Exception:
        return xy, r
        
    v_opt = result.x
    new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
    new_r = v_opt[2*n:3*n]
    
    total, valid = _eval_packing(new_xy, new_r)
    if valid:
        return new_xy, new_r
        
    return xy, r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Phase 1: Hexagonal Initialization
    xy_hex, r_hex = _hex_init()
    total_hex, valid_hex = _eval_packing(xy_hex, r_hex)
    
    best_xy = xy_hex.copy()
    best_r = r_hex.copy()
    best_total = total_hex if valid_hex else 0.0
    
    # Phase 2: Differential Evolution
    # This is a global search method that can escape local minima.
    # We use the hex packing as a starting point or reference.
    # DE requires bounds for all variables.
    
    # Bounds:
    # x, y in [0, 1]
    # r in [0, 0.5]
    
    bounds_de = []
    for i in range(N):
        bounds_de.append((0.0, 1.0)) # x
    for i in range(N):
        bounds_de.append((0.0, 1.0)) # y
    for i in range(N):
        bounds_de.append((0.0, 0.5)) # r
        
    # Initial population: mix of hex packing and random
    # DE generates its own population, but we can guide it.
    # However, standard DE starts random. To leverage hex, we might run DE
    # and then compare with hex. Or we can initialize the population.
    # scipy.optimize.differential_evolution does not accept an initial population directly in older versions,
    # but we can use 'init' parameter in newer versions? No, 'init' is a string or array.
    # Let's just run DE. It's robust.
    
    # To save time, we limit the population size and generations.
    # N=26, so 78 variables.
    
    try:
        # Use a smaller population to fit in time
        # popsize=10 is default, maybe 5-10 is okay.
        # maxiter=100 might be enough for a decent jump.
        
        # We need to be careful with the objective function calls.
        # DE calls the objective many times.
        
        res = differential_evolution(
            lambda v: _objective(v, N, deadline),
            bounds_de,
            seed=42,
            maxiter=200,
            popsize=5,
            tol=1e-4,
            polish=True,  # DE does a final polish using L-BFGS-B if available
            init='sobol' # Use Sobol sequence for better space filling
        )
        
        if res.success or (time.time() < deadline - 10):
            v_opt = res.x
            new_xy = np.column_stack([v_opt[0:N], v_opt[N:2*N]])
            new_r = v_opt[2*N:3*N]
            
            total_de, valid_de = _eval_packing(new_xy, new_r)
            
            if valid_de and total_de > best_total:
                best_total = total_de
                best_xy = new_xy
                best_r = new_r
                
    except Exception:
        pass
        
    # Phase 3: Gradient Polish on the best result
    if time.time() < deadline - 5:
        polished_xy, polished_r = _gradient_polish(best_xy, best_r, deadline)
        total_pol, valid_pol = _eval_packing(polished_xy, polished_r)
        
        if valid_pol and total_pol > best_total:
            best_total = total_pol
            best_xy = polished_xy
            best_r = polished_r
            
    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
        
    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        # Fallback to hex if it was valid
        if valid_hex:
            return np.column_stack([xy_hex, r_hex])
        else:
            # Absolute fallback: grid
            side = int(np.ceil(np.sqrt(N)))
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result
            
    return result
# EVOLVE-BLOCK-END
