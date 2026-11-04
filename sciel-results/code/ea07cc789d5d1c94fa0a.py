import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import itertools

N = 26

def _is_valid(params, tol=1e-9):
    """Check if packing is valid: circles inside [0,1]^2, no overlaps."""
    x = params[:, 0]
    y = params[:, 1]
    r = params[:, 2]
    
    # Boundary check
    if np.any(r < -tol):
        return False
    if np.any(x < r - tol) or np.any(x > 1.0 - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > 1.0 - r + tol):
        return False
    
    # Overlap check
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < r[i] + r[j] - tol:
                return False
    return True

def _score(params):
    """Sum of radii for a valid packing."""
    if not _is_valid(params):
        return -np.inf
    return np.sum(params[:, 2])

def _grow_radii(params, start_time, timeout):
    """
    Given a valid packing, try to grow radii by moving circles to new positions.
    Uses a local search approach.
    """
    best = params.copy()
    best_score = _score(best)
    if np.isinf(best_score):
        return best
    
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < timeout:
        improved = False
        for i in range(N):
            # Try to grow radius of circle i
            r_old = best[i, 2]
            r_new = r_old + 0.001
            if r_new > 0.5:
                continue
            best[i, 2] = r_new
            
            if _is_valid(best):
                improved = True
                best_score = _score(best)
                continue
            
            # If invalid, try to move circle i to make room
            # Find the direction that has the most clearance
            x_i = best[i, 0]
            y_i = best[i, 1]
            
            # Check which neighbors are causing overlap
            min_clearance = np.inf
            best_pos = None
            
            for dx_try in [-0.05, -0.025, 0, 0.025, 0.05]:
                for dy_try in [-0.05, -0.025, 0, 0.025, 0.05]:
                    new_x = x_i + dx_try
                    new_y = y_i + dy_try
                    
                    # Check boundary
                    if new_x < r_new or new_x > 1.0 - r_new:
                        continue
                    if new_y < r_new or new_y > 1.0 - r_new:
                        continue
                    
                    # Check overlaps
                    valid_pos = True
                    for j in range(N):
                        if j == i:
                            continue
                        dx = new_x - best[j, 0]
                        dy = new_y - best[j, 1]
                        d = np.sqrt(dx*dx + dy*dy)
                        if d < r_new + best[j, 2]:
                            valid_pos = False
                            break
                    
                    if valid_pos:
                        # Calculate clearance to nearest constraint
                        clearance = min(
                            new_x - r_new,
                            1.0 - new_x - r_new,
                            new_y - r_new,
                            1.0 - new_y - r_new
                        )
                        for j in range(N):
                            if j == i:
                                continue
                            dx = new_x - best[j, 0]
                            dy = new_y - best[j, 1]
                            d = np.sqrt(dx*dx + dy*dy)
                            clearance = min(clearance, d - (r_new + best[j, 2]))
                        
                        if clearance < min_clearance:
                            min_clearance = clearance
                            best_pos = (new_x, new_y)
            
            if best_pos is not None:
                best[i, 0] = best_pos[0]
                best[i, 1] = best_pos[1]
                if _is_valid(best):
                    improved = True
                    best_score = _score(best)
                else:
                    best[i, 2] = r_old
            
            if not _is_valid(best):
                best[i, 2] = r_old
        
        if not improved:
            # Try random moves
            for i in range(N):
                r_old = best[i, 2]
                # Try random new positions
                for _ in range(10):
                    x_new = np.clip(best[i, 0] + rng.uniform(-0.1, 0.1), r_old + 1e-3, 1.0 - r_old - 1e-3)
                    y_new = np.clip(best[i, 1] + rng.uniform(-0.1, 0.1), r_old + 1e-3, 1.0 - r_old - 1e-3)
                    best[i, 0] = x_new
                    best[i, 1] = y_new
                    if _is_valid(best):
                        improved = True
                        # Try to grow radius now
                        best[i, 2] = r_old + 0.001
                        if _is_valid(best):
                            best_score = _score(best)
                        else:
                            best[i, 2] = r_old
                        break
                    else:
                        best[i, 0] = r_old  # placeholder, will revert
                # Revert if not improved
                if best[i, 2] == r_old and (best[i, 0] != best[i, 0] or best[i, 1] != best[i, 1]):
                    best[i, 0] = x_new if 'x_new' in dir() else best[i, 0]
            
            if not improved:
                break
    
    return best

def _hex_packing_init():
    """Create a hexagonal close-packing initialization for 26 circles."""
    # Hexagonal packing in unit square
    # For n circles, try to fit them in hexagonal pattern
    rows = 5
    # Distribute 26 circles across 5 rows
    # Row sizes: 6, 5, 5, 5, 5 = 26
    row_sizes = [6, 5, 5, 5, 5]
    
    circles = []
    for row_idx, count in enumerate(row_sizes):
        y = (row_idx + 0.5) / rows
        for col_idx in range(count):
            x = (col_idx + 0.5) / count
            r = min(0.5 / rows, 0.5 / count) * 0.8  # Scale down to fit
            circles.append([x, y, r])
    
    return np.array(circles[:N])

def _grid_init():
    """Create a grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side * 0.9
                cells.append([x, y, r])
    return np.array(cells[:N])

def _random_init(rng):
    """Create a random initialization."""
    params = np.zeros((N, 3))
    params[:, 0] = rng.uniform(0.1, 0.9, N)
    params[:, 1] = rng.uniform(0.1, 0.9, N)
    params[:, 2] = rng.uniform(0.02, 0.1, N)
    return params

def _soft_objective(params, N, mu):
    """
    Soft objective with barrier terms for constraints.
    """
    x = params[:, 0]
    y = params[:, 1]
    r = params[:, 2]
    
    obj = -np.sum(r)
    
    # Boundary barriers: r <= min(x, 1-x, y, 1-y)
    # Use log barrier or soft penalty
    margin_x = np.minimum(x, 1 - x) - r
    margin_y = np.minimum(y, 1 - y) - r
    # Penalty if margin < 0
    boundary_viol = np.maximum(0, -margin_x) + np.maximum(0, -margin_y)
    penalty_boundary = mu * np.sum(boundary_viol**2)
    
    # Overlap penalties
    penalty_overlap = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            if d < min_dist:
                penalty_overlap += (min_dist - d)**2
    
    obj += penalty_boundary + mu * penalty_overlap
    
    # Keep radii positive
    penalty_negative_r = mu * np.sum(np.maximum(0, -r)**2)
    obj += penalty_negative_r
    
    return obj

def _tension_network_optimize(params, start_time, timeout, mu_base=1.0):
    """
    Optimize using a tension network approach with Delaunay triangulation.
    """
    best = params.copy()
    best_score = _score(best)
    
    if np.isinf(best_score):
        return best
    
    # Compute Delaunay triangulation
    centers = best[:, :2]
    try:
        tri = Delaunay(centers)
        # Get edges from triangles
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    edges.add((simplex[i], simplex[j]))
        edges = list(edges)
    except Exception:
        # Fallback: complete graph
        edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    def objective_with_springs(p, N, mu, edge_list):
        x = p[:, 0]
        y = p[:, 1]
        r = p[:, 2]
        
        obj = -np.sum(r)
        
        # Boundary penalties
        margin_x = np.minimum(x, 1 - x) - r
        margin_y = np.minimum(y, 1 - y) - r
        boundary_viol = np.maximum(0, -margin_x) + np.maximum(0, -margin_y)
        obj += mu * np.sum(boundary_viol**2)
        
        # Spring forces between connected circles
        spring_obj = 0.0
        for i, j in edge_list:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            rest_length = r[i] + r[j]
            # Penalty for d < rest_length
            if d < rest_length:
                spring_obj += (rest_length - d)**2
        obj += mu * spring_obj
        
        # Positive radii
        obj += mu * np.sum(np.maximum(0, -r)**2)
        
        return obj
    
    # Anneal: decrease mu to make constraints tighter
    mu = mu_base
    anneal_factor = 0.7
    min_mu = 0.01
    
    while time.time() - start_time < timeout:
        if mu < min_mu:
            mu = mu_base * 0.1
            # Recompute Delaunay
            try:
                tri = Delaunay(best[:, :2])
                edges = set()
                for simplex in tri.simplices:
                    for i in range(3):
                        for j in range(i + 1, 3):
                            edges.add((simplex[i], simplex[j]))
                edges = list(edges)
            except Exception:
                edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        
        res = minimize(
            objective_with_springs,
            best,
            args=(N, mu, edges),
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        
        candidate = res.x.reshape(N, 3)
        candidate[:, 2] = np.maximum(candidate[:, 2], 1e-6)
        
        score = _score(candidate)
        if not np.isinf(score) and score > best_score:
            best = candidate
            best_score = score
        
        mu *= anneal_factor
        
        # Check if we're making progress
        if mu < min_mu and best_score <= _score(params) * 1.001:
            break
    
    return best

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    # Multiple initialization strategies
    initializations = []
    
    # 1. Hexagonal packing
    initializations.append(_hex_packing_init())
    
    # 2. Grid
    initializations.append(_grid_init())
    
    # 3. Random initializations
    for _ in range(5):
        initializations.append(_random_init(rng))
    
    best_params = None
    best_score = -np.inf
    
    for init_params in initializations:
        if time.time() - start_time > timeout - 10:
            break
        
        # First, make it valid by shrinking circles
        p = init_params.copy()
        for i in range(N):
            min_dist = min(p[i, 0], 1 - p[i, 0], p[i, 1], 1 - p[i, 1])
            if p[i, 2] > min_dist:
                p[i, 2] = min_dist * 0.99
        p[:, 2] = np.maximum(p[:, 2], 1e-4)
        
        # Fix overlaps by shrinking
        for _ in range(50):
            changed = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = p[i, 0] - p[j, 0]
                    dy = p[i, 1] - p[j, 1]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < p[i, 2] + p[j, 2]:
                        # Shrink the smaller one
                        if p[i, 2] <= p[j, 2]:
                            p[i, 2] = max(0.001, d - p[j, 2])
                        else:
                            p[j, 2] = max(0.001, d - p[i, 2])
                        changed = True
            if not changed:
                break
        
        if not _is_valid(p):
            continue
        
        score = _score(p)
        if score > best_score:
            best_score = score
            best_params = p.copy()
        
        # Optimize with tension network
        if time.time() - start_time < timeout - 20:
            optimized = _tension_network_optimize(p, start_time, timeout - 15)
            opt_score = _score(optimized)
            if not np.isinf(opt_score) and opt_score > best_score:
                best_score = opt_score
                best_params = optimized.copy()
        
        # Try to grow radii
        if time.time() - start_time < timeout - 5 and best_params is not None:
            grown = _grow_radii(best_params, start_time, timeout)
            grown_score = _score(grown)
            if not np.isinf(grown_score) and grown_score > best_score:
                best_score = grown_score
                best_params = grown.copy()
    
    if best_params is None:
        # Fallback: return a valid grid
        best_params = _grid_init()
        # Make it valid
        for i in range(N):
            min_dist = min(best_params[i, 0], 1 - best_params[i, 0], best_params[i, 1], 1 - best_params[i, 1])
            if best_params[i, 2] > min_dist:
                best_params[i, 2] = min_dist * 0.99
        best_params[:, 2] = np.maximum(best_params[:, 2], 1e-4)
    
    return best_params
