"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26

# EVOLVE-BLOCK-START

def _constraints(pos, radii):
    """
    Return list of constraint functions (inequality: g(x) >= 0).
    """
    cons = []
    
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})  # x_i >= r_i
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N+i]})  # x_i <= 1-r_i
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})  # y_i >= r_i
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N+i]})  # y_i <= 1-r_i
    
    # Non-overlap constraints: dist_ij >= r_i + r_j
    for i in range(N):
        for j in range(i+1, N):
            def overlap_constraint(x, i=i, j=j):
                dx = x[2*i] - x[2*j]
                dy = x[2*i+1] - x[2*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                return dist - x[2*N+i] - x[2*N+j]
            cons.append({'type': 'ineq', 'fun': overlap_constraint})
    
    return cons

def _objective(x):
    """
    Maximize sum of radii => minimize negative sum of radii.
    x = [x_0, y_0, x_1, y_1, ..., x_{N-1}, y_{N-1}, r_0, r_1, ..., r_{N-1}]
    """
    radii = x[2*N:]
    return -np.sum(radii)

def _is_valid(pos, radii, tol=1e-8):
    """
    Strict validity check.
    """
    if np.any(pos[:, 0] < -tol) or np.any(pos[:, 0] > 1.0 + tol):
        return False
    if np.any(pos[:, 1] < -tol) or np.any(pos[:, 1] > 1.0 + tol):
        return False
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
        
    return True

def _slsqp_solve(x0, deadline):
    """
    Solve using SLSQP with explicit constraints.
    """
    if time.time() > deadline:
        return x0
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
    for i in range(N):
        bounds.append((1e-6, 0.5))  # r
    
    cons = _constraints(x0[:2*N].reshape(N, 2), x0[2*N:])
    
    try:
        result = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                         constraints=cons,
                         options={'maxiter': 200, 'ftol': 1e-12})
        if result.success and result.fun < _objective(x0):
            return result.x
    except Exception:
        pass
    
    return x0

def _local_search(x0, deadline, n_iter=50):
    """
    Local search: randomly perturb and try SLSQP.
    """
    best_x = x0.copy()
    best_obj = _objective(x0)
    rng = np.random.RandomState(int(time.time() * 1000) % 2**31)
    
    for _ in range(n_iter):
        if time.time() > deadline:
            break
        
        # Small random perturbation
        x_try = x0.copy()
        x_try += rng.normal(0, 0.02, len(x_try))
        x_try = np.clip(x_try, [b[0] for b in [(0,1),(0,1)]*N + [(1e-6, 0.5)]*N],
                        [b[1] for b in [(0,1),(0,1)]*N + [(1e-6, 0.5)]*N])
        
        x_new = _slsqp_solve(x_try, deadline)
        obj_new = _objective(x_new)
        
        if obj_new < best_obj - 1e-10:
            best_obj = obj_new
            best_x = x_new.copy()
            x0 = x_new.copy()  # Continue from improved point
    
    return best_x

def _de_global(x_bounds, deadline, seed=42):
    """
    Use differential evolution for global search.
    """
    if time.time() > deadline:
        return None
    
    # DE with constraints via penalty (since DE doesn't support constraints directly)
    def _penalty_objective(x):
        pos = x[:2*N].reshape(N, 2)
        radii = np.maximum(x[2*N:], 1e-6)
        
        # Objective: minimize -sum(radii)
        obj = -np.sum(radii)
        
        # Boundary violations
        bv = np.maximum(0, radii - pos[:, 0]) + np.maximum(0, pos[:, 0] + radii - 1) + \
             np.maximum(0, radii - pos[:, 1]) + np.maximum(0, pos[:, 1] + radii - 1)
        boundary_pen = np.sum(bv**2)
        
        # Overlap violations
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        overlap = np.maximum(0, min_dist - dist)
        overlap_pen = np.sum(overlap[mask]**2)
        
        return obj + 1000.0 * (boundary_pen + overlap_pen)
    
    try:
        result = differential_evolution(_penalty_objective, x_bounds, seed=seed,
                                        maxiter=100, popsize=15, tol=1e-8,
                                        mutation=(0.3, 1.2), recombination=0.9,
                                        polish=True, init='sobol',
                                        updating='deferred', seed=seed)
        if result.success:
            return result.x
    except Exception:
        pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Define bounds for DE
    x_bounds = [(0.0, 1.0)] * (2*N) + [(1e-6, 0.5)] * N
    
    best_x = None
    best_obj = np.inf
    
    # Phase 1: Global search with DE (if time permits)
    if deadline - time.time() > 40:
        x_de = _de_global(x_bounds, deadline - 10, seed=42)
        if x_de is not None:
            obj_de = _objective(x_de)
            if obj_de < best_obj:
                best_obj = obj_de
                best_x = x_de.copy()
    
    # Phase 2: Multi-start SLSQP from diverse initializations
    inits = []
    
    # Grid init
    pos_grid = np.zeros((N, 2))
    rad_grid = np.zeros(N)
    spacing = 1.0 / 5.0
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if idx < N:
                pos_grid[idx] = [(i + 0.5) * spacing, (j + 0.5) * spacing]
                rad_grid[idx] = spacing * 0.45
    if N > 25:
        pos_grid[25] = [0.5, 0.5]
        rad_grid[25] = 0.05
    x_grid = np.concatenate([pos_grid.flatten(), rad_grid])
    inits.append(x_grid)
    
    # Random init
    rng = np.random.RandomState(42)
    pos_rand = rng.uniform(0.1, 0.9, (N, 2))
    rad_rand = rng.uniform(0.05, 0.1, N)
    x_rand = np.concatenate([pos_rand.flatten(), rad_rand])
    inits.append(x_rand)
    
    # Corner-focused init
    pos_corner = np.zeros((N, 2))
    rad_corner = np.zeros(N)
    corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92]]
    for i, c in enumerate(corners):
        pos_corner[i] = c
        rad_corner[i] = 0.12
    edges = [
        [0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5],
        [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75],
        [0.5, 0.5]
    ]
    for i in range(4, N):
        if i - 4 < len(edges):
            pos_corner[i] = edges[i - 4]
            rad_corner[i] = 0.06
        else:
            pos_corner[i] = rng.uniform(0.2, 0.8, 2)
            rad_corner[i] = 0.05
    x_corner = np.concatenate([pos_corner.flatten(), rad_corner])
    inits.append(x_corner)
    
    # Use DE result as init if available
    if best_x is not None:
        inits.append(best_x.copy())
    
    for x_init in inits:
        if time.time() > deadline - 20:
            break
        
        # Local search from this init
        x_ls = _local_search(x_init, deadline - 5, n_iter=30)
        obj_ls = _objective(x_ls)
        
        if obj_ls < best_obj - 1e-10:
            best_obj = obj_ls
            best_x = x_ls.copy()
    
    # Phase 3: Final polish with SLSQP from best solution
    if best_x is not None and time.time() < deadline - 2:
        x_final = _slsqp_solve(best_x, deadline)
        obj_final = _objective(x_final)
        if obj_final < best_obj - 1e-12:
            best_obj = obj_final
            best_x = x_final.copy()
    
    # Extract and validate
    if best_x is None:
        # Fallback
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(5):
            for j in range(5):
                if i*5+j < N-1:
                    pos[i*5+j] = [(i+0.5)/5.0, (j+0.5)/5.0]
                    radii[i*5+j] = 0.09
        pos[N-1] = [0.5, 0.02]
        radii[N-1] = 0.01
        best_x = np.concatenate([pos.flatten(), radii])
    
    pos_out = best_x[:2*N].reshape(N, 2)
    radii_out = np.maximum(best_x[2*N:], 1e-6)
    
    # Validate and fix if needed
    if not _is_valid(pos_out, radii_out):
        # Shrink radii until valid
        for _ in range(100):
            valid = True
            for i in range(N):
                x, y = pos_out[i]
                r = radii_out[i]
                max_r = min(x, 1.0 - x, y, 1.0 - y)
                if r > max_r + 1e-8:
                    radii_out[i] = max(0.0, max_r - 1e-8)
                    valid = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos_out[i,0] - pos_out[j,0]
                    dy = pos_out[i,1] - pos_out[j,1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    sum_r = radii_out[i] + radii_out[j]
                    if dist < sum_r - 1e-8:
                        k = max(0.0, dist / (sum_r + 1e-12))
                        radii_out[i] *= k
                        radii_out[j] *= k
                        valid = False
            if valid:
                break
    
    result = np.column_stack([pos_out, radii_out])
    return result

# EVOLVE-BLOCK-END
