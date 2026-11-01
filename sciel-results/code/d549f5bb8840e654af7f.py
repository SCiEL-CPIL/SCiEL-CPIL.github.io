import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi, QhullError
import time

N = 32

def compute_inradii(p):
    """
    Compute inradius of each Voronoi cell for points p in [0,1]^2.
    Returns array of inradii.
    """
    p = p.reshape(N, 2)
    radii = np.zeros(N)
    
    # Add boundary points to handle Voronoi cells touching domain boundary
    # We'll use a different approach: compute distance to nearest other point and walls
    
    for i in range(N):
        px, py = p[i]
        
        # Distance to walls
        wall_dists = np.array([px, 1-px, py, 1-py])
        min_wall = np.min(wall_dists)
        
        # Distance to other points / 2
        if N > 1:
            diff = p - p[i]
            dists = np.sqrt(np.sum(diff**2, axis=1))
            dists[i] = np.inf
            min_neigh = np.min(dists) / 2.0
        else:
            min_neigh = min_wall
        
        radii[i] = min(min_wall, min_neigh)
    
    return radii

def sum_inradii(p):
    """Sum of inradii - the objective to maximize"""
    return np.sum(compute_inradii(p))

def grad_inradii(p):
    """
    Compute gradient of sum of inradii w.r.t. positions.
    Uses subgradient method for the non-smooth min operations.
    """
    p = p.reshape(N, 2)
    grad = np.zeros_like(p)
    
    for i in range(N):
        px, py = p[i]
        
        # Distance to walls
        wall_dists = np.array([px, 1-px, py, 1-py])
        min_wall = np.min(wall_dists)
        wall_grad = np.zeros(2)
        
        # Find which wall(s) are active
        for w, g in zip(wall_dists, [[1,0], [-1,0], [0,1], [0,-1]]):
            if abs(w - min_wall) < 1e-10:
                wall_grad += np.array(g)
        
        # Distance to neighbors
        if N > 1:
            diff = p - p[i]
            dists = np.sqrt(np.sum(diff**2, axis=1))
            dists[i] = np.inf
            min_neigh_val = np.min(dists)
            min_neigh_idx = np.argmin(dists)
            min_neigh = min_neigh_val / 2.0
            
            neigh_grad = np.zeros(2)
            if min_neigh_val > 1e-10:
                vec = p[i] - p[min_neigh_idx]
                neigh_grad = 0.5 * vec / min_neigh_val
        else:
            min_neigh = min_wall
            neigh_grad = np.zeros(2)
        
        # Determine which constraint is active
        if min_wall < min_neigh - 1e-10:
            # Wall-limited
            n_active_walls = sum(1 for w in wall_dists if abs(w - min_wall) < 1e-10)
            if n_active_walls > 0:
                wall_grad = wall_grad / n_active_walls
            grad[i] += wall_grad
        elif min_neigh < min_wall - 1e-10:
            # Neighbor-limited
            grad[i] += neigh_grad
            # Also update the neighbor's gradient
            if N > 1:
                j = min_neigh_idx
                d = dists[j]
                if d > 1e-10:
                    vec = p[j] - p[i]
                    grad[j] += 0.5 * vec / d
        else:
            # Kink point - average of both
            n_active_walls = sum(1 for w in wall_dists if abs(w - min_wall) < 1e-10)
            if n_active_walls > 0:
                wall_grad = wall_grad / n_active_walls
            grad[i] += 0.5 * (wall_grad + neigh_grad)
            if N > 1:
                j = min_neigh_idx
                d = dists[j]
                if d > 1e-10:
                    vec = p[j] - p[i]
                    grad[j] += 0.25 * vec / d
    
    return grad.flatten()

def voronoi_local_search(p, max_iter=200, step_size=0.01):
    """
    Local search using Voronoi inradius gradient.
    """
    p = p.reshape(N, 2).copy()
    
    for it in range(max_iter):
        val = sum_inradii(p)
        grad = grad_inradii(p)
        grad = grad.reshape(N, 2)
        
        # Normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            break
        
        # Take a step in gradient direction
        new_p = p + step_size * grad / gnorm
        new_p = np.clip(new_p, 0.001, 0.999)
        
        new_val = sum_inradii(new_p)
        
        if new_val > val:
            p = new_p
        else:
            # Try smaller step
            step_size *= 0.7
            if step_size < 1e-6:
                break
    
    return p.flatten()

def simulated_annealing(p, max_iter=500, T0=0.05, T_end=0.001):
    """
    Simulated annealing to escape local minima.
    """
    p = p.reshape(N, 2).copy()
    current_val = sum_inradii(p)
    best_p = p.copy()
    best_val = current_val
    
    for it in range(max_iter):
        # Annealing schedule
        T = T0 * (T_end / T0) ** (it / max_iter)
        
        # Propose a move
        move_idx = np.random.randint(N)
        move_size = 0.02 * (1 + 0.5 * np.sin(it / 50))
        delta = np.random.normal(0, move_size, 2)
        
        new_p = p.copy()
        new_p[move_idx] += delta
        new_p[move_idx] = np.clip(new_p[move_idx], 0.001, 0.999)
        
        new_val = sum_inradii(new_p)
        delta_val = new_val - current_val
        
        # Accept or reject
        if delta_val > 0 or np.random.rand() < np.exp(delta_val / T):
            p = new_p
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best_p = p.copy()
    
    return best_p.flatten(), best_val

def create_grid_init(rows, cols, n):
    """Create a grid initialization"""
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) < n:
                points.append([xs[j], ys[i]])
    return np.array(points[:n]).flatten()

def hexagonal_init(n):
    """Create hexagonal packing initialization"""
    points = []
    # Hexagonal packing with 6 rows
    for row in range(6):
        y = (row + 0.5) / 6.0
        if row % 2 == 0:
            for col in range(6):
                x = (col + 0.5) / 6.0
                if len(points) < n:
                    points.append([x, y])
        else:
            for col in range(5):
                x = (col + 1.0) / 6.0
                if len(points) < n:
                    points.append([x, y])
    
    # Add remaining points
    while len(points) < n:
        points.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
    
    return np.array(points[:n]).flatten()

def construct():
    start_time = time.time()
    timeout = 110
    
    best_c = None
    best_val = -np.inf
    
    np.random.seed(42)
    
    # Multiple initializations
    inits = []
    
    # 1. Grid initializations
    inits.append(create_grid_init(6, 6, N))
    inits.append(create_grid_init(5, 7, N))
    inits.append(create_grid_init(7, 5, N))
    inits.append(create_grid_init(4, 8, N))
    inits.append(create_grid_init(8, 4, N))
    
    # 2. Hexagonal
    inits.append(hexagonal_init(N))
    
    # 3. Jittered grids
    base = create_grid_init(6, 6, N).reshape(N, 2)
    for jitter_scale in [0.02, 0.04, 0.06]:
        jitter = np.random.uniform(-jitter_scale, jitter_scale, size=(N, 2))
        p = np.clip(base + jitter, 0.01, 0.99)
        inits.append(p.flatten())
    
    # 4. Random
    for _ in range(3):
        p = np.random.uniform(0.05, 0.95, size=(N, 2))
        inits.append(p.flatten())
    
    bounds = [(0.001, 0.999), (0.001, 0.999)] * N
    
    for idx, c_init in enumerate(inits):
        if time.time() - start_time > timeout - 20:
            break
        
        c = c_init.copy()
        
        # Phase 1: L-BFGS-B optimization
        try:
            res = minimize(lambda c: -sum_inradii(c.reshape(N, 2)), c,
                          jac=lambda c: -grad_inradii(c),
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            c = res.x
        except:
            pass
        
        # Phase 2: Local search with inradius gradient
        if time.time() - start_time < timeout - 15:
            try:
                c = voronoi_local_search(c, max_iter=100, step_size=0.01)
            except:
                pass
        
        # Phase 3: Simulated annealing
        if time.time() - start_time < timeout - 10:
            try:
                c_sa, val_sa = simulated_annealing(c, max_iter=300, T0=0.03, T_end=0.0005)
                if val_sa > sum_inradii(c.reshape(N, 2)):
                    c = c_sa
            except:
                pass
        
        # Phase 4: Final L-BFGS-B polish
        if time.time() - start_time < timeout - 5:
            try:
                res = minimize(lambda c: -sum_inradii(c.reshape(N, 2)), c,
                              jac=lambda c: -grad_inradii(c),
                              method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                c = res.x
            except:
                pass
        
        val = sum_inradii(c.reshape(N, 2))
        if val > best_val:
            best_val = val
            best_c = c.copy()
    
    if best_c is None:
        best_c = create_grid_init(6, 6, N)
        best_val = sum_inradii(best_c.reshape(N, 2))
    
    # Final polish
    if time.time() - start_time < timeout - 2:
        try:
            res = minimize(lambda c: -sum_inradii(c.reshape(N, 2)), best_c,
                          jac=lambda c: -grad_inradii(c),
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12})
            c = res.x
            val = sum_inradii(c.reshape(N, 2))
            if val > best_val:
                best_val = val
                best_c = c.copy()
        except:
            pass
    
    p = best_c.reshape(N, 2)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    
    result = np.column_stack((p, radii))
    return result
