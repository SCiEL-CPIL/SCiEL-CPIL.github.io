import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi
import time

N = 32

def sum_radii(c):
    p = c.reshape(N, 2)
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    return np.sum(radii)

def compute_radii_and_active(c):
    """Compute radii and which constraints are active for each circle."""
    p = c.reshape(N, 2)
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    min_neighbor_idx = np.argmin(dists, axis=1)
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    
    # Determine active constraints
    active_wall = np.zeros(N, dtype=bool)
    active_neighbor = np.zeros(N, dtype=bool)
    for i in range(N):
        if abs(min_wall_dist[i] - radii[i]) < 1e-9:
            active_wall[i] = True
        if abs(min_neighbor_dist[i] - radii[i]) < 1e-9:
            active_neighbor[i] = True
    return radii, active_wall, active_neighbor, min_neighbor_idx

def grad_sum_radii(c):
    """
    Compute gradient of sum of radii using Voronoi cell analysis.
    For each circle, the radius is limited by either wall distance or neighbor distance.
    We compute the gradient contribution from active constraints.
    """
    p = c.reshape(N, 2)
    radii, active_wall, active_neighbor, min_neighbor_idx = compute_radii_and_active(c)
    grad = np.zeros_like(p)
    
    for i in range(N):
        if radii[i] < 1e-12:
            continue
        
        # Gradient from wall constraint
        if active_wall[i]:
            px, py = p[i]
            walls = [px, 1-px, py, 1-py]
            wall_grads = [np.array([1,0]), np.array([-1,0]), np.array([0,1]), np.array([0,-1])]
            min_val = min(walls)
            active_grads = [g for w, g in zip(walls, wall_grads) if abs(w - min_val) < 1e-8]
            if active_grads:
                wall_grad = np.mean(active_grads, axis=0)
            else:
                wall_grad = np.zeros(2)
            
            if active_neighbor[i]:
                # Both constraints active: average the gradients
                j = min_neighbor_idx[i]
                d = np.linalg.norm(p[i] - p[j])
                if d > 1e-10:
                    vec = (p[i] - p[j]) / d
                    neigh_grad_i = 0.5 * vec
                    neigh_grad_j = -0.5 * vec
                    grad[i] += 0.5 * (wall_grad + neigh_grad_i)
                    grad[j] += 0.5 * neigh_grad_j
                else:
                    grad[i] += 0.5 * wall_grad
            else:
                grad[i] += wall_grad
        
        elif active_neighbor[i]:
            j = min_neighbor_idx[i]
            d = np.linalg.norm(p[i] - p[j])
            if d > 1e-10:
                vec = (p[i] - p[j]) / d
                grad[i] += 0.5 * vec
                grad[j] += -0.5 * vec
    
    return grad.flatten()

def phyllotaxis_points(n, radius=0.5, center=(0.5, 0.5)):
    """Generate points using phyllotaxis arrangement (sunflower pattern)"""
    points = []
    for i in range(n):
        r = radius * np.sqrt(i / n)
        theta = i * 137.508 * np.pi / 180.0 # Golden angle in radians
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        # Clip to unit square
        x = np.clip(x, 0.01, 0.99)
        y = np.clip(y, 0.01, 0.99)
        points.append([x, y])
    return np.array(points)

def voronoi_refine(c, max_iter=50, tol=1e-10, time_limit=10.0):
    """
    Refine packing using Voronoi-based local search.
    For each point, compute its Voronoi cell and move it to increase its inradius.
    """
    p = c.reshape(N, 2).copy()
    start_time = time.time()
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        old_val = sum_radii(p.flatten())
        
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            
            p_i = p[i].copy()
            
            # Objective: maximize radius of point i
            def neg_r_i(pos):
                pos = np.array(pos)
                dw = np.min(np.minimum(pos, 1-pos))
                diffs = p - pos
                diffs[i] = np.inf
                d_n = np.min(np.sqrt(np.sum(diffs**2, axis=1))) / 2.0
                return -min(dw, d_n)
            
            old_r_i = -neg_r_i(p_i)
            
            # Use Nelder-Mead for local optimization
            try:
                res = minimize(neg_r_i, p_i, method='Nelder-Mead',
                               options={'maxiter': 30, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                new_r_i = -res.fun
                if new_r_i > old_r_i + 1e-12:
                    p[i] = res.x
                    improved = True
            except:
                pass
        
        new_val = sum_radii(p.flatten())
        if not improved or abs(new_val - old_val) < tol:
            break
    
    return p.flatten()

def multi_start_optimize(c_init, bounds, time_budget=10.0):
    """
    Run multiple optimization passes with different methods and perturbations.
    """
    best_c = c_init.copy()
    best_val = sum_radii(c_init)
    start_time = time.time()
    
    # Pass 1: L-BFGS-B
    if time.time() - start_time < time_budget * 0.4:
        try:
            res = minimize(lambda c: -sum_radii(c), best_c, jac=lambda c: -grad_sum_radii(c),
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
        except:
            pass
    
    # Pass 2: Voronoi refinement
    if time.time() - start_time < time_budget * 0.7:
        try:
            refined = voronoi_refine(best_c, max_iter=20, time_limit=time_budget * 0.2)
            val = sum_radii(refined)
            if val > best_val:
                best_val = val
                best_c = refined.copy()
        except:
            pass
    
    # Pass 3: Another L-BFGS-B polish
    if time.time() - start_time < time_budget * 0.9:
        try:
            res = minimize(lambda c: -sum_radii(c), best_c, jac=lambda c: -grad_sum_radii(c),
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-13})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
        except:
            pass
    
    # Pass 4: Final Voronoi polish
    if time.time() - start_time < time_budget:
        try:
            refined = voronoi_refine(best_c, max_iter=10, time_limit=time_budget * 0.05)
            val = sum_radii(refined)
            if val > best_val:
                best_val = val
                best_c = refined.copy()
        except:
            pass
    
    return best_c, best_val

def coordinate_descent_refine(c, max_sweeps=10, time_limit=10.0):
    """
    Refine using coordinate descent: optimize each point's position 
    to maximize its individual radius, holding others fixed.
    """
    p = c.reshape(N, 2).copy()
    start_time = time.time()
    
    for sweep in range(max_sweeps):
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(N):
            p_i = p[i].copy()
            
            def neg_r_i(pos):
                pos = np.array(pos)
                dw = np.min(np.minimum(pos, 1-pos))
                diffs = p - pos
                diffs[i] = np.inf
                d_n = np.min(np.sqrt(np.sum(diffs**2, axis=1))) / 2.0
                return -min(dw, d_n)
            
            old_r_i = -neg_r_i(p_i)
            try:
                res = minimize(neg_r_i, p_i, method='Nelder-Mead',
                               options={'maxiter': 30, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                new_r_i = -res.fun
                if new_r_i > old_r_i + 1e-12:
                    p[i] = res.x
                    improved = True
            except:
                pass
            
        if not improved:
            break
            
    return p.flatten()

def create_grid_init(rows, cols, n):
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) < n:
                points.append([xs[j], ys[i]])
    return np.array(points[:n]).flatten()

def halton_sequence(n, dims=2):
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    points = []
    for i in range(1, n + 1):
        points.append([halton(i, 2), halton(i, 3)])
    return np.array(points)

def construct():
    start_time = time.time()
    timeout = 115
    
    best_c = None
    best_val = -np.inf
    
    inits = []
    
    # 1. Grids
    for r, c in [(6,6), (5,7), (7,5), (4,8), (8,4), (6,5), (5,6)]:
        inits.append(create_grid_init(r, c, N))
        
    # 2. Halton
    inits.append(halton_sequence(N).flatten())
    
    # 3. Phyllotaxis
    inits.append(phyllotaxis_points(N).flatten())
    
    # 4. Jittered Grids
    np.random.seed(42)
    base = create_grid_init(6, 6, N).reshape(N, 2)
    for jitter_scale in [0.02, 0.04, 0.06]:
        for _ in range(3):
            jitter = np.random.uniform(-jitter_scale, jitter_scale, size=(N, 2))
            p = np.clip(base + jitter, 0.01, 0.99)
            inits.append(p.flatten())
            
    # 5. Random
    for _ in range(5):
        p = np.random.uniform(0.05, 0.95, size=(N, 2))
        inits.append(p.flatten())
        
    # 6. Concentric ring patterns
    for n_rings in [2, 3, 4]:
        points = []
        # Center
        points.append([0.5, 0.5])
        for ring in range(1, n_rings + 1):
            n_pts = int(2 * np.pi * ring / (0.3 * ring))  # Approximate number of points per ring
            n_pts = max(4, min(n_pts, 32 - len(points)))
            r = 0.45 / n_rings * ring
            for k in range(n_pts):
                if len(points) >= N:
                    break
                theta = 2 * np.pi * k / n_pts
                x = 0.5 + r * np.cos(theta)
                y = 0.5 + r * np.sin(theta)
                x = np.clip(x, 0.01, 0.99)
                y = np.clip(y, 0.01, 0.99)
                points.append([x, y])
        if len(points) == N:
            inits.append(np.array(points).flatten())
    
    bounds = [(0.001, 0.999), (0.001, 0.999)] * N
    
    # Determine how much time to allocate per initialization
    n_inits = len(inits)
    time_per_init = (timeout - 15) / n_inits
    
    for idx, c_init in enumerate(inits):
        if time.time() - start_time > timeout - 15:
            break
        
        remaining_time = timeout - 15 - (time.time() - start_time)
        alloc_time = min(time_per_init, remaining_time)
        
        if alloc_time < 1.0:
            break
        
        # Multi-start optimization for this initialization
        try:
            c_opt, val = multi_start_optimize(c_init, bounds, time_budget=alloc_time)
            if val > best_val:
                best_val = val
                best_c = c_opt.copy()
        except:
            pass
    
    # Final global polish with remaining time
    if best_c is not None and time.time() - start_time < timeout - 3:
        remaining = timeout - 3 - (time.time() - start_time)
        if remaining > 2.0:
            try:
                res = minimize(lambda c: -sum_radii(c), best_c, jac=lambda c: -grad_sum_radii(c),
                               method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-13})
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_c = res.x.copy()
            except:
                pass
        
        if time.time() - start_time < timeout - 1:
            try:
                refined = voronoi_refine(best_c, max_iter=10, time_limit=min(2.0, timeout - 1 - (time.time() - start_time)))
                val = sum_radii(refined)
                if val > best_val:
                    best_val = val
                    best_c = refined.copy()
            except:
                pass

    if best_c is None:
        best_c = create_grid_init(6, 6, N)
        
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
