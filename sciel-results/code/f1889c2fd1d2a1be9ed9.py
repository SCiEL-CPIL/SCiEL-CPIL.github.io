import numpy as np
from scipy.optimize import minimize
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

def grad_sum_radii(c):
    p = c.reshape(N, 2)
    grad = np.zeros_like(p)
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1)
    min_neighbor_idx = np.argmin(dists, axis=1)
    
    for i in range(N):
        wall_r = min_wall_dist[i]
        neigh_r = min_neighbor_dist[i] / 2.0
        
        # Tolerance for active constraints
        tol = 1e-8
        
        active_wall_grads = []
        active_neigh_grads = []
        
        # Check walls
        px, py = p[i]
        walls = [px, 1-px, py, 1-py]
        wall_grads = [[1,0], [-1,0], [0,1], [0,-1]]
        
        if wall_r < neigh_r - tol:
            # Only wall active
            min_val = wall_r
            for w, g in zip(walls, wall_grads):
                if abs(w - min_val) < tol:
                    active_wall_grads.append(g)
        elif neigh_r < wall_r - tol:
            # Only neighbor active
            j = min_neighbor_idx[i]
            d = min_neighbor_dist[i]
            if d > 1e-10:
                vec = p[i] - p[j]
                active_neigh_grads.append(0.5 * vec / d)
                # Store for j later? No, we need to add to grad[i] and grad[j]
                # We handle j's contribution in its own loop or symmetrically.
                # To be safe, let's accumulate contributions to i and j separately.
                # Actually, the current code adds to grad[i] and grad[j] inside the loop.
                # This is correct if we iterate i.
                pass
        else:
            # Both active or close
            min_val = wall_r
            for w, g in zip(walls, wall_grads):
                if abs(w - min_val) < tol:
                    active_wall_grads.append(g)
            
            j = min_neighbor_idx[i]
            d = min_neighbor_dist[i]
            if d > 1e-10:
                vec = p[i] - p[j]
                active_neigh_grads.append(0.5 * vec / d)

        # Average active gradients for i
        if active_wall_grads:
            grad_wall = np.mean(active_wall_grads, axis=0)
        else:
            grad_wall = np.zeros(2)
            
        if active_neigh_grads:
            grad_neigh = np.mean(active_neigh_grads, axis=0)
        else:
            grad_neigh = np.zeros(2)
        
        # The subgradient of min(a,b) is in the convex hull of the gradients of a and b.
        # If both are active, any convex combination is a valid subgradient.
        # Taking the average is a common choice.
        grad[i] += 0.5 * grad_wall + 0.5 * grad_neigh
        
        # We also need to update grad[j] if neighbor was active?
        # The term is 0.5 * dist(i, j).
        # grad of dist(i, j) w.r.t p_i is (p_i - p_j)/dist.
        # grad of dist(i, j) w.r.t p_j is (p_j - p_i)/dist.
        # So if neighbor j is the min for i, we add 0.5 * (p_i - p_j)/dist to grad[i]
        # and 0.5 * (p_j - p_i)/dist to grad[j].
        
        # Let's redo the logic to be symmetric and correct.
        
    # Re-implementing grad for correctness
    grad = np.zeros_like(p)
    for i in range(N):
        wall_r = min_wall_dist[i]
        neigh_r = min_neighbor_dist[i] / 2.0
        
        # Determine if wall or neighbor is the binding constraint
        # We use a small tolerance to handle ties
        if abs(wall_r - neigh_r) < 1e-8:
            # Tie: average gradients
            px, py = p[i]
            walls = [px, 1-px, py, 1-py]
            wall_grads = [[1,0], [-1,0], [0,1], [0,-1]]
            min_val = wall_r
            active_walls = [g for w, g in zip(walls, wall_grads) if abs(w - min_val) < 1e-8]
            g_wall = np.mean(active_walls, axis=0) if active_walls else np.zeros(2)
            
            j = min_neighbor_idx[i]
            d = min_neighbor_dist[i]
            if d > 1e-10:
                vec = p[i] - p[j]
                g_neigh_i = 0.5 * vec / d
                g_neigh_j = -0.5 * vec / d
            else:
                g_neigh_i = np.zeros(2)
                g_neigh_j = np.zeros(2)
            
            grad[i] += 0.5 * g_wall + 0.5 * g_neigh_i
            grad[j] += 0.5 * g_neigh_j
            
        elif wall_r < neigh_r:
            # Wall binding
            px, py = p[i]
            walls = [px, 1-px, py, 1-py]
            wall_grads = [[1,0], [-1,0], [0,1], [0,-1]]
            min_val = wall_r
            active_walls = [g for w, g in zip(walls, wall_grads) if abs(w - min_val) < 1e-8]
            g_wall = np.mean(active_walls, axis=0) if active_walls else np.zeros(2)
            grad[i] += g_wall
        else:
            # Neighbor binding
            j = min_neighbor_idx[i]
            d = min_neighbor_dist[i]
            if d > 1e-10:
                vec = p[i] - p[j]
                grad[i] += 0.5 * vec / d
                grad[j] += -0.5 * vec / d
            # else: overlap, gradient undefined/zero, skip

    return grad.flatten()

def phyllotaxis_points(n, radius=0.5, center=(0.5, 0.5)):
    points = []
    for i in range(n):
        r = radius * np.sqrt(i / n)
        theta = i * 137.508 * np.pi / 180.0
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        x = np.clip(x, 0.01, 0.99)
        y = np.clip(y, 0.01, 0.99)
        points.append([x, y])
    return np.array(points)

def voronoi_local_search(c, max_sweeps=20, time_limit=5.0):
    """
    Refines the packing by iteratively optimizing each circle's position
    to maximize its radius (Voronoi incenter approximation).
    """
    p = c.reshape(N, 2).copy()
    start_time = time.time()
    current_sum = sum_radii(c)
    
    for sweep in range(max_sweeps):
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(N):
            p_i_old = p[i].copy()
            
            # Define objective for circle i: maximize r_i
            # r_i = min( min_wall(p_i), min_j 0.5*dist(p_i, p_j) )
            def neg_r_i(pos):
                pos = np.array(pos)
                dw = np.min(np.minimum(pos, 1-pos))
                diffs = p - pos
                diffs[i] = np.inf
                d_n = np.min(np.sqrt(np.sum(diffs**2, axis=1))) / 2.0
                return -min(dw, d_n)
            
            # Use Powell method for robustness on non-smooth 2D function
            try:
                res = minimize(neg_r_i, p_i_old, method='Powell', 
                               options={'maxiter': 100, 'xtol': 1e-8, 'ftol': 1e-10})
                if res.fun < neg_r_i(p_i_old) - 1e-10:
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
    for jitter_scale in [0.02, 0.04, 0.06, 0.1]:
        for _ in range(10):
            jitter = np.random.uniform(-jitter_scale, jitter_scale, size=(N, 2))
            p = np.clip(base + jitter, 0.01, 0.99)
            inits.append(p.flatten())
            
    # 5. Random
    for _ in range(30):
        p = np.random.uniform(0.05, 0.95, size=(N, 2))
        inits.append(p.flatten())
        
    bounds = [(0.001, 0.999), (0.001, 0.999)] * N
    
    for idx, c_init in enumerate(inits):
        if time.time() - start_time > timeout - 10:
            break
            
        # 1. L-BFGS-B for global structure
        try:
            res = minimize(lambda c: -sum_radii(c), c_init, jac=lambda c: -grad_sum_radii(c),
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
        except:
            pass
        
        # 2. Voronoi Local Search Refinement
        if time.time() - start_time < timeout - 5:
            try:
                # Use the result of LBFGS or the init if LBFGS failed
                start_point = res.x if 'res' in locals() and res.x is not None else c_init
                refined = voronoi_local_search(start_point, max_sweeps=10, time_limit=3.0)
                val = sum_radii(refined)
                if val > best_val:
                    best_val = val
                    best_c = refined.copy()
            except:
                pass

    if best_c is not None:
        # Final Polish with LBFGS
        if time.time() - start_time < timeout - 5:
            try:
                res = minimize(lambda c: -sum_radii(c), best_c, jac=lambda c: -grad_sum_radii(c),
                               method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12})
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_c = res.x.copy()
            except:
                pass
        
        # Final Voronoi Polish
        if time.time() - start_time < timeout - 2:
            try:
                refined = voronoi_local_search(best_c, max_sweeps=5, time_limit=2.0)
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
