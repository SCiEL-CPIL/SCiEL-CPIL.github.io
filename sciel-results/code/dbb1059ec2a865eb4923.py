"""Circle packing in unit square: pack 26 circles, maximize sum of radii.
Strategy: Voronoi dual representation - radius = inradius of Voronoi cell."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _voronoi_radii(x):
    """
    Compute Voronoi inradii for N centers given as (N,2) array.
    r_i = min(dist to other centers, dist to walls)
    """
    n = x.shape[0]
    # Distances to walls
    wall_dist = np.minimum(np.minimum(x[:, 0], 1.0 - x[:, 0]),
                           np.minimum(x[:, 1], 1.0 - x[:, 1]))
    # Distances to other centers
    # For each i, min over j != i of ||x_i - x_j||
    dx = x[:, 0, np.newaxis] - x[np.newaxis, :, 0]
    dy = x[:, 1, np.newaxis] - x[np.newaxis, :, 1]
    dists = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dists, np.inf)
    center_dist = np.min(dists, axis=1)
    return np.minimum(wall_dist, center_dist)


def _smooth_voronoi_sum(x, beta=200.0):
    """
    Smooth approximation of sum of Voronoi inradii.
    Uses soft-min: softmin(a,b) = -(1/beta)*logaddexp(-beta*a, -beta*b)
    """
    n = x.shape[0]
    wall_dist = np.minimum(np.minimum(x[:, 0], 1.0 - x[:, 0]),
                           np.minimum(x[:, 1], 1.0 - x[:, 1]))
    # Pairwise distances
    dx = x[:, 0, np.newaxis] - x[np.newaxis, :, 0]
    dy = x[:, 1, np.newaxis] - x[np.newaxis, :, 1]
    dists = np.sqrt(dx * dx + dy * dy + 1e-30)
    np.fill_diagonal(dists, np.inf)
    
    # For each i, compute soft-min over all constraints (walls + other centers)
    total = 0.0
    for i in range(n):
        # Constraints for circle i: wall_dist[i] and dists[i, j] for j != i
        constraints = np.concatenate([wall_dist[i:i+1], dists[i, :]])
        # softmin = -(1/beta) * log(sum(exp(-beta * c)))
        # Use log-sum-exp for stability
        mc = np.min(constraints)
        total += mc - np.log(np.sum(np.exp(-beta * (constraints - mc)))) / beta
    return total


def _smooth_voronoi_grad(x, beta=200.0):
    """
    Gradient of smooth sum of Voronoi inradii.
    d/dx_i of softmin(c_1, ..., c_m) where c_k are the constraint values.
    softmin = mc - (1/beta)*log(sum(exp(-beta*(c_k - mc))))
    d(softmin)/dx = sum_k w_k * dc_k/dx where w_k = exp(-beta*(c_k - mc)) / sum(exp(-beta*(c_j - mc)))
    """
    n = x.shape[0]
    grad = np.zeros_like(x)
    
    for i in range(n):
        wall_dist = np.minimum(np.minimum(x[i, 0], 1.0 - x[i, 0]),
                               np.minimum(x[i, 1], 1.0 - x[i, 1]))
        # Get all constraint values for circle i
        dx = x[i, 0] - x[:, 0]
        dy = x[i, 1] - x[:, 1]
        dists = np.sqrt(dx * dx + dy * dy + 1e-30)
        dists[i] = np.inf
        
        constraints = np.concatenate([wall_dist, dists])
        mc = np.min(constraints)
        
        # Weights: w_k = exp(-beta*(c_k - mc)) / sum(exp(-beta*(c_j - mc)))
        exps = np.exp(-beta * (constraints - mc))
        w = exps / np.sum(exps)
        
        # Gradient of each constraint w.r.t. x[i]
        # Wall: d(wall_dist)/dx[i] depends on which wall is closest
        # For the softmin gradient, we need d(constraint_k)/dx[i] for each k
        
        # Wall constraint gradient
        # wall_dist = min(x[i,0], 1-x[i,0], x[i,1], 1-x[i,1])
        # Its gradient is the gradient of the minimum component
        wx = np.array([x[i, 0], 1.0 - x[i, 0], x[i, 1], 1.0 - x[i, 1]])
        min_w_idx = np.argmin(wx)
        wall_grad = np.zeros(2)
        if min_w_idx == 0:
            wall_grad[0] = 1.0
        elif min_w_idx == 1:
            wall_grad[0] = -1.0
        elif min_w_idx == 2:
            wall_grad[1] = 1.0
        else:
            wall_grad[1] = -1.0
        
        # Center distance gradients: d||x_i - x_j||/dx_i = (x_i - x_j)/||x_i - x_j||
        for j in range(n):
            if j == i:
                continue
            d = dists[i, j]
            if d < 1e-15:
                continue
            # Gradient of dist(i,j) w.r.t. x[i]
            g = np.array([dx[j] / d, dy[j] / d])
            grad[i] += w[j + 1] * g  # w[j+1] because constraints[0] is wall, constraints[1:] are dists
        
        # Wall contribution
        grad[i] += w[0] * wall_grad
    
    return grad


def _true_voronoi_sum(x):
    """Exact sum of Voronoi inradii (non-smooth)."""
    r = _voronoi_radii(x)
    return float(np.sum(r))


def _hex_init(n):
    """Hexagonal close-packed initialization."""
    points = []
    row = 0
    while len(points) < n:
        x0 = 0.5 / (row + 1) if row % 2 == 0 else 0.0
        for col in range(row + 1):
            if len(points) >= n:
                break
            x = (col + 0.5) / (row + 1) if row % 2 == 0 else (col + 1.0) / (row + 1)
            y = (row + 0.5) / (row + 1)
            points.append([x, y])
        row += 1
    # Scale to fit in [0.05, 0.95]
    pts = np.array(points[:n])
    pts = 0.05 + 0.9 * pts
    return pts


def _grid_init(n):
    """Grid initialization."""
    side = int(np.ceil(np.sqrt(n)))
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) < n:
                pts.append([(i + 0.5) / side, (j + 0.5) / side])
    while len(pts) < n:
        pts.append([0.5, 0.5])
    return np.array(pts[:n])


def _voronoi_optimize(x0, deadline, beta=200.0, method='L-BFGS-B'):
    """
    Optimize center positions to maximize sum of Voronoi inradii.
    """
    n = x0.shape[0]
    
    def neg_obj(v):
        if time.time() > deadline:
            return 1e10
        return -_smooth_voronoi_sum(v, beta)
    
    def neg_grad(v):
        if time.time() > deadline:
            return np.zeros(2 * n)
        return -_smooth_voronoi_grad(v, beta)
    
    bounds = [(0.005, 0.995)] * (2 * n)
    v0 = x0.ravel()
    
    try:
        result = minimize(neg_obj, v0, jac=neg_grad, method=method,
                          bounds=bounds,
                          options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10,
                                   'maxfun': 100000})
        x_opt = result.x.reshape(n, 2)
    except Exception:
        x_opt = x0
    
    # Verify and return
    total = _true_voronoi_sum(x_opt)
    return x_opt, total


def _voronoi_anneal(x0, deadline, seed=42):
    """
    Simulated annealing on Voronoi inradius sum.
    Moves: perturb a center position.
    """
    rng = np.random.RandomState(seed)
    n = x0.shape[0]
    cur = x0.copy()
    cur_val = _true_voronoi_sum(cur)
    
    best = cur.copy()
    best_val = cur_val
    
    T = 0.02
    T_end = 0.0001
    max_iter = 80000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        i = rng.randint(n)
        scale = 0.005 + 0.02 * (T / T)
        new = cur.copy()
        new[i, 0] += rng.uniform(-scale, scale)
        new[i, 1] += rng.uniform(-scale, scale)
        new[i, 0] = np.clip(new[i, 0], 0.005, 0.995)
        new[i, 1] = np.clip(new[i, 1], 0.005, 0.995)
        
        new_val = _true_voronoi_sum(new)
        delta = new_val - cur_val
        
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            cur = new
            cur_val = new_val
            if cur_val > best_val:
                best_val = cur_val
                best = cur.copy()
        
        T *= 0.9997
    
    return best, best_val


def _local_voronoi_polish(x, deadline, n_sweeps=50):
    """
    Coordinate descent: for each circle, find the best position that maximizes
    its inradius given others fixed. Use fine grid around current position.
    """
    n = x.shape[0]
    cur = x.copy()
    cur_val = _true_voronoi_sum(cur)
    best = cur.copy()
    best_val = cur_val
    
    for sweep in range(n_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            # Try moving circle i on a fine grid
            best_r_i = _voronoi_radii(cur)[i]
            best_pos = cur[i].copy()
            
            # Local grid search
            for dx in np.linspace(-0.03, 0.03, 13):
                for dy in np.linspace(-0.03, 0.03, 13):
                    test = cur.copy()
                    test[i, 0] = np.clip(cur[i, 0] + dx, 0.005, 0.995)
                    test[i, 1] = np.clip(cur[i, 1] + dy, 0.005, 0.995)
                    r_i = _voronoi_radii(test)[i]
                    if r_i > best_r_i + 1e-12:
                        best_r_i = r_i
                        best_pos = test[i].copy()
            
            if best_r_i > cur_val - _true_voronoi_sum(cur) + 1e-12 or best_pos[0] != cur[i, 0] or best_pos[1] != cur[i, 1]:
                cur[i] = best_pos
                new_val = _true_voronoi_sum(cur)
                if new_val > best_val:
                    best_val = new_val
                    best = cur.copy()
                improved = True
        
        if not improved:
            break
    
    return best, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    best_x = None
    best_val = 0.0
    
    # Phase 1: Multiple initializations + Voronoi gradient optimization
    inits = []
    inits.append(_hex_init(N))
    inits.append(_grid_init(N))
    
    # Random inits
    for seed in [1, 2, 3, 4, 5]:
        rng = np.random.RandomState(seed)
        inits.append(rng.uniform(0.1, 0.9, size=(N, 2)))
    
    for x0 in inits:
        if time.time() > deadline - 10:
            break
        x_opt, val = _voronoi_optimize(x0, deadline - 5)
        if val > best_val:
            best_val = val
            best_x = x_opt
    
    # Phase 2: SA refinement from best
    if best_x is not None and time.time() < deadline - 20:
        for seed in [100, 200, 300]:
            if time.time() > deadline - 10:
                break
            x_sa, val_sa = _voronoi_anneal(best_x.copy(), deadline - 5, seed=seed)
            if val_sa > best_val:
                best_val = val_sa
                best_x = x_sa
    
    # Phase 3: Local Voronoi polish
    if best_x is not None and time.time() < deadline - 10:
        x_pol, val_pol = _local_voronoi_polish(best_x.copy(), deadline - 3)
        if val_pol > best_val:
            best_val = val_pol
            best_x = x_pol
    
    # Phase 4: Final gradient polish with higher beta
    if best_x is not None and time.time() < deadline - 3:
        x_final, val_final = _voronoi_optimize(best_x.copy(), deadline, beta=500.0)
        if val_final > best_val:
            best_val = val_final
            best_x = x_final
    
    # Build result
    if best_x is None:
        # Fallback
        x_fallback = _grid_init(N)
        best_x = x_fallback
        best_val = _true_voronoi_sum(x_fallback)
    
    radii = _voronoi_radii(best_x)
    
    # Final safety: clip to ensure validity
    result = np.zeros((N, 3))
    for i in range(N):
        x, y = best_x[i, 0], best_x[i, 1]
        r = radii[i]
        # Ensure circle is inside [0,1]^2
        r = min(r, x, 1.0 - x, y, 1.0 - y)
        r = max(r, 0.001)
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        result[i] = [x, y, r]
    
    return result
# EVOLVE-BLOCK-END
