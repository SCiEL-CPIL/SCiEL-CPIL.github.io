"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_radius(pts):
    """Compute the minimum radius for a given set of centers."""
    n = len(pts)
    # Distance to boundary
    boundary_dist = 1.0 - np.linalg.norm(pts, axis=1)
    min_boundary = np.min(boundary_dist)
    
    # Pairwise distances
    min_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_pair:
                min_pair = d
    min_pair = min_pair / 2.0
    
    return min(min_boundary, min_pair)


def _soft_min_radius(pts, beta=100.0):
    """
    Compute a soft-min approximation of the radius using log-sum-exp.
    This provides a smooth objective for optimization.
    """
    n = len(pts)
    
    # Boundary distances
    boundary_dist = 1.0 - np.linalg.norm(pts, axis=1)
    
    # Pairwise distances (half)
    pair_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j]) / 2.0
            pair_dists.append(d)
    pair_dists = np.array(pair_dists)
    
    # Combine all constraints into one array
    all_dists = np.concatenate([boundary_dist, pair_dists])
    
    # Soft min via log-sum-exp
    # min(x) ≈ -1/beta * log(sum(exp(-beta * x)))
    # For numerical stability, use the standard log-sum-exp trick
    min_val = np.min(all_dists)
    if min_val < -100:  # Prevent overflow
        return min_val
    
    exp_vals = np.exp(-beta * (all_dists - min_val))
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0:
        return min_val
    
    soft_min = min_val - np.log(sum_exp) / beta
    
    return soft_min


def _soft_min_radius_grad(pts, beta=100.0):
    """
    Compute the gradient of the soft-min radius.
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    
    # Boundary distances
    norms = np.linalg.norm(pts, axis=1)
    boundary_dist = 1.0 - norms
    
    # Pairwise distances (half)
    pair_indices = []
    pair_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j]) / 2.0
            pair_dists.append(d)
            pair_indices.append((i, j))
    pair_dists = np.array(pair_dists)
    
    # Combine all constraints
    all_dists = np.concatenate([boundary_dist, pair_dists])
    
    min_val = np.min(all_dists)
    if min_val < -100:
        return grad
    
    exp_vals = np.exp(-beta * (all_dists - min_val))
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0:
        return grad
    
    # Weights for soft min
    weights = exp_vals / sum_exp
    
    # Gradient contribution from boundary
    for i in range(n):
        w = weights[i]
        if norms[i] > 1e-10:
            grad[i] += w * (pts[i] / norms[i])  # d/dx (1 - ||x||) = -x/||x||, so we want to maximize, hence positive
        # Actually, we want to maximize the soft min, so gradient of -soft_min is what we minimize
        # Let's think: we want to maximize min(dists). 
        # soft_min = min_val - log(sum(exp(-beta*(d-min))))/beta
        # d(soft_min)/d(pts) = sum_i w_i * d(dist_i)/d(pts)
        # where w_i = exp(-beta*(d_i - min)) / sum(exp(...))
        # For boundary: dist_i = 1 - ||p_i||, d(dist_i)/d(p_i) = -p_i/||p_i||
        # For pair: dist_ij = ||p_i - p_j||/2, d/d(p_i) = (p_i - p_j)/||p_i - p_j||
    
    # Let's recompute properly
    grad = np.zeros_like(pts)
    
    # Boundary terms
    for i in range(n):
        w = weights[i]
        if norms[i] > 1e-10:
            grad[i] += w * (-pts[i] / norms[i])
    
    # Pair terms
    for k, (i, j) in enumerate(pair_indices):
        w = weights[n + k]
        diff = pts[i] - pts[j]
        norm_diff = np.linalg.norm(diff)
        if norm_diff > 1e-10:
            grad[i] += w * (diff / norm_diff)
            grad[j] += w * (-diff / norm_diff)
    
    return grad


def _hex_init(spacing):
    """Generate hexagonal lattice points."""
    k = int(1.0 / spacing) + 2
    rows = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            rows.append([x, y])
    points = np.array(rows)
    radius = np.linalg.norm(points, axis=1)
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    return points


def _ring_init():
    """Generate ring-based initialization."""
    points = [[0.0, 0.0]]
    
    # Inner ring
    r1 = 0.3
    n1 = 6
    for i in range(n1):
        angle = 2 * np.pi * i / n1
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    # Outer ring
    r2 = 0.65
    n2 = 12
    for i in range(n2):
        angle = 2 * np.pi * i / n2 + np.pi / 12
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    
    # Second outer ring
    r3 = 0.85
    n3 = 18
    for i in range(n3):
        angle = 2 * np.pi * i / n3
        points.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    
    return np.array(points)


def _random_init(seed):
    """Generate random initialization."""
    np.random.seed(seed)
    points = []
    for _ in range(N):
        r = np.sqrt(np.random.rand())
        theta = 2 * np.pi * np.random.rand()
        points.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(points)


def _optimize_from_init(init_pts, beta_start=50.0, beta_end=200.0, n_anneal=10, deadline=None):
    """
    Optimize from an initial configuration using annealing.
    """
    if deadline is not None and time.time() > deadline:
        return init_pts, _compute_radius(init_pts)
    
    best_pts = init_pts.copy()
    best_r = _compute_radius(init_pts)
    
    n_anneal = max(1, min(n_anneal, 15))
    
    for step in range(n_anneal):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        # Annealing schedule
        t = step / max(1, n_anneal - 1)
        beta = beta_start * (beta_end / beta_start) ** t
        
        # Use L-BFGS-B to maximize soft_min_radius (i.e., minimize negative soft_min)
        def objective(x):
            pts = x.reshape(-1, 2)
            return -_soft_min_radius(pts, beta)
        
        def gradient(x):
            pts = x.reshape(-1, 2)
            return -_soft_min_radius_grad(pts, beta)
        
        x0 = best_pts.flatten()
        
        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < objective(x0):
                new_pts = res.x.reshape(-1, 2)
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_pts = new_pts
                    best_r = new_r
        except Exception:
            pass
    
    return best_pts, best_r


def _nelder_mead_polish(pts, deadline=None):
    """Polish with Nelder-Mead for local refinement."""
    if deadline is not None and time.time() > deadline - 5:
        return pts
    
    def objective(x):
        p = x.reshape(-1, 2)
        return -_compute_radius(p)
    
    try:
        res = minimize(objective, pts.flatten(), method='Nelder-Mead',
                      options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.fun < objective(pts.flatten()):
            return res.x.reshape(-1, 2)
    except Exception:
        pass
    
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Generate diverse initializations
    inits = []
    
    # Hexagonal lattices with different spacings
    for spacing in [0.2, 0.22, 0.24, 0.26]:
        pts = _hex_init(spacing)
        if len(pts) >= N:
            # Take the N closest to center
            radii = np.linalg.norm(pts, axis=1)
            idx = np.argsort(radii, kind='stable')[:N]
            inits.append(pts[idx])
        elif len(pts) > 0:
            inits.append(pts)
    
    # Ring initialization
    ring_pts = _ring_init()
    if len(ring_pts) >= N:
        inits.append(ring_pts[:N])
    elif len(ring_pts) > 0:
        inits.append(ring_pts)
    
    # Random initializations
    for seed in [42, 123, 456]:
        if time.time() > deadline - 20:
            break
        inits.append(_random_init(seed))
    
    # Filter valid inits
    valid_inits = []
    for pts in inits:
        if len(pts) == N:
            r = _compute_radius(pts)
            if r > 0:
                valid_inits.append((pts, r))
    
    if not valid_inits:
        # Fallback: simple hex grid
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98
    
    # Sort by initial radius
    valid_inits.sort(key=lambda x: x[1], reverse=True)
    
    best_pts = valid_inits[0][0]
    best_r = valid_inits[0][1]
    
    # Optimize from each initialization
    for i, (init_pts, init_r) in enumerate(valid_inits):
        if time.time() > deadline - 15:
            break
        
        # Skip if already very good
        if init_r > 0.125:
            # Polish directly
            polished = _nelder_mead_polish(init_pts, deadline)
            r = _compute_radius(polished)
            if r > best_r:
                best_pts = polished
                best_r = r
            continue
        
        # Annealing optimization
        optimized_pts, optimized_r = _optimize_from_init(
            init_pts, 
            beta_start=50.0, 
            beta_end=200.0, 
            n_anneal=8,
            deadline=deadline
        )
        
        if optimized_r > best_r:
            best_pts = optimized_pts
            best_r = optimized_r
        
        # Final polish
        if time.time() < deadline - 10:
            polished = _nelder_mead_polish(best_pts, deadline)
            r = _compute_radius(polished)
            if r > best_r:
                best_pts = polished
                best_r = r
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        # Use higher beta for final L-BFGS pass
        def final_objective(x):
            pts = x.reshape(-1, 2)
            return -_soft_min_radius(pts, beta=500.0)
        
        def final_gradient(x):
            pts = x.reshape(-1, 2)
            return -_soft_min_radius_grad(pts, beta=500.0)
        
        try:
            res = minimize(final_objective, best_pts.flatten(), jac=final_gradient, 
                          method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < final_objective(best_pts.flatten()):
                new_pts = res.x.reshape(-1, 2)
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_pts = new_pts
                    best_r = new_r
        except Exception:
            pass
    
    # Nelder-Mead final polish
    if time.time() < deadline - 2:
        polished = _nelder_mead_polish(best_pts, deadline)
        r = _compute_radius(polished)
        if r > best_r:
            best_pts = polished
            best_r = r
    
    return best_pts
# EVOLVE-BLOCK-END
