"""Improved program for packing 300 equal circles in a unit circle."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 300

def _compute_radius(points):
    """Compute the feasible packing radius for a set of points."""
    if points is None or len(points) == 0:
        return 0.0
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dist_boundary)
    d = pdist(points)
    min_dist = np.min(d)
    min_pair = min_dist / 2.0
    return min(min_boundary, min_pair)

def _generate_hexagonal(n_points, margin=0.3):
    """Generate a hexagonal lattice configuration."""
    # Estimate spacing
    area_per_point = np.pi / n_points
    spacing = np.sqrt(area_per_point / (np.sqrt(3) / 2))
    
    k = int(1.0 / spacing) + 5
    points = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3) / 2
            if x * x + y * y <= (1.0 - margin) ** 2:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) >= n_points:
        # Sort by distance from origin and take closest n_points
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n_points]
        return points[idx]
    elif len(points) > 0:
        return points
    else:
        # Fallback: random
        return np.random.randn(n_points, 2) * 0.5

def _negative_radius_softmin(points, beta=20.0):
    """Objective: minimize -soft_min of (pair distances, boundary distances)."""
    n = len(points)
    d = pdist(points)
    min_boundary = np.min(1.0 - np.linalg.norm(points, axis=1))
    
    # Combine pair distances and boundary distances
    # We want to maximize min(d_i/2 for pairs, 1-r_i for boundaries)
    # Using soft min: -log(sum(exp(-beta*x)))/beta
    values = np.concatenate([d / 2.0, [min_boundary] * n])
    # Actually let's be more careful
    # We have C(n,2) pair distances and n boundary distances
    pair_vals = d / 2.0
    boundary_vals = 1.0 - np.linalg.norm(points, axis=1)
    
    all_vals = np.concatenate([pair_vals, boundary_vals])
    
    # Soft min
    min_val = -np.log(np.sum(np.exp(-beta * all_vals))) / beta
    return -min_val

def _optimize_gradient(points, max_time=60.0):
    """L-BFGS-B optimization on soft-min objective."""
    start_time = time.time()
    
    def objective(x):
        if time.time() - start_time > max_time:
            return np.inf
        pts = x.reshape(-1, 2)
        return _negative_radius_softmin(pts, beta=25.0)
    
    x0 = points.flatten()
    
    result = minimize(
        objective, x0, method='L-BFGS-B',
        options={'maxiter': 1000, 'maxfun': 5000, 'maxls': 50}
    )
    
    if result.success or result.fun < np.inf:
        return result.x.reshape(-1, 2)
    return points

def _slsqp_optimize(points, max_time=90.0):
    """SLSQP with explicit constraints."""
    start_time = time.time()
    n = len(points)
    
    def objective(x):
        if time.time() - start_time > max_time:
            return np.inf
        pts = x.reshape(-1, 2)
        d = pdist(pts)
        min_pair = np.min(d) / 2.0
        min_boundary = np.min(1.0 - np.linalg.norm(pts, axis=1))
        return -min(min_pair, min_boundary)
    
    def objective_grad(x):
        if time.time() - start_time > max_time:
            return np.zeros_like(x)
        pts = x.reshape(-1, 2)
        d = pdist(pts)
        min_pair = np.min(d) / 2.0
        min_boundary = np.min(1.0 - np.linalg.norm(pts, axis=1))
        r = min(min_pair, min_boundary)
        
        # Approximate gradient via finite differences for key constraints
        grad = np.zeros_like(x)
        eps = 1e-5
        
        if r == min_pair:
            # Find closest pair
            d_mat = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    diff = pts[i] - pts[j]
                    dist = np.linalg.norm(diff)
                    d_mat[i, j] = d_mat[j, i] = dist
            min_idx = np.unravel_index(np.argmin(d_mat), d_mat.shape)
            i, j = min_idx[0], min_idx[1]
            if d_mat[i, j] < 1e-10:
                return grad
            diff = pts[i] - pts[j]
            unit = diff / d_mat[i, j]
            grad[i*2] += 0.5 * unit[0] / eps
            grad[i*2+1] += 0.5 * unit[1] / eps
            grad[j*2] -= 0.5 * unit[0] / eps
            grad[j*2+1] -= 0.5 * unit[1] / eps
        else:
            # Boundary constraint active
            idx = np.argmin(1.0 - np.linalg.norm(pts, axis=1))
            norm = np.linalg.norm(pts[idx])
            if norm < 1e-10:
                return grad
            unit = pts[idx] / norm
            grad[idx*2] += unit[0]
            grad[idx*2+1] += unit[1]
        
        return grad
    
    # Constraints: points inside unit circle
    def constraint_func(x):
        if time.time() - start_time > max_time:
            return np.zeros(n)
        pts = x.reshape(-1, 2)
        return 1.0 - np.linalg.norm(pts, axis=1)
    
    x0 = points.flatten()
    
    try:
        result = minimize(
            objective, x0, method='SLSQP',
            constraints=[{'type': 'ineq', 'fun': constraint_func}],
            options={'maxiter': 300, 'ftol': 1e-8, 'maxfun': 2000}
        )
        if result.success or result.fun < np.inf:
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    
    return points

def _simulated_annealing(points, max_time=30.0, n_iter=5000):
    """Simulated annealing for local refinement."""
    start_time = time.time()
    pts = points.copy()
    n = len(pts)
    
    current_r = _compute_radius(pts)
    best_r = current_r
    best_pts = pts.copy()
    
    T = 0.01
    T_min = 0.0001
    cooling = 0.999
    
    for it in range(n_iter):
        if time.time() - start_time > max_time:
            break
        
        # Random perturbation
        idx = np.random.randint(n)
        delta = np.random.randn(2) * T
        pts[idx] += delta
        
        # Keep inside circle
        norm = np.linalg.norm(pts[idx])
        if norm > 1.0:
            pts[idx] *= 0.999 / norm
        
        new_r = _compute_radius(pts)
        
        if new_r > current_r:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_pts = pts.copy()
        else:
            # Accept with probability
            if np.random.rand() < np.exp((new_r - current_r) / T):
                current_r = new_r
        
        T = max(T_min, T * cooling)
    
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    # Step 1: Generate initial hexagonal lattice
    points = _generate_hexagonal(N, margin=0.2)
    
    if len(points) != N:
        # Ensure exactly N points
        if len(points) > N:
            points = points[:N]
        else:
            while len(points) < N:
                points = np.vstack([points, np.random.randn(1, 2) * 0.5])
    
    best_points = points
    best_r = _compute_radius(points)
    
    # Step 2: L-BFGS-B gradient optimization
    if time.time() < deadline - 40:
        try:
            opt_pts = _optimize_gradient(points, max_time=35.0)
            r = _compute_radius(opt_pts)
            if r > best_r:
                best_r = r
                best_points = opt_pts
        except Exception:
            pass
    
    # Step 3: SLSQP with constraints
    if time.time() < deadline - 20:
        try:
            slsqp_pts = _slsqp_optimize(best_points, max_time=15.0)
            r = _compute_radius(slsqp_pts)
            if r > best_r:
                best_r = r
                best_points = slsqp_pts
        except Exception:
            pass
    
    # Step 4: Simulated annealing refinement
    if time.time() < deadline - 5:
        try:
            sa_pts = _simulated_annealing(best_points, max_time=8.0, n_iter=3000)
            r = _compute_radius(sa_pts)
            if r > best_r:
                best_r = r
                best_points = sa_pts
        except Exception:
            pass
    
    # Final validation: ensure all points are inside unit circle
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms <= 1.0
    if np.sum(mask) < N:
        # Push violating points inward
        for i in range(N):
            if norms[i] > 1.0:
                best_points[i] *= 0.99 / norms[i]
    
    return best_points
