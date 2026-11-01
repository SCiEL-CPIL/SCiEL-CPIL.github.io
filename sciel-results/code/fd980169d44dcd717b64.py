import numpy as np
import itertools
from scipy.optimize import minimize
import time

N = 14
D = 3

def _pairwise_distances_squared(points):
    """Compute squared pairwise distances for a set of points."""
    # points: (N, D)
    # Returns (N, N) matrix of squared distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sum(diff ** 2, axis=2)

def _exact_objective(points):
    """Compute the squared ratio of max to min pairwise distance."""
    d2 = _pairwise_distances_squared(points)
    # Mask diagonal to ignore 0 distances
    mask = ~np.eye(N, dtype=bool)
    d2_masked = d2[mask]
    min_d2 = np.min(d2_masked)
    max_d2 = np.max(d2_masked)
    if min_d2 <= 1e-10:
        return 1e10
    return (max_d2 / min_d2)

def _soft_objective(points, tau):
    """
    Compute a smooth surrogate for the objective.
    Minimizes log(max_d2) - log(min_d2) approximated by log-sum-exp.
    """
    d2 = _pairwise_distances_squared(points)
    mask = ~np.eye(N, dtype=bool)
    d2_masked = d2[mask]
    
    # Soft max
    # log(sum(exp(x/tau))) * tau approx max(x)
    # We want log(max(d2)) - log(min(d2))
    # log(max(d2)) approx tau * log(sum(exp(d2/tau)))
    # log(min(d2)) approx -tau * log(sum(exp(-d2/tau)))
    
    # To avoid overflow, shift by max/min
    max_d2_val = np.max(d2_masked)
    min_d2_val = np.min(d2_masked)
    
    # Soft max term
    exp_max = np.exp((d2_masked - max_d2_val) / tau)
    log_sum_max = max_d2_val / tau + np.log(np.sum(exp_max))
    soft_max_d2 = tau * log_sum_max
    
    # Soft min term
    exp_min = np.exp((min_d2_val - d2_masked) / tau)
    log_sum_min = min_d2_val / tau + np.log(np.sum(exp_min))
    soft_min_d2 = tau * log_sum_min
    
    # We want to minimize soft_max_d2 / soft_min_d2
    # Or minimize log(soft_max_d2) - log(soft_min_d2)
    if soft_min_d2 <= 1e-10:
        return 1e10
    return np.log(soft_max_d2) - np.log(soft_min_d2)

def _soft_objective_and_grad(points, tau):
    """
    Compute soft objective and its gradient.
    """
    d2 = _pairwise_distances_squared(points)
    mask = ~np.eye(N, dtype=bool)
    d2_masked = d2[mask]
    
    max_d2_val = np.max(d2_masked)
    min_d2_val = np.min(d2_masked)
    
    # Soft max
    exp_max = np.exp((d2_masked - max_d2_val) / tau)
    sum_exp_max = np.sum(exp_max)
    log_sum_max = max_d2_val / tau + np.log(sum_exp_max)
    soft_max_d2 = tau * log_sum_max
    
    # Soft min
    exp_min = np.exp((min_d2_val - d2_masked) / tau)
    sum_exp_min = np.sum(exp_min)
    log_sum_min = min_d2_val / tau + np.log(sum_exp_min)
    soft_min_d2 = tau * log_sum_min
    
    obj = np.log(soft_max_d2) - np.log(soft_min_d2)
    
    # Gradient
    # d/dx soft_max_d2 = (1/soft_max_d2) * d(soft_max_d2)/dx
    # d(soft_max_d2)/dx = d(tau * log_sum_max)/dx
    # log_sum_max = max/tau + log(sum(exp((d2-max)/tau)))
    # d(log_sum_max)/dx = (1/sum_exp_max) * sum(exp((d2-max)/tau) * (1/tau) * d(d2)/dx)
    # d(soft_max_d2)/dx = tau * (1/sum_exp_max) * (1/tau) * sum(exp_max * d(d2)/dx)
    # = (1/sum_exp_max) * sum(exp_max * d(d2)/dx)
    # = sum( (exp_max/sum_exp_max) * d(d2)/dx )
    
    # Similarly for soft_min
    # d(soft_min_d2)/dx = sum( (exp_min/sum_exp_min) * d(d2)/dx )
    
    # d(d2)/dx:
    # d2_ij = ||x_i - x_j||^2
    # d(d2_ij)/dx_i = 2(x_i - x_j)
    # d(d2_ij)/dx_j = 2(x_j - x_i)
    
    grad = np.zeros_like(points)
    
    # Compute d(d2)/dx for all pairs
    # This is efficient if we do it vectorized
    
    # We need to sum over all pairs (i,j) with i<j or all i!=j
    # Let's use the full matrix approach for gradient
    
    # d2 is (N, N)
    # diff is (N, N, D)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    
    # Weights for max
    w_max = exp_max / sum_exp_max
    # Reshape w_max to (N, N)
    w_max_full = np.zeros((N, N))
    w_max_full[mask] = w_max
    
    # Weights for min
    w_min = exp_min / sum_exp_min
    w_min_full = np.zeros((N, N))
    w_min_full[mask] = w_min
    
    # Gradient of soft_max_d2 w.r.t points
    # sum_{i,j} w_max_full[i,j] * 2 * (points[i] - points[j])
    # This can be computed as:
    # For each point i: sum_j w_max_full[i,j] * 2 * (points[i] - points[j])
    # = 2 * (points[i] * sum_j w_max_full[i,j] - sum_j w_max_full[i,j] * points[j])
    
    row_sums_max = np.sum(w_max_full, axis=1)
    weighted_points_max = w_max_full @ points
    
    grad_max = 2 * (points * row_sums_max[:, np.newaxis] - weighted_points_max)
    
    row_sums_min = np.sum(w_min_full, axis=1)
    weighted_points_min = w_min_full @ points
    
    grad_min = 2 * (points * row_sums_min[:, np.newaxis] - weighted_points_min)
    
    # Objective is log(soft_max) - log(soft_min)
    # Grad = (1/soft_max) * grad_max - (1/soft_min) * grad_min
    
    grad = (1.0 / soft_max_d2) * grad_max - (1.0 / soft_min_d2) * grad_min
    
    return obj, grad

def _generate_initializations():
    """Generate diverse initial points."""
    inits = []
    
    # 1. Random in sphere
    for _ in range(5):
        pts = np.random.randn(N, D)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)
        
    # 2. Fibonacci Sphere
    # Approximate uniform distribution on sphere
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(N)
    z = 1 - 2 * (indices + 0.5) / N
    radius = np.sqrt(1 - z**2)
    theta = 2 * np.pi * indices / golden_ratio
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    pts = np.column_stack((x, y, z))
    inits.append(pts)
    
    # 3. BCC Lattice
    # Generate a grid and take a subset
    side = 3
    grid = list(itertools.product(range(side), repeat=D))
    # Take first N
    pts_grid = np.array(grid[:N], dtype=float)
    # Center and scale
    pts_grid -= np.mean(pts_grid, axis=0)
    pts_grid /= np.linalg.norm(pts_grid, axis=1).max()
    inits.append(pts_grid)
    
    # 4. Concentric Shells
    # 8 points on inner shell, 6 on outer
    pts_shell = np.zeros((N, D))
    # Inner: 8 points of a cube
    cube = np.array(list(itertools.product([-1, 1], repeat=3)))
    pts_shell[:8] = cube * 0.5
    # Outer: 6 points of an octahedron
    octa = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]])
    pts_shell[8:] = octa * 1.5
    inits.append(pts_shell)
    
    return inits

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    best_score = 1e10
    best_points = None
    
    inits = _generate_initializations()
    
    # Taus for annealing
    taus = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02]
    
    for init in inits:
        if time.time() > deadline:
            break
            
        current_points = init.copy()
        
        # Annealing loop
        for tau in taus:
            if time.time() > deadline:
                break
                
            # Use L-BFGS-B with gradient
            try:
                res = minimize(
                    _soft_objective_and_grad, 
                    current_points, 
                    args=(tau,), 
                    method='L-BFGS-B', 
                    jac=True,
                    options={'maxiter': 100, 'maxls': 50}
                )
                current_points = res.x
            except Exception:
                pass
        
        # Final polish with exact objective
        if time.time() < deadline - 5:
            try:
                res = minimize(
                    _exact_objective, 
                    current_points, 
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                current_points = res.x
            except Exception:
                pass
        
        score = _exact_objective(current_points)
        if score < best_score:
            best_score = score
            best_points = current_points.copy()
            
    if best_points is None:
        # Fallback to grid
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_points = np.array(grid, dtype=float)
        
    return best_points
