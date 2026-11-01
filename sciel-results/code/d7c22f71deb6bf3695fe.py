"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize

def _pairwise_distances_squared(coords):
    """Calculate squared distances between all pairs."""
    # coords: (N, D)
    # diff: (N, N, D)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    return dist_sq[mask]

def _objective_and_grad(coords, beta):
    """
    Objective: log(softmax(d^2)) - log(softmin(d^2))
    Returns value and gradient.
    """
    dist_sq = _pairwise_distances_squared(coords)
    
    # Soft max
    # To avoid overflow, subtract max
    max_ds = np.max(dist_sq)
    exp_max = np.exp(beta * (dist_sq - max_ds))
    sum_exp_max = np.sum(exp_max)
    # S_max = (1/beta) * log(sum_exp_max) + max_ds
    # We want log(S_max). 
    # log(S_max) = log( (1/beta) * exp(beta*max_ds) * sum(exp(beta*(d^2 - max_ds))) )
    # = log(1/beta) + beta*max_ds + log(sum_exp_max)
    # Wait, S_max is an approximation of max(d^2).
    # Let's just compute S_max directly.
    S_max = max_ds + (1.0 / beta) * np.log(sum_exp_max)
    
    # Soft min
    min_ds = np.min(dist_sq)
    exp_min = np.exp(-beta * (dist_sq - min_ds))
    sum_exp_min = np.sum(exp_min)
    # S_min = min_ds - (1/beta) * log(sum_exp_min)
    S_min = min_ds - (1.0 / beta) * np.log(sum_exp_min)
    
    # Avoid log(0) or negative values (should not happen if distinct)
    S_max = np.maximum(S_max, 1e-10)
    S_min = np.maximum(S_min, 1e-10)
    
    val = np.log(S_max) - np.log(S_min)
    
    # Gradient
    # d(log S_max)/dx = (1/S_max) * d(S_max)/dx
    # d(S_max)/dx = (1/beta) * d(log(sum_exp_max))/dx = (1/beta) * (sum_exp_max / sum_exp_max) * ...
    # Actually:
    # S_max = max_ds + (1/beta) log(sum e^{beta(d^2 - max_ds)})
    # d S_max / d x_i = (1/beta) * (1/sum_exp_max) * sum (e^{beta(d_k^2 - max_ds)} * beta * d(d_k^2)/dx_i)
    # = (1/sum_exp_max) * sum (w_k * d(d_k^2)/dx_i)
    # where w_k = e^{beta(d_k^2 - max_ds)}
    
    # d(log S_max)/dx = (1/S_max) * (1/sum_exp_max) * sum (w_k * grad_d_k^2)
    
    # Similarly for S_min:
    # S_min = min_ds - (1/beta) log(sum e^{-beta(d^2 - min_ds)})
    # d S_min / d x_i = - (1/beta) * (1/sum_exp_min) * sum (e^{-beta(d_k^2 - min_ds)} * (-beta) * d(d_k^2)/dx_i)
    # = (1/sum_exp_min) * sum (v_k * grad_d_k^2)
    # where v_k = e^{-beta(d_k^2 - min_ds)}
    
    # d(log S_min)/dx = (1/S_min) * (1/sum_exp_min) * sum (v_k * grad_d_k^2)
    
    # Total grad = d(log S_max)/dx - d(log S_min)/dx
    
    # We need gradients of d_k^2 with respect to each coordinate.
    # dist_sq is a flat array of pairs.
    # We need to map back to coordinates.
    
    # Efficient way:
    # grad_d_sq[i, j, :] = 2 * (x_i - x_j)
    
    coords_flat = coords.reshape(-1)
    grad = np.zeros_like(coords)
    
    # Precompute weights
    # w_k corresponds to pair k
    # We need to accumulate contributions to each point's coordinates.
    
    # Let's iterate over pairs for gradient accumulation to keep code simple and robust for N=14
    # 91 pairs is small.
    
    pair_indices = []
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            pair_indices.append((i, j))
            idx += 1
            
    # Calculate d(d_k^2)/dx for all pairs
    # d_k^2 = ||x_i - x_j||^2
    # d/dx_i = 2(x_i - x_j)
    # d/dx_j = 2(x_j - x_i) = -2(x_i - x_j)
    
    # S_max term
    # coeff_max = 1 / (S_max * sum_exp_max)
    coeff_max = 1.0 / (S_max * sum_exp_max)
    
    # S_min term
    # coeff_min = 1 / (S_min * sum_exp_min)
    coeff_min = 1.0 / (S_min * sum_exp_min)
    
    for k, (i, j) in enumerate(pair_indices):
        diff = coords[i] - coords[j]
        grad_pair = 2.0 * diff
        
        # Contribution to x_i and x_j
        w_k = exp_max[k]
        v_k = exp_min[k]
        
        # d(log S_max)/dx_i = coeff_max * w_k * 2*(x_i - x_j)
        # d(log S_max)/dx_j = coeff_max * w_k * 2*(x_j - x_i)
        
        # d(log S_min)/dx_i = coeff_min * v_k * 2*(x_i - x_j)
        # d(log S_min)/dx_j = coeff_min * v_k * 2*(x_j - x_i)
        
        # Total grad for x_i: coeff_max*w_k*grad_pair - coeff_min*v_k*grad_pair
        # Total grad for x_j: coeff_max*w_k*(-grad_pair) - coeff_min*v_k*(-grad_pair)
        
        grad_i = (coeff_max * w_k - coeff_min * v_k) * grad_pair
        grad_j = -grad_i
        
        grad[i] += grad_i
        grad[j] += grad_j
        
    return val, grad

def _fibonacci_sphere(n, d=3):
    """Generate n points on a sphere using Fibonacci spiral."""
    if d != 3:
        return np.random.randn(n, d)
    
    points = np.zeros((n, 3))
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        z = 1 - (i / (n - 1)) * 2  # z goes from 1 to -1
        radius = np.sqrt(1 - z*z)
        theta = golden_angle * i
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points[i] = [x, y, z]
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    best_score = np.inf
    best_coords = None
    
    # Initializations
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(np.random.randn(N, D) * 1.0)
    inits.append(np.random.randn(N, D) * 0.5)
    
    # Add a grid initialization
    side = int(np.ceil(N ** (1 / D)))
    grid = list(itertools.product(range(side), repeat=D))[:N]
    inits.append(np.array(grid, dtype=float))
    
    # Beta annealing schedule
    # Start low to find global structure, increase to sharpen
    betas = [1.0, 5.0, 20.0, 100.0, 500.0]
    
    for init_coords in inits:
        if time.time() > deadline - 10.0:
            break
            
        current_coords = init_coords.copy()
        
        # Multi-stage optimization
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
                
            try:
                # Use L-BFGS-B
                res = minimize(
                    _objective_and_grad,
                    current_coords,
                    args=(beta,),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.success:
                    current_coords = res.x
            except Exception:
                pass
        
        # Calculate final score using the actual max/min ratio (not soft)
        dist_sq = _pairwise_distances_squared(current_coords)
        min_d = np.min(dist_sq)
        max_d = np.max(dist_sq)
        
        if min_d < 1e-12:
            score = np.inf
        else:
            score = (max_d / min_d)
            
        # The task score is (max_dist / min_dist)^2
        # Wait, the task says: "score is (max pairwise distance / min pairwise distance)^2"
        # My objective was log(S_max) - log(S_min) which approximates log(max_d^2) - log(min_d^2) = 2(log(max_d) - log(min_d)).
        # Minimizing this is equivalent to minimizing max_d/min_d.
        # The final score to report is (max_d/min_d)^2.
        
        if score < best_score:
            best_score = score
            best_coords = current_coords.copy()
            
    if best_coords is None:
        # Fallback to grid
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_coords = np.array(grid, dtype=float)
        
    return best_coords
# EVOLVE-BLOCK-END
