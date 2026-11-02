import numpy as np
from scipy.optimize import minimize
import time

N = 300

def _hex_lattice_init(n_points, seed=42):
    """Generate a hexagonal lattice of points inside the unit circle."""
    rng = np.random.RandomState(seed)
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    k = int(1.0 / s) + 2
    
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
                
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    points = points[indices[:n_points]]
    points += rng.normal(0, s * 0.01, size=points.shape)
    return points

def _compute_true_score(points):
    """Compute the true packing radius."""
    D = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    np.fill_diagonal(L, np.inf)
    min_pair = np.min(L) / 2.0
    L_wall = np.linalg.norm(points, axis=1)
    min_wall = 1.0 - np.max(L_wall)
    return min(min_pair, min_wall)

def _soft_min_obj_and_grad(x, beta):
    """
    Compute the negative soft-min value and its gradient.
    We want to maximize the soft-min of distances, so we minimize its negative.
    """
    C = x.reshape(N, 2)
    
    # 1. Pairwise distances
    D = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    
    # Avoid division by zero
    L_safe = np.where(L > 1e-12, L, 1.0)
    
    x_pairs = L_safe / 2.0
    
    # Wall distances
    L_wall = np.linalg.norm(C, axis=1)
    x_walls = 1.0 - L_wall
    
    # Get upper triangle indices for pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    x_pairs_flat = x_pairs[i_idx, j_idx]
    
    # Combine all x values
    x_all = np.concatenate([x_pairs_flat, x_walls])
    
    # Compute weights using log-sum-exp trick for stability
    min_x = np.min(x_all)
    shifted_x = x_all - min_x
    exp_terms = np.exp(-beta * shifted_x)
    Z = np.sum(exp_terms)
    P = exp_terms / Z
    
    # Value: S = min_x - (1/beta) * log(Z)
    val = min_x - (1.0 / beta) * np.log(Z)
    
    # Gradient of S
    grad = np.zeros_like(C)
    
    # Pair contributions
    P_pairs = P[:len(i_idx)]
    
    D_flat = D[i_idx, j_idx]
    L_flat = L[i_idx, j_idx]
    L_flat_safe = np.where(L_flat > 1e-12, L_flat, 1.0)
    
    U = D_flat / L_flat_safe[:, np.newaxis]
    
    W = P_pairs * 0.5
    
    np.add.at(grad, i_idx, W[:, np.newaxis] * U)
    np.add.at(grad, j_idx, -W[:, np.newaxis] * U)
    
    # Wall contributions
    P_walls = P[len(i_idx):]
    
    L_wall_safe = np.where(L_wall > 1e-12, L_wall, 1.0)
    U_walls = -C / L_wall_safe[:, np.newaxis]
    
    grad += P_walls[:, np.newaxis] * U_walls
    
    return -val, -grad

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for 150s limit
    
    # Use hexagonal lattice initialization
    points = _hex_lattice_init(N, seed=42)
    best_score = _compute_true_score(points)
    best_points = points.copy()
    current_points = points.copy()
    
    # Beta schedule: start moderate, increase gradually
    beta_values = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]
    
    for beta in beta_values:
        if time.time() - start_time > time_limit:
            break
        
        def obj_and_grad(x):
            return _soft_min_obj_and_grad(x, beta)
        
        try:
            # Use more iterations for lower beta, fewer for higher
            if beta < 200:
                maxiter = 500
                gtol = 1e-8
            elif beta < 2000:
                maxiter = 300
                gtol = 1e-9
            else:
                maxiter = 200
                gtol = 1e-10
            
            res = minimize(obj_and_grad, current_points.flatten(), method='L-BFGS-B', 
                           jac=True, options={'maxiter': maxiter, 'gtol': gtol, 'ftol': 1e-15})
            
            new_points = res.x.reshape(N, 2)
            
            if np.all(np.isfinite(new_points)):
                new_score = _compute_true_score(new_points)
                
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
                
                current_points = new_points
        
        except Exception:
            pass
    
    # Final high-precision polish if time remains
    if time.time() - start_time < time_limit - 2.0:
        beta_final = 20000.0
        def obj_and_grad_final(x):
            return _soft_min_obj_and_grad(x, beta_final)
        
        try:
            res = minimize(obj_and_grad_final, current_points.flatten(), method='L-BFGS-B', 
                           jac=True, options={'maxiter': 500, 'gtol': 1e-11, 'ftol': 1e-16})
            
            new_points = res.x.reshape(N, 2)
            if np.all(np.isfinite(new_points)):
                new_score = _compute_true_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
        except Exception:
            pass
    
    return best_points
