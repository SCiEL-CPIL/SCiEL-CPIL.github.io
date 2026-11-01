import numpy as np
from scipy.optimize import minimize
import time

N = 300

def _hex_lattice_init(n_points):
    """Generate a hexagonal lattice of points inside the unit circle."""
    # Estimate spacing
    # Area of unit circle is pi.
    # Area per point in hex lattice is (sqrt(3)/2) * s^2.
    # n * (sqrt(3)/2) s^2 ~ pi => s ~ sqrt(2 pi / (n sqrt(3)))
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    
    # Generate grid
    # Extend grid to cover unit circle
    k = int(1.0 / s) + 2
    
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    # Sort by distance to center, take closest N
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    points = points[indices[:n_points]]
    
    # Add small random noise to break symmetry
    points += np.random.normal(0, s * 0.01, size=points.shape)
    
    return points

def _compute_true_score(points):
    """Compute the true packing radius."""
    # Pairwise distances
    D = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(L, np.inf)
    min_pair = np.min(L) / 2.0
    
    # Wall distances
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
    
    # Compute weights using log-sum-exp trick for stability if needed, 
    # but simple exp is fine for moderate beta.
    # To prevent underflow for large beta, we can shift by min.
    min_x = np.min(x_all)
    
    # If min_x is very large, exp(-beta * (x - min_x)) is exp(0) for min, 
    # and exp(negative) for others.
    shifted_x = x_all - min_x
    exp_terms = np.exp(-beta * shifted_x)
    Z = np.sum(exp_terms)
    P = exp_terms / Z
    
    # Value: S = min_x - (1/beta) * log(Z)
    # We return -S
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
    
    # We want to minimize -S, so gradient is -grad_S
    return -val, -grad

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin
    
    # Initial points
    points = _hex_lattice_init(N)
    best_points = points.copy()
    best_score = _compute_true_score(points)
    
    beta = 10.0
    max_beta = 2000.0
    beta_factor = 1.25
    
    current_points = points.copy()
    
    while time.time() - start_time < time_limit:
        # Define wrapper for scipy
        def obj_and_grad(x):
            return _soft_min_obj_and_grad(x, beta)
        
        try:
            res = minimize(obj_and_grad, current_points.flatten(), method='L-BFGS-B', jac=True, 
                           options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-12})
            
            if res.success or res.status in [1, 2, 4]:
                new_points = res.x.reshape(N, 2)
                new_score = _compute_true_score(new_points)
                
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
                
                # Update current points for next iteration
                # Even if not best, continue from here if valid
                if np.all(np.isfinite(new_points)):
                    current_points = new_points
            
        except Exception:
            pass
            
        # Increase beta
        beta *= beta_factor
        if beta > max_beta:
            # One final high-precision run
            beta = max_beta * 5
            try:
                res = minimize(obj_and_grad, current_points.flatten(), method='L-BFGS-B', jac=True, 
                               options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-14})
                if res.success or res.status in [1, 2, 4]:
                    new_points = res.x.reshape(N, 2)
                    new_score = _compute_true_score(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points.copy()
            except Exception:
                pass
            break

    return best_points
