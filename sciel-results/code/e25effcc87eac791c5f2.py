import numpy as np
import time
from scipy.optimize import minimize

N = 130

def construct():
    start_time = time.time()
    deadline = 145.0 # Leave 5s margin
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Grid points for distance calculation
    # Use a denser Fibonacci spiral for the grid
    N_grid = 20000
    i_g = np.arange(N_grid) + 0.5
    z_g = 1.0 - 2.0 * i_g / N_grid
    phi_g = np.pi * (1.0 + np.sqrt(5.0)) * i_g
    r_g = np.sqrt(1.0 - z_g * z_g)
    G = np.column_stack([r_g * np.cos(phi_g), r_g * np.sin(phi_g), z_g])
    
    # Precompute G for efficiency
    # We will compute distances between G (N_grid, 3) and P (N, 3)
    
    beta = 100.0 # Sharpness of soft-max
    
    def get_flattened(P):
        return P.flatten()
        
    def get_reshaped(x):
        P_reshaped = x.reshape((N, 3))
        # Normalize
        norms = np.linalg.norm(P_reshaped, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return P_reshaped / norms

    def objective(x):
        P = get_reshaped(x)
        # Compute squared distances
        # dist_sq[i, j] = || G[i] - P[j] ||^2
        # = ||G[i]||^2 + ||P[j]||^2 - 2 G[i] dot P[j]
        # Since both are unit vectors, ||G||^2 = 1, ||P||^2 = 1
        # dist_sq = 2 - 2 G dot P
        G_dot_P = G @ P.T # (N_grid, N)
        dist_sq = 2.0 - 2.0 * G_dot_P
        
        # Clip for numerical stability
        dist_sq = np.clip(dist_sq, 1e-10, 2.0)
        
        # Soft-max of dist_sq
        # To avoid overflow, subtract max
        max_ds = np.max(dist_sq, axis=1) # This is wrong, we want max over grid points for each config? No.
        # The objective is max over grid points of (min over config points of dist).
        # Let d_i = min_j dist_sq[i, j]
        d_min = np.min(dist_sq, axis=1)
        
        # Soft-max of d_min
        max_d = np.max(d_min)
        if max_d > 0:
            exp_d = np.exp(beta * (d_min - max_d))
            sum_exp = np.sum(exp_d)
            obj_val = max_d + np.log(sum_exp) / beta
        else:
            obj_val = 0.0
            
        return obj_val

    def gradient(x):
        P = get_reshaped(x)
        G_dot_P = G @ P.T
        dist_sq = 2.0 - 2.0 * G_dot_P
        dist_sq = np.clip(dist_sq, 1e-10, 2.0)
        
        d_min = np.min(dist_sq, axis=1)
        min_idx = np.argmin(dist_sq, axis=1)
        
        max_d = np.max(d_min)
        exp_d = np.exp(beta * (d_min - max_d))
        sum_exp = np.sum(exp_d)
        weights = exp_d / sum_exp
        
        # Gradient of soft-max w.r.t d_min[i] is beta * weights[i]
        # But we need gradient w.r.t P.
        # d_min[i] = dist_sq[i, min_idx[i]]
        # dist_sq[i, j] = 2 - 2 G[i] dot P[j]
        # d(dist_sq[i, j]) / d P[j] = -2 G[i]
        
        # So grad_P[j] = sum_{i: min_idx[i] == j} weights[i] * beta * (-2 G[i])
        
        grad_P = np.zeros_like(P)
        
        # Efficient accumulation
        # We need to sum weights[i] * G[i] for all i where min_idx[i] == j
        # Use np.add.at or bincount
        # weights shape (N_grid,)
        # G shape (N_grid, 3)
        # min_idx shape (N_grid,)
        
        # We want sum over i for each j
        # grad_P[j] = -2 * beta * sum_{i: min_idx[i]==j} weights[i] * G[i]
        
        # Let's compute sum_G = sum_{i} weights[i] * G[i] * indicator(min_idx[i]==j)
        # We can use np.add.at
        weighted_G = weights[:, np.newaxis] * G
        np.add.at(grad_P, min_idx, -2.0 * beta * weighted_G)
        
        # Now project gradient to tangent space?
        # The points are constrained to the sphere.
        # The gradient in the ambient space is grad_P.
        # The gradient on the manifold is (I - P P^T) grad_P
        # But since we are using L-BFGS-B which works in Euclidean space, 
        # and we normalize P in the objective, the gradient computed above is 
        # the gradient of the objective w.r.t the *unnormalized* P?
        # No, we normalized P inside get_reshaped.
        # The chain rule is tricky here.
        # It is safer to use a finite difference gradient or ensure the gradient is consistent.
        # Actually, if we treat x as the coordinates on the sphere, the constraint is x^T x = 1.
        # The gradient of the objective f(P) where P is on the sphere is the projection of the ambient gradient.
        # However, scipy.optimize.minimize with L-BFGS-B doesn't handle manifold constraints directly.
        # It will step in Euclidean space.
        # If we provide the ambient gradient, and then normalize P in the next iteration, it might work but converge slowly or incorrectly.
        # Better: Use a parameterization that doesn't require normalization? Or use a penalty method?
        # Or, simply use a derivative-free method like Powell or Nelder-Mead, which is slower but robust to constraints.
        # Given the time limit, L-BFGS-B is preferred.
        # Let's try to compute the gradient correctly for the normalized points.
        # Let P_norm = P / ||P||.
        # The objective depends on P_norm.
        # d obj / d P = (d obj / d P_norm) * (d P_norm / d P)
        # d P_norm / d P is the projection matrix (I - P_norm P_norm^T) / ||P|| ... roughly.
        # Actually, if we assume P is already normalized (which we enforce in get_reshaped),
        # the gradient w.r.t the coordinates on the sphere is the tangential component of the ambient gradient.
        # grad_tangent = grad_P - (grad_P dot P) P
        
        for k in range(N):
            grad_tangent = grad_P[k] - np.dot(grad_P[k], P[k]) * P[k]
            grad_P[k] = grad_tangent
            
        return grad_P.flatten()

    # Optimize
    x0 = get_flattened(P)
    
    # Use L-BFGS-B
    # We need to be careful with the time.
    # Let's run minimize with a maxiter limit and check time.
    
    best_x = x0
    best_val = objective(x0)
    
    try:
        res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
        
        if res.success or res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception as e:
        pass
        
    # Check time
    if time.time() < deadline:
        # Try another round with higher beta or different start?
        # Or just polish with Nelder-Mead
        pass
        
    P_best = get_reshaped(best_x)
    
    return P_best
