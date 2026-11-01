import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100
START_TIME = time.time()
TIME_LIMIT = 110.0  # Leave 10s buffer

def _energy_and_grad(x_flat):
    """
    Calculate the logarithmic energy and its gradient.
    x_flat is a vector of shape (3*N,).
    """
    x = x_flat.reshape(N, 3)
    
    # Calculate pairwise distances
    # dist[i, j] = ||x_i - x_j||
    # We only need i < j for energy, but for gradient we need all j != i
    # Using broadcasting for efficiency
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    
    # Avoid division by zero (though points should be distinct)
    # If distance is very small, energy is huge, gradient is huge.
    # We assume valid inputs.
    
    # Energy: sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
    # We can compute sum over all i!=j and divide by 2
    # log(d_ii) is log(0) which is -inf, so we must mask the diagonal
    mask = ~np.eye(N, dtype=bool)
    d_masked = dist[mask]
    
    # If any distance is 0, return a very high energy
    if np.any(d_masked < 1e-12):
        return 1e10, np.zeros_like(x_flat)
        
    energy = -np.sum(np.log(d_masked)) / 2.0
    
    # Gradient
    # grad_i = - sum_{j != i} (x_i - x_j) / d_ij^2
    # Note: The term for j=i is 0/0, so we can include it if we handle 0/0=0
    # Let's compute the full matrix term and sum over j
    
    # diff is (N, N, 3) where diff[i,j] = x_i - x_j
    # dist_sq is (N, N)
    dist_sq = dist ** 2
    
    # We want to compute sum_j (x_i - x_j) / dist_sq[i,j]
    # This is equivalent to summing the rows of a matrix M where M[i,j] = (x_i - x_j) / dist_sq[i,j]
    
    # Handle diagonal: dist_sq[i,i] = 0. We want the contribution to be 0.
    # So we set dist_sq[i,i] to 1 to avoid div by zero, and the numerator is 0 anyway.
    dist_sq_safe = np.where(dist_sq > 1e-12, dist_sq, 1.0)
    
    # factor[i,j] = 1 / dist_sq_safe[i,j]
    # For diagonal, factor is 1, but diff is 0, so product is 0.
    factors = 1.0 / dist_sq_safe
    
    # weighted_diff[i,j,k] = diff[i,j,k] * factors[i,j]
    weighted_diff = diff * factors[:, :, np.newaxis]
    
    # grad[i] = - sum_j weighted_diff[i,j]
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad.flatten()

def _constraints(x_flat):
    """
    Constraints: ||x_i||^2 = 1 for all i.
    Returns a list of constraint dicts.
    """
    x = x_flat.reshape(N, 3)
    norms_sq = np.sum(x ** 2, axis=1)
    # We need norms_sq == 1
    # scipy minimize constraints: fun(x) >= 0 for inequality, == 0 for equality
    # Let's use equality constraints
    constraints = []
    for i in range(N):
        constraints.append({
            'type': 'eq',
            'fun': lambda x, i=i: np.sum(x[3*i:3*i+3]**2) - 1.0,
            'jac': lambda x, i=i: np.array([0.0 if k < 3*i or k >= 3*i+3 else 2.0*x[k] for k in range(3*N)])
        })
    return constraints

def _optimize_single(x0):
    """
    Run L-BFGS-B optimization starting from x0.
    """
    # Define constraints more efficiently for scipy
    # Actually, defining 100 separate constraint functions is slow.
    # It's better to define a single vector-valued equality constraint.
    
    def eq_fun(x):
        x = x.reshape(N, 3)
        return np.sum(x**2, axis=1) - 1.0
        
    def eq_jac(x):
        x = x.reshape(N, 3)
        # Jacobian is 300x100
        # Row i corresponds to constraint i: sum(x_{3i}^2 + x_{3i+1}^2 + x_{3i+2}^2) - 1
        # Derivative w.r.t x_k is 2*x_k if k is in i's block, else 0.
        # We can construct this using a sparse matrix or just a dense array if N is small.
        # N=100, so 300x100 = 30000 elements. Dense is fine.
        J = np.zeros((3*N, N))
        for i in range(N):
            J[3*i, i] = 2.0 * x[i, 0]
            J[3*i+1, i] = 2.0 * x[i, 1]
            J[3*i+2, i] = 2.0 * x[i, 2]
        return J

    constraints = [{'type': 'eq', 'fun': eq_fun, 'jac': eq_jac}]

    try:
        res = minimize(
            _energy_and_grad, 
            x0, 
            jac=True, 
            method='L-BFGS-B', 
            constraints=constraints, 
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
        )
        return res.x
    except Exception:
        return x0

def _fibonacci_sphere(n, offset=0.5, perturb=0.0):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if perturb > 0:
        x += perturb * np.random.randn(n, 3)
        # Renormalize
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / norms
    return x

def construct():
    global START_TIME
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT:
        # Fallback to seed if out of time (shouldn't happen at start)
        return _fibonacci_sphere(N).flatten()

    best_x = None
    best_energy = float('inf')
    
    # Generate a few diverse starting points
    start_points = []
    start_points.append(_fibonacci_sphere(N))
    
    # Add some perturbed versions
    np.random.seed(42) # Reproducibility for perturbations
    for i in range(5):
        start_points.append(_fibonacci_sphere(N, perturb=0.05 * (i+1)))
        
    # Shuffle to ensure we try different ones if time runs out
    np.random.shuffle(start_points)
    
    for idx, x0 in enumerate(start_points):
        current_time = time.time()
        elapsed = current_time - START_TIME
        if elapsed > TIME_LIMIT - 10: # Stop if less than 10s left
            break
            
        # Flatten for optimizer
        x0_flat = x0.flatten()
        
        # Optimization
        x_opt_flat = _optimize_single(x0_flat)
        
        # Normalize again to be safe
        x_opt = x_opt_flat.reshape(N, 3)
        norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
        x_opt = x_opt / norms
        
        # Evaluate energy
        energy, _ = _energy_and_grad(x_opt.flatten())
        
        if energy < best_energy:
            best_energy = energy
            best_x = x_opt
            
        # Check time
        elapsed = time.time() - START_TIME
        if elapsed > TIME_LIMIT - 20:
             break

    if best_x is None:
        # Fallback
        best_x = _fibonacci_sphere(N)
        
    return best_x.flatten()
