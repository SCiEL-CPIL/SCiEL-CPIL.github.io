"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Minimize the covering radius by maximizing the minimum pairwise angular separation.
    Formulated as: minimize the soft-max of pairwise dot products (proxy for max dot product),
    using projected gradient descent on the sphere.
    """
    
    def fibonacci_sphere(n):
        """Generate n points on the unit sphere using Fibonacci spiral."""
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts
    
    def objective_and_grad(X):
        """
        Compute the soft-max objective and its gradient.
        Objective: (1/beta) * log(sum_{i!=j} exp(beta * dot(x_i, x_j)))
        This approximates the max pairwise dot product.
        
        Gradient for point k:
        grad_k = 2 * sum_{j!=k} w_{kj} * x_j
        where w_{kj} = exp(beta * dot(x_k, x_j)) / sum_{i!=j} exp(beta * dot(x_i, x_j))
        """
        beta = 30.0
        
        # Compute pairwise dot products
        dots = X @ X.T
        np.fill_diagonal(dots, -np.inf)  # Mask out diagonal
        
        # For numerical stability, subtract max
        max_dot = np.max(dots)
        exp_dots = np.exp(beta * (dots - max_dot))
        
        # Sum over all pairs (excluding diagonal)
        sum_exp = np.sum(exp_dots)
        
        # Objective
        obj = max_dot + np.log(sum_exp) / beta
        
        # Gradient: for each point k, grad_k = 2 * sum_{j!=k} (exp_dots[k,j] / sum_exp) * x_j
        # = 2 * (exp_dots @ X) / sum_exp
        grad = 2.0 * (exp_dots @ X) / sum_exp
        
        # Project gradient onto tangent space of sphere
        # For each point k, project grad_k onto the plane perpendicular to X[k]
        proj_grad = grad - np.sum(grad * X, axis=1, keepdims=True) * X
        
        return obj, proj_grad
    
    # Initialize with Fibonacci sphere
    X = fibonacci_sphere(N)
    
    # Project gradient descent with adaptive learning rate
    lr = 0.05
    n_iter = 2000
    patience = 100
    best_obj = np.inf
    no_improve = 0
    
    for it in range(n_iter):
        obj, grad = objective_and_grad(X)
        
        # Update
        X_new = X - lr * grad
        
        # Project back to unit sphere
        norms = np.linalg.norm(X_new, axis=1, keepdims=True)
        X_new = X_new / norms
        
        # Check for improvement
        if obj < best_obj - 1e-10:
            best_obj = obj
            no_improve = 0
        else:
            no_improve += 1
        
        X = X_new
        
        # Adaptive learning rate: reduce if no improvement
        if no_improve >= patience:
            lr *= 0.5
            no_improve = 0
            if lr < 1e-6:
                break
    
    # Final projection to ensure unit norm
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms
    
    return X

# EVOLVE-BLOCK-END
