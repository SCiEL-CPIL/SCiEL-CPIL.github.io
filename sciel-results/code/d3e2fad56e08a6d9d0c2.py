"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def construct():
    """
    Minimize the covering radius by maximizing the minimum pairwise angular separation.
    Uses simulated annealing with soft-max objective and multiple restarts.
    """
    
    def fibonacci_sphere(n):
        """Generate n points on the unit sphere using Fibonacci spiral."""
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    def random_sphere(n, rng):
        """Generate n random points on the unit sphere."""
        pts = rng.normal(size=(n, 3))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        return pts / norms

    def objective_and_grad(X, beta):
        """
        Compute the soft-max objective and its gradient.
        Objective: (1/beta) * log(sum_{i!=j} exp(beta * dot(x_i, x_j)))
        This approximates the max pairwise dot product.
        """
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

    def compute_covering_radius(X):
        """Compute the covering radius in degrees."""
        try:
            hull = ConvexHull(X)
            max_radius = 0.0
            for simplex in hull.simplices:
                a, b, c = X[simplex[0]], X[simplex[1]], X[simplex[2]]
                M = np.array([
                    [1.0, np.dot(a,b), np.dot(a,c)],
                    [np.dot(a,b), 1.0, np.dot(b,c)],
                    [np.dot(a,c), np.dot(b,c), 1.0]
                ])
                v = np.array([1.0, 1.0, 1.0])
                try:
                    coeffs = np.linalg.solve(M, v)
                    p = coeffs[0]*a + coeffs[1]*b + coeffs[2]*c
                    norm_p = np.linalg.norm(p)
                    if norm_p < 1e-9:
                        continue
                    p = p / norm_p
                    # The distance from p to the vertices is the circumradius
                    dist = np.arccos(np.clip(np.dot(p, a), -1.0, 1.0))
                    if dist > max_radius:
                        max_radius = dist
                except np.linalg.LinAlgError:
                    pass
            return np.degrees(max_radius)
        except Exception:
            return np.inf

    def optimize(X_init, beta_start, beta_end, n_iter):
        """Run gradient descent with annealing beta."""
        X = X_init.copy()
        
        for it in range(n_iter):
            # Anneal beta
            progress = it / n_iter
            beta = beta_start + (beta_end - beta_start) * progress
            
            obj, grad = objective_and_grad(X, beta)
            
            # Adaptive learning rate
            grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
            grad_norm[grad_norm == 0] = 1.0
            grad_normalized = grad / grad_norm
            
            # Step size decreases with iterations
            lr = 0.1 * (1.0 - progress) + 0.01
            
            X_new = X - lr * grad_normalized
            
            # Project back to unit sphere
            norms = np.linalg.norm(X_new, axis=1, keepdims=True)
            X_new = X_new / norms
            
            X = X_new
            
        return X

    rng = np.random.default_rng(42)
    
    best_X = None
    best_score = np.inf
    
    # Multiple restarts
    for restart in range(10):
        if restart == 0:
            X_init = fibonacci_sphere(N)
        else:
            X_init = random_sphere(N, rng)
        
        X_opt = optimize(X_init, 5.0, 50.0, 2000)
        
        # Evaluate actual covering radius
        score = compute_covering_radius(X_opt)
        
        if score < best_score:
            best_score = score
            best_X = X_opt
    
    # Final projection
    norms = np.linalg.norm(best_X, axis=1, keepdims=True)
    best_X = best_X / norms
    
    return best_X

# EVOLVE-BLOCK-END
