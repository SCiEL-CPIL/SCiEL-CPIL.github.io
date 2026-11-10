# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    
    # 1. Initialize Gram matrix G
    # Use Icosahedron (12 points) + 2 random points as a good start
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Add 2 more points. To make it symmetric, maybe place them at poles?
    # Icosahedron has poles? No, vertices are (0,1,phi) etc.
    # Let's just add 2 random points and let the optimizer fix it.
    rng = np.random.RandomState(42)
    extra = rng.randn(2, d)
    extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
    
    X_init = np.vstack([ico, extra])
    # Ensure unit norm
    X_init = X_init / np.linalg.norm(X_init, axis=1, keepdims=True)
    
    G_init = X_init @ X_init.T
    # Enforce unit diagonal
    np.fill_diagonal(G_init, 1.0)
    
    # Extract off-diagonal elements for optimization
    # We need to map indices (i, j) with i < j to a flat vector
    idx_i, idx_j = np.triu_indices(n, k=1)
    x0 = G_init[idx_i, idx_j].copy()
    
    # Objective function
    def objective_and_grad(x):
        G = np.zeros((n, n))
        np.fill_diagonal(G, 1.0)
        G[idx_i, idx_j] = x
        G[idx_j, idx_i] = x
        
        # Smooth max of off-diagonal elements
        beta = 20.0
        vals = x
        max_val = np.max(vals)
        # Shift for numerical stability
        shifted = vals - max_val
        exp_vals = np.exp(beta * shifted)
        smooth_max = max_val + np.log(np.sum(exp_vals)) / beta
        
        # Gradient of smooth max
        # d/dx_k [ max_val + (1/beta) log sum exp(beta x_i) ]
        # Note: max_val depends on x, but its gradient is 1 for the max element.
        # The log-sum-exp gradient is the softmax distribution.
        # grad = softmax(beta * x)
        # Actually, derivative of log(sum(exp(beta x))) is beta * exp(beta x) / sum(exp(beta x)).
        # So grad_smooth_max = exp_vals / np.sum(exp_vals)
        grad_smooth = exp_vals / np.sum(exp_vals)
        
        # PSD Penalty
        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(G)
        # Penalty for negative eigenvalues
        neg_eigvals = np.clip(-eigvals, 0, None)
        penalty = np.sum(neg_eigvals)
        
        # Gradient of penalty
        # d/dG sum(max(0, -lambda_i)) = sum_{i: lambda_i < 0} -v_i v_i^T
        # So grad_penalty_G = - eigvecs[:, neg_mask] @ eigvecs[:, neg_mask].T
        neg_mask = eigvals < 0
        if np.any(neg_mask):
            V_neg = eigvecs[:, neg_mask]
            grad_penalty_G = - (V_neg @ V_neg.T)
        else:
            grad_penalty_G = np.zeros_like(G)
        
        # Map gradient back to x
        grad_x = grad_penalty_G[idx_i, idx_j]
        
        # Combine gradients
        # We are minimizing smooth_max + penalty
        # But we want to maximize the ratio, which is approximated by minimizing smooth_max?
        # Yes, minimizing max G_ij maximizes min distance.
        # Wait, the ratio is (1-G_max)/(1-G_min).
        # Minimizing G_max increases numerator.
        # What about G_min?
        # If we just minimize G_max, we might push G_min down (increasing max distance), which decreases the ratio?
        # No, if points are on unit sphere, G_min is the max dot product? No, G_min is min dot product.
        # Min dot product corresponds to max distance.
        # We want to MAXIMIZE min distance (minimize G_max) AND MINIMIZE max distance (maximize G_min).
        # So we want to compress the range of G_ij.
        # Objective: Minimize (G_max - G_min)?
        # Or Minimize G_max - lambda * G_min?
        
        # Let's try to minimize the variance of G_ij?
        # Or simply minimize G_max. The Tammes problem solution usually has a good ratio.
        # For N=14, the Tammes solution is likely the best ratio solution.
        
        total_grad = grad_smooth + grad_x
        total_val = smooth_max + penalty
        
        return total_val, total_grad

    # Run optimization
    # Use L-BFGS-B or BFGS
    # We need to ensure G stays PSD. The penalty helps, but might not be enough.
    # We can also project G after each step?
    # Minimize with jac=True
    
    # To improve robustness, we can use a multi-start or higher beta annealing.
    # Let's stick to one run with a reasonable beta.
    
    res = minimize(
        objective_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    x_opt = res.x
    
    # Reconstruct G
    G_opt = np.zeros((n, n))
    np.fill_diagonal(G_opt, 1.0)
    G_opt[idx_i, idx_j] = x_opt
    G_opt[idx_j, idx_i] = x_opt
    
    # Final PSD projection
    eigvals, eigvecs = np.linalg.eigh(G_opt)
    eigvals = np.clip(eigvals, 0, None)
    G_opt = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Extract coordinates (top 3 eigenvectors)
    # We want the 3 dimensions with largest variance
    # Sort eigenvalues descending
    order = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[order]
    eigvecs_sorted = eigvecs[:, order]
    
    # Take top 3
    X_final = eigvecs_sorted[:, :3] * np.sqrt(eigvals_sorted[:3])
    
    # Normalize rows to unit length (to be safe)
    norms = np.linalg.norm(X_final, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_final = X_final / norms
    
    # Check if this is a valid improvement
    # Calculate score
    dists = pdist(X_final)
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max > 1e-15:
        score = (d_min / d_max) ** 2
    else:
        score = 0.0
    
    # If the score is bad, fall back to the initial icosahedron-based points
    # or a random search
    if score < 0.5: # Threshold
        # Fall back to coordinate optimization or just return the initial icosahedron+2
        # Let's try a quick coordinate optimization on the initial points
        def neg_ratio(flat_points):
            pts = flat_points.reshape(n, d)
            dists = pdist(pts)
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max < 1e-15:
                return 0.0
            return -(d_min / d_max)
            
        res_coord = minimize(
            neg_ratio, 
            X_init.flatten(), 
            method='Nelder-Mead',
            options={'maxiter': 5000}
        )
        X_final = res_coord.x.reshape(n, d)
        
    return X_final

# EVOLVE-BLOCK-END
