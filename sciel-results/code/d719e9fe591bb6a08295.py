"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    deadline = time.time() + 130.0
    rng = np.random.default_rng(42)
    
    # Start with a good initial configuration: hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    best_points = None
    best_score = 0.0
    
    # Generate initial points from hexagonal lattice
    k = int(1.0 / spacing) + 3
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
    
    if len(points) < N:
        spacing *= 0.95
        k = int(1.0 / spacing) + 3
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
    
    if len(points) >= N:
        # Sort by radius and take the innermost N
        idx = np.argsort(radius, kind='stable')[:N]
        x0 = points[idx].copy()
    else:
        # Fallback: random points
        x0 = rng.uniform(-0.95, 0.95, size=(N, 2))
        # Project inside unit circle
        r = np.linalg.norm(x0, axis=1)
        too_far = r > 0.95
        x0[too_far] *= 0.95 / r[too_far][:, None]
    
    def score(x):
        """Compute the packing score: min of pair distances/2 and boundary distances."""
        # Pairwise distances
        diff = x[:, None, :] - x[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Boundary distances
        norms = np.linalg.norm(x, axis=1)
        min_boundary = np.min(1.0 - norms)
        
        return min(min_pair, min_boundary)
    
    # Objective: maximize score, so minimize negative score
    # Use smooth approximation: soft-min via log-sum-exp
    def objective(x, beta=50.0):
        """Smooth objective that approximates -min score."""
        # Pairwise distances
        diff = x[:, None, :] - x[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        half_dists = dists / 2.0
        
        # Boundary distances
        norms = np.linalg.norm(x, axis=1)
        boundary = 1.0 - norms
        
        # Soft-min of all half_dists (off-diagonal) and boundary
        # Use log-sum-exp for soft-min: softmin = -logsumexp(-x) / beta
        all_vals = np.concatenate([half_dists[~np.eye(N, dtype=bool)].flatten(), boundary])
        
        # For numerical stability
        m = np.min(all_vals)
        logsumexp = m + np.log(np.sum(np.exp(-beta * (all_vals - m))))
        softmin = -logsumexp / beta
        
        return -softmin
    
    def gradient(x, beta=50.0):
        """Gradient of the smooth objective."""
        # Pairwise distances
        diff = x[:, None, :] - x[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        half_dists = dists / 2.0
        
        # Boundary distances
        norms = np.linalg.norm(x, axis=1)
        boundary = 1.0 - norms
        
        all_vals = np.concatenate([half_dists[~np.eye(N, dtype=bool)].flatten(), boundary])
        m = np.min(all_vals)
        exp_vals = np.exp(-beta * (all_vals - m))
        logsumexp = m + np.log(np.sum(exp_vals))
        softmin = -logsumexp / beta
        
        # Gradient computation
        weights = exp_vals / np.sum(exp_vals)
        
        grad = np.zeros_like(x)
        
        # Pairwise contribution
        # For each pair (i,j), the contribution to d_ij/2 is (x_i - x_j) / (2 * d_ij)
        # weight for pair (i,j) is weights[idx] where idx maps to the flattened off-diagonal
        pair_weights = weights[:N*(N-1)//2]  # This indexing needs care
        
        # Actually, let's use a simpler approach: numerical gradient or analytic
        # Let's use analytic gradient with proper indexing
        
        # Reset
        grad = np.zeros_like(x)
        
        # Compute weights for all pairs and boundary
        n_pairs = N * (N - 1) // 2
        pair_weights = weights[:n_pairs]
        boundary_weights = weights[n_pairs:]
        
        # Pairwise gradient
        # For pair (i, j) with i < j, distance d_ij
        # d(d_ij/2)/dx_i = (x_i - x_j) / d_ij
        # d(d_ij/2)/dx_j = (x_j - x_i) / d_ij
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                d = dists[i, j]
                if d > 1e-12:
                    w = pair_weights[idx]
                    grad[i] += w * (x[i] - x[j]) / d
                    grad[j] += w * (x[j] - x[i]) / d
                idx += 1
        
        # Boundary gradient
        # d(1 - |x_i|)/dx_i = -x_i / |x_i|
        for i in range(N):
            norm = norms[i]
            if norm > 1e-12:
                grad[i] -= boundary_weights[i] * x[i] / norm
        
        return grad
    
    # Try optimization with L-BFGS-B
    best_x = x0.copy()
    best_s = score(best_x)
    
    # Try different betas and number of iterations
    for beta in [20.0, 50.0, 100.0]:
        if time.time() > deadline - 30.0:
            break
        
        try:
            result = minimize(
                objective, best_x, args=(beta,),
                jac=lambda x, b=beta: -gradient(x, b),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.success or result.fun < 0:
                candidate = result.x
                # Project inside unit circle
                r = np.linalg.norm(candidate, axis=1)
                too_far = r > 0.99
                candidate[too_far] *= 0.99 / r[too_far][:, None]
                
                s = score(candidate)
                if s > best_s:
                    best_s = s
                    best_x = candidate.copy()
        except Exception as e:
            pass
    
    # Final polish: local search with small perturbations
    if time.time() < deadline - 20.0:
        for iteration in range(100):
            if time.time() > deadline - 10.0:
                break
            
            # Try random perturbations
            perturbation = rng.normal(0, 0.001, size=(N, 2))
            candidate = best_x + perturbation
            
            # Project inside unit circle
            r = np.linalg.norm(candidate, axis=1)
            too_far = r > 0.99
            candidate[too_far] *= 0.99 / r[too_far][:, None]
            
            s = score(candidate)
            if s > best_s:
                best_s = s
                best_x = candidate.copy()
    
    # Ensure final result is valid
    r = np.linalg.norm(best_x, axis=1)
    too_far = r > 1.0
    if np.any(too_far):
        best_x[too_far] *= 0.999 / r[too_far][:, None]
    
    return best_x

# EVOLVE-BLOCK-END
