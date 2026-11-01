"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def _compute_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    n = points.shape[0]
    if n < 2:
        return 1e10
    # Compute all pairwise distances
    dists = pdist(points)
    if len(dists) == 0:
        return 1e10
    min_d = np.min(dists)
    max_d = np.max(dists)
    if min_d < 1e-10:
        return 1e10
    return (max_d / min_d) ** 2

def construct():
    n = N
    d = D
    
    # Precompute indices for efficiency in Gram space
    indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(indices)
    
    def get_distance_vector(G):
        """Return vector of squared distances for all pairs from Gram matrix G."""
        diag = np.diag(G)
        d_sq = np.zeros(num_pairs)
        for k, (i, j) in enumerate(indices):
            d_sq[k] = diag[i] + diag[j] - 2 * G[i, j]
        return d_sq

    def objective_gram(G_flat):
        """Objective function for Gram matrix optimization."""
        G = G_flat.reshape(n, n)
        # Enforce symmetry
        G = (G + G.T) / 2
        
        # Compute eigenvalues of G for PSD and Rank penalties
        eigvals = np.linalg.eigvalsh(G)
        
        # PSD penalty: sum of negative eigenvalues squared
        neg_eigs = eigvals[eigvals < 0]
        psd_penalty = np.sum(neg_eigs ** 2)
        
        # Rank penalty: sum of eigenvalues from index d to end squared
        rank_penalty = np.sum(eigvals[d:] ** 2)
        
        # Compute squared distances
        d_sq = get_distance_vector(G)
        
        # Handle invalid distances (negative due to non-PSD G)
        if np.any(d_sq < -1e-8):
            return 1e10 + psd_penalty + rank_penalty
            
        # Clip small negative distances to 0 for log-sum-exp stability
        d_sq_clipped = np.maximum(d_sq, 1e-10)
        
        # Smooth max and min using log-sum-exp
        # Temperature T controls the sharpness. Lower T is sharper (closer to true max/min).
        # We use a moderate T for stability.
        T = 0.5 
        max_approx = T * np.log(np.sum(np.exp(d_sq_clipped / T)))
        min_approx = -T * np.log(np.sum(np.exp(-d_sq_clipped / T)))
        
        if min_approx < 1e-10:
            return 1e10 + psd_penalty + rank_penalty
            
        ratio = max_approx / min_approx
        
        # Scale invariance penalty: encourage trace(G) = 1
        trace_G = np.trace(G)
        scale_penalty = (trace_G - 1.0) ** 2
        
        # Total objective
        # We tune the weights to balance the ratio minimization with the constraints.
        alpha_rank = 50.0
        beta_psd = 50.0
        gamma_scale = 10.0
        
        return ratio + alpha_rank * rank_penalty + beta_psd * psd_penalty + gamma_scale * scale_penalty

    def extract_points_from_gram(G):
        """Extract n points in d-dimensional space from Gram matrix G."""
        # Symmetrize
        G = (G + G.T) / 2
        eigvals, eigvecs = np.linalg.eigh(G)
        
        # Sort by eigenvalue descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Keep top d eigenvectors
        # Clip negative eigenvalues to 0
        eigvals = np.maximum(eigvals, 0)
        
        # Points are V = eigvecs[:, :d] * sqrt(eigvals[:d])
        # Note: eigvecs are columns. So V[i, j] = eigvecs[i, j] * sqrt(eigvals[j])
        V = eigvecs[:, :d] * np.sqrt(eigvals[:d])
        
        # Center the points (remove mean) to reduce translation degrees of freedom
        V = V - np.mean(V, axis=0)
        
        return V

    def objective_coords(x):
        """Objective function for direct coordinate optimization."""
        V = x.reshape(n, d)
        d_sq = pdist(V) ** 2
        if np.any(d_sq < 1e-12):
            return 1e10
        
        # Use a smaller T for the final refinement to be closer to the true max/min ratio
        T = 0.1
        max_approx = T * np.log(np.sum(np.exp(d_sq / T)))
        min_approx = -T * np.log(np.sum(np.exp(-d_sq / T)))
        
        if min_approx < 1e-12:
            return 1e10
            
        return max_approx / min_approx

    # Multi-start strategy
    np.random.seed(42)
    
    best_points = None
    best_score = 1e10
    
    # Number of random restarts for Gram optimization
    num_restarts = 5
    
    for restart in range(num_restarts):
        # Generate random initial Gram matrix
        # Start with a random rank-d matrix G = V V^T
        V0 = np.random.randn(n, d)
        # Scale it to have trace 1
        trace = np.trace(V0 @ V0.T)
        if trace > 0:
            V0 = V0 / np.sqrt(trace)
        G0 = V0 @ V0.T
        G0_flat = G0.flatten()
        
        try:
            # Optimize Gram matrix
            res_gram = minimize(objective_gram, G0_flat, method='L-BFGS-B', 
                                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res_gram.fun < 1e9: # Check if optimization converged reasonably
                G_opt = res_gram.x.reshape(n, n)
                points_gram = extract_points_from_gram(G_opt)
                
                # Refine in coordinate space
                try:
                    res_coords = minimize(objective_coords, points_gram.flatten(), method='L-BFGS-B', 
                                         options={'maxiter': 1000, 'ftol': 1e-12})
                    points_final = res_coords.x.reshape(n, d)
                except Exception:
                    points_final = points_gram
                
                score = _compute_score(points_final)
                
                if score < best_score:
                    best_score = score
                    best_points = points_final
                    
        except Exception:
            pass

    # If no valid points found, fallback to random
    if best_points is None:
        best_points = np.random.randn(n, d)
        
    return best_points

# EVOLVE-BLOCK-END
