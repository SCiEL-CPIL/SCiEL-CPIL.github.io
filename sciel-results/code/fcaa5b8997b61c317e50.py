"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools
from scipy.spatial.distance import pdist

def _pairwise_indices(n):
    """Return list of (i, j) pairs with i < j."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _compute_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    n = points.shape[0]
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
    
    indices = _pairwise_indices(n)
    num_pairs = len(indices)
    
    def get_distances_from_G(G):
        """Compute squared distances from Gram matrix G."""
        # D_ij = G_ii + G_jj - 2 G_ij
        # G is n x n
        diag = np.diag(G)
        # Create matrix of squared distances
        D_sq = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d2 = diag[i] + diag[j] - 2 * G[i, j]
                D_sq[i, j] = d2
                D_sq[j, i] = d2
        # Extract upper triangle for objective
        return D_sq[indices[0][0], indices[0][1]] # Placeholder, need vector
        
    def get_distance_vector(G):
        """Return vector of squared distances for all pairs."""
        diag = np.diag(G)
        d_sq = np.zeros(num_pairs)
        for k, (i, j) in enumerate(indices):
            d_sq[k] = diag[i] + diag[j] - 2 * G[i, j]
        return d_sq

    def objective(G_flat):
        G = G_flat.reshape(n, n)
        # Enforce symmetry
        G = (G + G.T) / 2
        
        # 1. Compute eigenvalues of G for PSD and Rank penalties
        eigvals = np.linalg.eigvalsh(G)
        
        # PSD penalty: sum of negative eigenvalues squared
        neg_eigs = eigvals[eigvals < 0]
        psd_penalty = np.sum(neg_eigs ** 2)
        
        # Rank penalty: sum of eigenvalues from index d to end squared
        # We want rank <= d, so eigenvalues d..n-1 should be 0
        rank_penalty = np.sum(eigvals[d:] ** 2)
        
        # 2. Compute distances
        d_sq = get_distance_vector(G)
        
        # Handle invalid distances (negative due to non-PSD G)
        if np.any(d_sq < -1e-6):
            return 1e10 + psd_penalty + rank_penalty
            
        # Clip small negative distances to 0 for log-sum-exp stability
        d_sq_clipped = np.maximum(d_sq, 1e-12)
        
        # Smooth max and min
        T_max = 0.1
        T_min = 0.1
        max_approx = T_max * np.log(np.sum(np.exp(d_sq_clipped / T_max)))
        min_approx = -T_min * np.log(np.sum(np.exp(-d_sq_clipped / T_min)))
        
        if min_approx < 1e-12:
            return 1e10 + psd_penalty + rank_penalty
            
        ratio = max_approx / min_approx
        
        # Scale invariance penalty: encourage trace(G) = 1
        # If trace is 0, G is 0, which is bad.
        trace_G = np.trace(G)
        scale_penalty = (trace_G - 1.0) ** 2
        
        # Total objective
        # Weights need tuning. 
        # Ratio is the main target. Penalties should be strong enough to keep G valid.
        alpha_rank = 10.0
        beta_psd = 10.0
        gamma_scale = 10.0
        
        return ratio + alpha_rank * rank_penalty + beta_psd * psd_penalty + gamma_scale * scale_penalty

    # Initializations
    np.random.seed(42)
    
    # 1. Random rank-d matrix G = V V^T
    V0 = np.random.randn(n, d)
    G0 = V0 @ V0.T
    G0_flat = G0.flatten()
    
    # 2. Identity-like (scaled)
    G1 = np.eye(n) / n
    G1_flat = G1.flatten()
    
    # 3. Random full matrix (will be penalized heavily)
    G2 = np.random.randn(n, n)
    G2 = (G2 + G2.T) / 2
    G2_flat = G2.flatten()
    
    best_x = None
    best_val = 1e10
    
    for x0 in [G0_flat, G1_flat, G2_flat]:
        try:
            # Use L-BFGS-B with numerical gradient
            # The dimension is n*n = 196. This is manageable.
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception as e:
            pass

    if best_x is None:
        # Fallback to random points
        return np.random.randn(n, d)
    
    G_opt = best_x.reshape(n, n)
    G_opt = (G_opt + G_opt.T) / 2
    
    # Extract points from Gram matrix
    # We need to find V such that V V^T = G_opt.
    # Since G_opt might not be exactly rank 3 due to optimization noise,
    # we perform eigen-decomposition and keep top 3 eigenvectors.
    eigvals, eigvecs = np.linalg.eigh(G_opt)
    
    # Sort by eigenvalue descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Keep top d eigenvectors
    V = eigvecs[:, :d] * np.sqrt(np.maximum(eigvals[:d], 0))
    
    points = V
    
    # Post-processing: Refine coordinates directly
    # The Gram matrix approach might leave points in a slightly suboptimal configuration 
    # due to the penalty terms. We can use the points from V as an initial guess 
    # for a direct coordinate optimization to fine-tune.
    
    def objective_coords(x):
        V = x.reshape(n, d)
        d_sq = pdist(V) ** 2
        if np.any(d_sq < 1e-12):
            return 1e10
        T = 0.1
        max_approx = T * np.log(np.sum(np.exp(d_sq / T)))
        min_approx = -T * np.log(np.sum(np.exp(-d_sq / T)))
        if min_approx < 1e-12:
            return 1e10
        return max_approx / min_approx

    try:
        res_final = minimize(objective_coords, points.flatten(), method='L-BFGS-B', 
                             options={'maxiter': 500, 'ftol': 1e-12})
        if res_final.fun < _compute_score(points):
            points = res_final.x.reshape(n, d)
    except Exception:
        pass

    return points

# EVOLVE-BLOCK-END
