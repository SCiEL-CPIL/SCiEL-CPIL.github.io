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

def _smooth_min_max_ratio(d_sq, T=0.05):
    """Compute smooth approximation of max/min ratio using log-sum-exp."""
    d_sq = np.maximum(d_sq, 1e-12)
    max_approx = T * np.log(np.sum(np.exp(d_sq / T)))
    min_approx = -T * np.log(np.sum(np.exp(-d_sq / T)))
    if min_approx < 1e-12:
        return 1e10
    return max_approx / min_approx

def construct():
    n = N
    d = D
    
    indices = _pairwise_indices(n)
    num_pairs = len(indices)
    
    def get_distance_vector(G):
        """Return vector of squared distances for all pairs."""
        diag = np.diag(G)
        d_sq = np.zeros(num_pairs)
        for k, (i, j) in enumerate(indices):
            d_sq[k] = diag[i] + diag[j] - 2 * G[i, j]
        return d_sq

    def objective_gram(G_flat):
        """Objective for Gram matrix optimization with PSD and rank constraints."""
        G = G_flat.reshape(n, n)
        G = (G + G.T) / 2
        
        # Eigenvalue decomposition for PSD and rank penalties
        eigvals = np.linalg.eigvalsh(G)
        
        # PSD penalty: negative eigenvalues should be zero
        neg_eigs = eigvals[eigvals < 0]
        psd_penalty = np.sum(neg_eigs ** 2)
        
        # Rank penalty: eigenvalues beyond d should be zero
        # Use a stronger penalty for rank violation
        rank_penalty = np.sum(eigvals[d:] ** 2)
        
        # Scale penalty: fix trace to prevent trivial solutions
        trace_G = np.trace(G)
        scale_penalty = (trace_G - 1.0) ** 2
        
        # Compute squared distances
        d_sq = get_distance_vector(G)
        
        # If distances are invalid (negative), return large penalty
        if np.any(d_sq < -1e-6):
            return 1e10 + 100.0 * psd_penalty + 100.0 * rank_penalty + 10.0 * scale_penalty
        
        d_sq_clipped = np.maximum(d_sq, 1e-12)
        
        # Smooth max/min ratio
        T = 0.05
        max_approx = T * np.log(np.sum(np.exp(d_sq_clipped / T)))
        min_approx = -T * np.log(np.sum(np.exp(-d_sq_clipped / T)))
        
        if min_approx < 1e-12:
            return 1e10 + 100.0 * psd_penalty + 100.0 * rank_penalty + 10.0 * scale_penalty
            
        ratio = max_approx / min_approx
        
        # Total objective with weighted penalties
        alpha_rank = 50.0
        beta_psd = 50.0
        gamma_scale = 5.0
        
        return ratio + alpha_rank * rank_penalty + beta_psd * psd_penalty + gamma_scale * scale_penalty

    def objective_coords(x):
        """Objective for direct coordinate optimization."""
        V = x.reshape(n, d)
        d_sq = pdist(V) ** 2
        if np.any(d_sq < 1e-12):
            return 1e10
        return _smooth_min_max_ratio(d_sq, T=0.03)

    def objective_coords_with_spread(x):
        """Objective with additional spread constraint to avoid degenerate solutions."""
        V = x.reshape(n, d)
        d_sq = pdist(V) ** 2
        if np.any(d_sq < 1e-12):
            return 1e10
        ratio = _smooth_min_max_ratio(d_sq, T=0.03)
        
        # Add a small penalty if points are too clustered
        mean_dist = np.mean(d_sq)
        if mean_dist < 0.1:
            ratio += (0.1 - mean_dist) ** 2 * 10
        
        return ratio

    # Multiple random restarts for Gram matrix optimization
    np.random.seed(42)
    
    best_points = None
    best_score = 1e10
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Random rank-d matrices
    for seed in range(10):
        rng = np.random.RandomState(seed)
        V = rng.randn(n, d)
        # Normalize to unit trace
        trace = np.trace(V @ V.T)
        if trace > 1e-10:
            V = V * np.sqrt(1.0 / trace)
        G = V @ V.T
        initializations.append(G.flatten())
    
    # 2. Structured initializations (e.g., vertices of regular polytopes projected)
    # Icosahedron vertices projected to 3D
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
        [1, phi, 0], [-1, -phi, 0]
    ])
    # Normalize
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    # Take first 14 points (we have exactly 12, add 2 more)
    if len(ico_verts) >= n:
        V_ico = ico_verts[:n]
        trace = np.trace(V_ico @ V_ico.T)
        if trace > 1e-10:
            V_ico = V_ico * np.sqrt(1.0 / trace)
        G_ico = V_ico @ V_ico.T
        initializations.append(G_ico.flatten())
    
    # 3. Random full-rank matrices (will be penalized but may find good local minima)
    for seed in range(5):
        rng = np.random.RandomState(100 + seed)
        G = rng.randn(n, n)
        G = (G + G.T) / 2
        # Make it PSD by G = G @ G.T
        G = G @ G.T
        trace = np.trace(G)
        if trace > 1e-10:
            G = G / trace
        initializations.append(G.flatten())
    
    # Optimize from each initialization
    for x0 in initializations:
        try:
            res = minimize(objective_gram, x0, method='L-BFGS-B', 
                           options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            G_opt = res.x.reshape(n, n)
            G_opt = (G_opt + G_opt.T) / 2
            
            # Extract points via eigen-decomposition
            eigvals, eigvecs = np.linalg.eigh(G_opt)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            
            V = eigvecs[:, :d] * np.sqrt(np.maximum(eigvals[:d], 0))
            points = V
            
            # Compute actual score
            score = _compute_score(points)
            
            if score < best_score:
                best_score = score
                best_points = points.copy()
                
        except Exception as e:
            continue
    
    # If no good points found, use random fallback
    if best_points is None:
        best_points = np.random.randn(n, d)
        best_score = _compute_score(best_points)
    
    # Final refinement: direct coordinate optimization from best Gram-derived points
    # Try multiple coordinate optimization strategies
    
    candidates = [best_points]
    
    # Also try from other initializations
    for seed in range(20):
        rng = np.random.RandomState(seed)
        V = rng.randn(n, d)
        # Normalize
        trace = np.trace(V @ V.T)
        if trace > 1e-10:
            V = V * np.sqrt(1.0 / trace)
        candidates.append(V)
    
    # Optimize each candidate in coordinate space
    for points0 in candidates:
        try:
            # Try with smooth objective
            res1 = minimize(objective_coords, points0.flatten(), method='L-BFGS-B', 
                           options={'maxiter': 2000, 'ftol': 1e-14})
            
            points1 = res1.x.reshape(n, d)
            score1 = _compute_score(points1)
            
            # Try with spread constraint
            res2 = minimize(objective_coords_with_spread, points0.flatten(), method='L-BFGS-B', 
                           options={'maxiter': 2000, 'ftol': 1e-14})
            
            points2 = res2.x.reshape(n, d)
            score2 = _compute_score(points2)
            
            # Pick best from this candidate
            if score1 < best_score:
                best_score = score1
                best_points = points1.copy()
            if score2 < best_score:
                best_score = score2
                best_points = points2.copy()
                
        except Exception as e:
            continue
    
    # One more round of refinement from the best point
    try:
        res_final = minimize(objective_coords, best_points.flatten(), method='L-BFGS-B', 
                             options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12})
        points_final = res_final.x.reshape(n, d)
        score_final = _compute_score(points_final)
        
        if score_final < best_score:
            best_score = score_final
            best_points = points_final.copy()
    except Exception:
        pass
    
    return best_points

# EVOLVE-BLOCK-END
