"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def _pairwise_indices(n):
    """Return list of (i, j) pairs with i < j."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _compute_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    n = points.shape[0]
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
    
    def get_distance_vector(G):
        """Return vector of squared distances for all pairs from Gram matrix."""
        diag = np.diag(G)
        d_sq = np.zeros(num_pairs)
        for k, (i, j) in enumerate(indices):
            d_sq[k] = diag[i] + diag[j] - 2 * G[i, j]
        return d_sq

    def make_gram_objective(T, w_psd, w_rank, w_scale):
        def objective_gram(G_flat):
            G = G_flat.reshape(n, n)
            G = (G + G.T) / 2
            
            # Eigenvalue decomposition
            eigvals = np.linalg.eigvalsh(G)
            
            # PSD penalty
            neg_eigs = eigvals[eigvals < 0]
            psd_penalty = np.sum(neg_eigs ** 2)
            
            # Rank penalty
            rank_penalty = np.sum(eigvals[d:] ** 2)
            
            # Distances
            d_sq = get_distance_vector(G)
            
            if np.any(d_sq < -1e-8):
                return 1e10 + psd_penalty + rank_penalty
            
            d_sq_pos = np.maximum(d_sq, 1e-10)
            
            # Soft max/min with temperature T
            if T < 1e-6:
                max_d_sq = np.max(d_sq_pos)
                min_d_sq = np.min(d_sq_pos)
            else:
                # Stable log-sum-exp
                max_d_sq_val = np.max(d_sq_pos)
                min_d_sq_val = np.min(d_sq_pos)
                
                # Soft Max
                max_d_sq = max_d_sq_val + T * np.log(np.sum(np.exp((d_sq_pos - max_d_sq_val) / T)))
                
                # Soft Min
                min_d_sq = min_d_sq_val - T * np.log(np.sum(np.exp((min_d_sq_val - d_sq_pos) / T)))
            
            if min_d_sq < 1e-10:
                return 1e10 + psd_penalty + rank_penalty
            
            ratio = max_d_sq / min_d_sq
            
            # Scale normalization: fix trace(G) = n
            trace_G = np.trace(G)
            scale_penalty = (trace_G - float(n)) ** 2
            
            return ratio + w_psd * psd_penalty + w_rank * rank_penalty + w_scale * scale_penalty
        
        return objective_gram

    def extract_points_from_gram(G):
        G = (G + G.T) / 2
        eigvals, eigvecs = np.linalg.eigh(G)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Keep top d eigenvectors
        V = eigvecs[:, :d] * np.sqrt(np.maximum(eigvals[:d], 0))
        return V

    def objective_coords(x):
        """Refine in coordinate space."""
        V = x.reshape(n, d)
        d_sq = pdist(V) ** 2
        if np.any(d_sq < 1e-12):
            return 1e10
        T = 1e-5 # Very small, effectively hard min/max
        max_d_sq = np.max(d_sq)
        min_d_sq = np.min(d_sq)
        if min_d_sq < 1e-12:
            return 1e10
        return (max_d_sq / min_d_sq) ** 2

    # Initializations for Gram Matrix
    np.random.seed(42)
    initializations = []
    
    # 1. Random
    V_rand = np.random.randn(n, d)
    G_rand = V_rand @ V_rand.T
    initializations.append(G_rand.flatten())
    
    # 2. Sphere-like (Fibonacci)
    phi = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
    theta = np.pi * (1 + 5 ** 0.5) * np.arange(n)
    V_sphere = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    G_sphere = V_sphere @ V_sphere.T
    initializations.append(G_sphere.flatten())
    
    # 3. Lattice-like
    V_lat = np.random.randn(n, d) * 0.5
    G_lat = V_lat @ V_lat.T
    initializations.append(G_lat.flatten())

    best_points = None
    best_score = 1e10
    
    # Annealing parameters
    temperatures = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 1e-5]
    
    for G0_flat in initializations:
        current_G = G0_flat.copy()
        
        for T in temperatures:
            # Increase weights for constraints as T decreases?
            # Or keep them constant.
            w_psd = 100.0
            w_rank = 100.0
            w_scale = 10.0
            
            obj = make_gram_objective(T, w_psd, w_rank, w_scale)
            
            try:
                res = minimize(obj, current_G, method='L-BFGS-B',
                               options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if res.success or res.fun < 1e9:
                    current_G = res.x
            except Exception:
                pass
        
        # Extract points from final Gram
        G_opt = current_G.reshape(n, n)
        G_opt = (G_opt + G_opt.T) / 2
        points = extract_points_from_gram(G_opt)
        
        # Refine in coordinates
        try:
            res_coords = minimize(objective_coords, points.flatten(), method='L-BFGS-B',
                                  options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            if res_coords.success or res_coords.fun < 1e9:
                refined_points = res_coords.x.reshape(n, d)
                score = _compute_score(refined_points)
                
                if score < best_score:
                    best_score = score
                    best_points = refined_points
        except Exception:
            pass

    if best_points is None:
        return np.random.randn(n, d)
        
    return best_points

# EVOLVE-BLOCK-END
