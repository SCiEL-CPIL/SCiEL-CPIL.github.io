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
    dists = pdist(points)
    if len(dists) == 0:
        return 1e10
    min_d = np.min(dists)
    max_d = np.max(dists)
    if min_d < 1e-10:
        return 1e10
    return (max_d / min_d) ** 2

def _gram_from_points(points):
    """Compute Gram matrix G = X @ X.T from point coordinates."""
    return points @ points.T

def _points_from_gram(G, d):
    """Extract d-dimensional points from a Gram matrix via eigen-decomposition."""
    eigvals, eigvecs = np.linalg.eigh(G)
    # Keep top d eigenvalues
    idx = np.argsort(eigvals)[::-1][:d]
    V = eigvecs[:, idx] * np.sqrt(np.maximum(eigvals[idx], 0))
    return V

def _rank_penalty(G, d, strength=1.0):
    """Penalty for eigenvalues beyond rank d (encourages low-rank structure)."""
    eigvals = np.linalg.eigvalsh(G)
    eigvals = np.sort(eigvals)[::-1]
    # Penalize eigenvalues beyond the first d
    excess = eigvals[d:]
    return strength * np.sum(excess ** 2)

def _sdp_objective_and_grad(G_flat, n, d, indices, num_pairs, rank_penalty_strength=0.1):
    """
    Objective for SDP relaxation: minimize distance ratio with rank penalty.
    G_flat: flattened upper-triangular Gram matrix
    Returns objective value and gradient w.r.t. G_flat
    """
    # Reconstruct full symmetric Gram matrix from upper triangle
    G = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i, n):
            G[i, j] = G_flat[idx]
            G[j, i] = G_flat[idx]
            idx += 1
    
    # Ensure PSD by projecting
    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals = np.maximum(eigvals, 0)
    G_psd = (eigvecs * eigvals) @ eigvecs.T
    
    # Compute squared distances from Gram matrix
    d_sq = np.zeros(num_pairs)
    for k, (i, j) in enumerate(indices):
        d_sq[k] = G_psd[i, i] + G_psd[j, j] - 2 * G_psd[i, j]
    
    min_d_sq = np.min(d_sq)
    max_d_sq = np.max(d_sq)
    
    if min_d_sq < 1e-12:
        return 1e10, np.zeros_like(G_flat)
    
    # Smooth approximation for max and min
    T = 0.05
    max_approx = T * np.log(np.sum(np.exp(d_sq / T)))
    min_approx = -T * np.log(np.sum(np.exp(-d_sq / T)))
    
    if min_approx < 1e-12:
        return 1e10, np.zeros_like(G_flat)
    
    ratio = max_approx / min_approx
    
    # Add rank penalty
    rp = _rank_penalty(G_psd, d, rank_penalty_strength)
    
    obj = ratio + rp
    
    # Numerical gradient
    eps = 1e-6
    grad = np.zeros_like(G_flat)
    for idx_g in range(len(G_flat)):
        G_plus = G_flat.copy()
        G_plus[idx_g] += eps
        f_plus, _ = _sdp_objective_and_grad(G_plus, n, d, indices, num_pairs, rank_penalty_strength)
        grad[idx_g] = (f_plus - obj) / eps
    
    return obj, grad

def construct():
    n = N
    d = D
    
    indices = _pairwise_indices(n)
    num_pairs = len(indices)
    
    # Number of unique elements in symmetric matrix (upper triangle including diagonal)
    num_upper = n * (n + 1) // 2
    
    np.random.seed(42)
    
    # Generate multiple initial Gram matrices
    initial_Gs = []
    
    # 1. Random PSD initialization via low-rank factorization
    V_rand = np.random.randn(n, d)
    G_rand = V_rand @ V_rand.T
    initial_Gs.append(G_rand)
    
    # 2. Fibonacci sphere initialization
    G_fib = np.zeros((n, n))
    golden_ratio = (1 + np.sqrt(5)) / 2
    points_fib = np.zeros((n, d))
    for i in range(n):
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * (i + 0.5) / n)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        points_fib[i] = [x, y, z]
    G_fib = points_fib @ points_fib.T
    initial_Gs.append(G_fib)
    
    # 3. Regular simplex-like (projected to 3D)
    # Start with regular (n-1)-simplex in n-1 dims, project to 3D
    # For n=14, we can't have a regular 13-simplex in 3D, so use random
    V_simplex = np.random.randn(n, d)
    # Add some structure: center of mass at origin
    V_simplex -= V_simplex.mean(axis=0, keepdims=True)
    G_simplex = V_simplex @ V_simplex.T
    initial_Gs.append(G_simplex)
    
    # 4. Two-cluster initialization (two groups of 7)
    V_cluster = np.zeros((n, d))
    center1 = np.array([1.0, 0.0, 0.0])
    center2 = np.array([-1.0, 0.0, 0.0])
    for i in range(7):
        V_cluster[i] = center1 + 0.3 * np.random.randn(d)
        V_cluster[i + 7] = center2 + 0.3 * np.random.randn(d)
    G_cluster = V_cluster @ V_cluster.T
    initial_Gs.append(G_cluster)
    
    best_points = None
    best_score = 1e10
    
    for G_init in initial_Gs:
        # Flatten upper triangle
        G_flat = np.zeros(num_upper)
        idx = 0
        for i in range(n):
            for j in range(i, n):
                G_flat[idx] = G_init[i, j]
                idx += 1
        
        # Optimize using L-BFGS-B
        try:
            def obj_func(x):
                val, _ = _sdp_objective_and_grad(x, n, d, indices, num_pairs, 0.1)
                return val
            
            res = minimize(obj_func, G_flat, method='L-BFGS-B',
                          options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < best_score:
                # Reconstruct Gram matrix
                G_opt = np.zeros((n, n))
                idx = 0
                for i in range(n):
                    for j in range(i, n):
                        G_opt[i, j] = res.x[idx]
                        G_opt[j, i] = res.x[idx]
                        idx += 1
                
                # Extract points
                points_opt = _points_from_gram(G_opt, d)
                
                # Compute actual score
                score = _compute_score(points_opt)
                if score < best_score:
                    best_score = score
                    best_points = points_opt
                    
        except Exception:
            pass
    
    # If SDP approach didn't find good points, fall back to coordinate optimization
    if best_points is None or best_score > 3.0:
        # Use coordinate-based optimization as fallback
        def coord_objective(x):
            V = x.reshape(n, d)
            d_sq = np.zeros(num_pairs)
            for k, (i, j) in enumerate(indices):
                diff = V[i] - V[j]
                d_sq[k] = np.dot(diff, diff)
            
            min_d_sq = np.min(d_sq)
            max_d_sq = np.max(d_sq)
            
            if min_d_sq < 1e-12:
                return 1e10
            
            T = 0.05
            max_approx = T * np.log(np.sum(np.exp(d_sq / T)))
            min_approx = -T * np.log(np.sum(np.exp(-d_sq / T)))
            
            if min_approx < 1e-12:
                return 1e10
            
            return max_approx / min_approx
        
        # Try from best Gram-derived points or random
        if best_points is not None:
            x0 = best_points.flatten()
        else:
            x0 = np.random.randn(n * d)
        
        try:
            res = minimize(coord_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            points_coord = res.x.reshape(n, d)
            score_coord = _compute_score(points_coord)
            if score_coord < best_score:
                best_score = score_coord
                best_points = points_coord
        except Exception:
            pass
    
    # Final refinement with coordinate optimization
    if best_points is not None:
        def coord_objective_final(x):
            V = x.reshape(n, d)
            d_sq = np.zeros(num_pairs)
            for k, (i, j) in enumerate(indices):
                diff = V[i] - V[j]
                d_sq[k] = np.dot(diff, diff)
            
            min_d_sq = np.min(d_sq)
            max_d_sq = np.max(d_sq)
            
            if min_d_sq < 1e-12:
                return 1e10
            
            return (max_d_sq / min_d_sq)
        
        try:
            res = minimize(coord_objective_final, best_points.flatten(), method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            points_final = res.x.reshape(n, d)
            score_final = _compute_score(points_final)
            if score_final < best_score:
                best_points = points_final
        except Exception:
            pass
    
    if best_points is None:
        # Absolute fallback
        return np.random.randn(n, d)
    
    return best_points

# EVOLVE-BLOCK-END
