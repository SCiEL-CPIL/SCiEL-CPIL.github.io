import time
import numpy as np
from scipy.spatial.distance import cdist
from itertools import combinations

N = 14
D = 3
DEADLINE = 140.0
_t0 = None

def _true_score(pts):
    """
    Calculate the squared ratio of max to min pairwise distance.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _gram_to_points(G, max_iter=1000, tol=1e-8):
    """
    Extract points from a Gram matrix.
    We want G = X X^T where X is (N, D).
    This is equivalent to finding a rank-D PSD matrix close to G.
    We use eigenvalue decomposition.
    """
    # Ensure G is symmetric
    G = (G + G.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(G)
    
    # We want the top D eigenvalues to be positive and others close to 0.
    # However, the optimization might have made some "negative" eigenvalues 
    # or spread energy. 
    # A robust way to get a rank-D approximation is to take the top D eigenvectors
    # and scale them by the square root of their eigenvalues (clipped to 0).
    
    # Sort eigenvalues in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Keep top D
    top_vals = eigvals[:D]
    top_vecs = eigvecs[:, :D]
    
    # Clip small negative values (numerical noise)
    top_vals = np.maximum(top_vals, 0)
    sqrt_vals = np.sqrt(top_vals)
    
    # Reconstruct X: (N, D)
    X = top_vecs @ np.diag(sqrt_vals)
    
    # Center the points (Gram matrix optimization often allows for translation drift 
    # if not constrained, but usually G = X X^T implies centered if we assume 
    # the "origin" is the centroid. To be safe, we center X.)
    X = X - X.mean(axis=0, keepdims=True)
    
    return X

def _soft_ratio_obj(G, lam_psd=100.0, lam_rank=10.0, lam_trace=100.0, temp=1.0):
    """
    Objective function for Gram matrix optimization.
    
    We minimize a soft version of (max_dist / min_dist)^2.
    Let D be the distance matrix derived from G.
    d_ij = sqrt(G_ii + G_jj - 2 G_ij).
    
    To handle the non-differentiability of max/min, we use a soft-max/soft-min
    approximation or simply optimize the variance of distances subject to constraints.
    However, the directive asks for "minimize the spectral radius ... relative to min entry".
    Actually, minimizing the ratio is equivalent to making all distances equal (equidistant points).
    Since 14 equidistant points don't exist in 3D, we want to minimize the spread.
    
    A robust smooth surrogate for minimizing the ratio R = (max d / min d)^2 is:
    1. Normalize distances so sum d^2 = const (trace constraint).
    2. Minimize the variance of d^2.
    3. Add penalties for violating PSD, Rank, and Trace.
    
    Let's use: Objective = Var(d^2) + penalties.
    Minimizing variance of squared distances tends to equalize distances, which minimizes the ratio.
    """
    # 1. Extract distances from G
    # G is (N, N). d_ij^2 = G_ii + G_jj - 2 G_ij.
    # We can compute this efficiently.
    diag = np.diag(G)
    d2 = diag[:, None] + diag[None, :] - 2 * G
    # Ensure non-negative (numerical stability)
    d2 = np.maximum(d2, 0)
    np.fill_diagonal(d2, 0)
    
    # Get pairwise squared distances
    d2_flat = d2[np.triu_indices(N, k=1)]
    
    if len(d2_flat) == 0 or np.any(d2_flat < 1e-12):
        return 1e6 # Avoid division by zero issues in log or ratios if used
    
    # 2. Soft Ratio / Variance Objective
    # We want to minimize the spread. 
    # A common metric is the coefficient of variation or simply the variance.
    # Let's use a soft max/min approach to directly target the ratio.
    # Soft max of d: sum(exp(d/temp)) -> log(sum(exp(d/temp))) * temp
    # Soft min of d: -log(sum(exp(-d/temp))) * temp
    # This is expensive. 
    # Alternative: Minimize (Max_d - Min_d)^2 + lambda * Var(d).
    
    max_d2 = np.max(d2_flat)
    min_d2 = np.min(d2_flat)
    
    # If min_d2 is very small, the ratio is huge. We want to push min_d2 up and max_d2 down.
    # Objective: (max_d2 / min_d2) is hard. 
    # Let's use: (max_d2 - min_d2)^2 + penalty * (1/min_d2)
    # This encourages increasing the minimum distance and decreasing the range.
    
    # To keep it smooth and avoid 1/x singularity, use:
    # Obj = (max_d2 - min_d2)^2 + 1.0 / (min_d2 + 1e-6)
    
    obj_val = (max_d2 - min_d2)**2 + 1.0 / (min_d2 + 1e-6)
    
    # 3. Penalties
    # a) PSD Penalty: sum of negative eigenvalues squared
    eigvals_G = np.linalg.eigvalsh(G)
    neg_eig = eigvals_G[eigvals_G < 0]
    penalty_psd = lam_psd * np.sum(neg_eig**2)
    
    # b) Rank Penalty: sum of eigenvalues beyond rank D squared
    # We want eigvals[3:] to be 0.
    if len(eigvals_G) > D:
        high_eig = eigvals_G[D:]
        penalty_rank = lam_rank * np.sum(high_eig**2)
    else:
        penalty_rank = 0.0
        
    # c) Trace Constraint: We want the points to be on the unit sphere (or similar scale).
    # The scale of G determines the distances. 
    # If we don't fix the scale, the optimizer might shrink all distances to 0 
    # to minimize the variance/range, but the 1/min_d2 term prevents that.
    # However, to be safe, we can penalize deviation from a target trace, e.g., N * 1.0 (unit sphere).
    trace_G = np.trace(G)
    penalty_trace = lam_trace * (trace_G - N)**2
    
    return obj_val + penalty_psd + penalty_rank + penalty_trace

def _gradient_step(G, grad, lr, max_norm=1.0):
    """
    Perform a gradient step on the Gram matrix, ensuring symmetry.
    """
    G_new = G - lr * grad
    G_new = (G_new + G_new.T) / 2.0
    return G_new

def _construct_from_gram(G_init, max_steps=2000, lr_init=0.01, time_limit=10.0):
    """
    Optimize the Gram matrix using gradient descent (numerical or analytic if possible, 
    but numerical is safer for complex objectives).
    """
    start_time = time.time()
    G = G_init.copy()
    best_G = G.copy()
    best_score = np.inf
    
    # We need a gradient. Since _soft_ratio_obj involves max/min/eigh, 
    # analytic gradient is complex. We will use finite differences for the gradient 
    # of the objective function. This is slow but robust for N=14 (105 unique params).
    # To speed up, we can use a smaller batch or approximate gradient.
    # Actually, for N=14, 105 params is small enough for finite diff if we are careful.
    
    # Let's try a simpler approach: 
    # Use scipy.optimize.minimize on the vectorized unique elements of G.
    # This is much more efficient and robust than manual gradient descent.
    
    from scipy.optimize import minimize
    
    def vec_to_G(v):
        G_mat = np.zeros((N, N))
        idx = np.triu_indices(N)
        G_mat[idx] = v
        G_mat = G_mat + G_mat.T - np.diag(np.diag(G_mat))
        return G_mat

    def G_to_vec(G_mat):
        idx = np.triu_indices(N)
        return G_mat[idx]
        
    def obj_func(v):
        G_mat = vec_to_G(v)
        return _soft_ratio_obj(G_mat)
        
    # Initial vector
    v0 = G_to_vec(G)
    
    # Run optimization
    try:
        res = minimize(
            obj_func, 
            v0, 
            method='L-BFGS-B', # Good for large scale, box constraints if needed
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        if res.success or res.fun < obj_func(v0):
            G_opt = vec_to_G(res.x)
            pts = _gram_to_points(G_opt)
            score = _true_score(pts)
            if score < best_score:
                best_score = score
                best_G = G_opt
    except Exception:
        pass
        
    return best_G, best_score

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    rng = np.random.default_rng(42)
    best_pts = None
    best_score = np.inf
    
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() - _t0 > DEADLINE - 20:
            break
            
        # 1. Generate Initial Gram Matrix
        # Strategy: Start with a random configuration or a structured one.
        # A good initial guess for Gram is the Gram of a random set of points on the sphere,
        # or a matrix with specific symmetry.
        
        # Let's try initializing with the Gram matrix of a "perfect" sphere projection
        # or just random points, then let the optimizer find the structure.
        
        # Option A: Random points on sphere
        pts_rand = rng.normal(size=(N, D))
        pts_rand = pts_rand / np.linalg.norm(pts_rand, axis=1, keepdims=True)
        G_init = pts_rand @ pts_rand.T
        
        # Option B: Icosahedron-based (if we have it, but let's stick to generic for now)
        # We can mix: 50% random, 50% from a known good discrete set (e.g. Fibonacci)
        
        if r % 2 == 1:
            # Fibonacci sphere init
            i = np.arange(N)
            phi_angle = np.arccos(1 - 2 * (i + 0.5) / N)
            theta_angle = np.pi * (1 + 5 ** 0.5) * i
            x = np.sin(phi_angle) * np.cos(theta_angle)
            y = np.sin(phi_angle) * np.sin(theta_angle)
            z = np.cos(phi_angle)
            pts_init = np.column_stack([x, y, z])
            G_init = pts_init @ pts_init.T
        else:
            G_init = G_init # Use random
            
        # 2. Optimize Gram Matrix
        G_opt, score = _construct_from_gram(G_init, time_limit=10.0)
        
        if score < best_score:
            best_score = score
            # Extract points from best Gram
            best_pts = _gram_to_points(G_opt)
            
        # Early stopping check
        if best_score < 1.1:
            break

    # 3. Final Polish: 
    # The Gram optimization might yield points that are not perfectly centered or scaled.
    # Also, we can do a few steps of local coordinate optimization on the result.
    
    if best_pts is not None:
        pts = best_pts.copy()
        # Simple force-directed relaxation to clean up numerical artifacts
        for _ in range(50):
            d = cdist(pts, pts)
            np.fill_diagonal(d, np.inf)
            mn = d.min()
            mx = d.max()
            if mn < 1e-6: break
            
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(N):
                    if i == j: continue
                    diff = pts[i] - pts[j]
                    dist = d[i, j]
                    
                    # Push apart if close, pull together if far
                    # Target distance is roughly the mean distance
                    target = np.mean(d[np.triu_indices(N, k=1)])
                    
                    # Force proportional to (dist - target)
                    f = (dist - target) * diff / dist
                    forces[i] -= f # If dist > target, pull back (negative gradient of dist^2)
                    # Wait, if dist > target, we want to reduce dist, so force should be attractive.
                    # Gradient of dist^2 w.r.t x_i is 2(x_i - x_j). 
                    # We want to minimize variance. 
                    # Let's stick to the previous simple logic which worked in the seed:
                    # Push min, pull max.
                    
                    w_min = 1.0 / (1.0 + (dist - mn)**2 * 100)
                    w_max = 1.0 / (1.0 + (mx - dist)**2 * 100)
                    
                    f_push = diff / dist * w_min
                    f_pull = -diff / dist * w_max
                    
                    forces[i] += f_push + f_pull
            
            norm_f = np.linalg.norm(forces, axis=1, keepdims=True)
            norm_f[norm_f < 1e-9] = 1.0
            forces = forces / norm_f
            
            pts += 0.01 * forces
            pts = pts - pts.mean(axis=0)
            # Do NOT normalize to sphere strictly, as the optimal shape might not be on a sphere
            # but for N=14 in 3D, it's likely close to spherical. 
            # However, the Gram optimization already handled the scale. 
            # Let's just center.
            
            new_score = _true_score(pts)
            if new_score < best_score:
                best_score = new_score
                best_pts = pts.copy()
            else:
                pts = best_pts.copy()
                break

    # Fallback
    if best_pts is None:
        i = np.arange(N)
        phi = np.arccos(1 - 2*(i+0.5)/N)
        theta = np.pi*(1+5**0.5)*i
        x = np.sin(phi)*np.cos(theta)
        y = np.sin(phi)*np.sin(theta)
        z = np.cos(phi)
        best_pts = np.column_stack([x, y, z])

    return best_pts
