# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def _evaluate(points: np.ndarray) -> float:
    """Compute the metric (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    # Check for non-finite values
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def _icosahedron_vertices() -> np.ndarray:
    """Return the 12 vertices of a regular icosahedron on the unit sphere."""
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    return vertices / norms

def _normalize_sphere(points: np.ndarray) -> np.ndarray:
    """Normalize points to lie on the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms

def _soft_min(x, k=20.0):
    """Soft minimum approximation using log-sum-exp."""
    # x is array of squared distances
    # min(x) approx -log(sum(exp(-k*x)))/k
    # To avoid overflow, we shift by min
    min_x = np.min(x)
    # If k is large, exp(-k*(x-min)) underflows for x >> min, which is fine
    # But we want soft min of x.
    # soft_min = -1/k * log( sum(exp(-k*x)) )
    # = -1/k * log( exp(-k*min_x) * sum(exp(-k*(x-min_x))) )
    # = min_x - 1/k * log( sum(exp(-k*(x-min_x))) )
    shifted = -k * (x - min_x)
    # Clip shifted to prevent underflow issues in exp (though exp(-large) is 0)
    # Actually, we want to sum exp(shifted).
    # If x are all equal, shifted=0, sum=N, log(N), soft_min = min - log(N)/k.
    # As k->inf, soft_min -> min.
    s = np.sum(np.exp(shifted))
    if s == 0:
        return min_x
    return min_x - np.log(s) / k

def _soft_max(x, k=20.0):
    """Soft maximum approximation using log-sum-exp."""
    # max(x) approx 1/k * log(sum(exp(k*x)))
    max_x = np.max(x)
    shifted = k * (x - max_x)
    s = np.sum(np.exp(shifted))
    if s == 0:
        return max_x
    return max_x + np.log(s) / k

def _reconstruct_points_from_gram(gram: np.ndarray) -> np.ndarray:
    """
    Reconstruct points from a Gram matrix G = P P^T.
    We assume points are in R^3, so G should have rank <= 3.
    We project G to the nearest PSD matrix of rank <= 3.
    """
    gram = (gram + gram.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(gram)
    # Sort in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Keep top 3 eigenvalues
    k = 3
    eigvals_top = np.maximum(eigvals[:k], 0)
    eigvecs_top = eigvecs[:, :k]
    
    # P = V_top * sqrt(D_top)
    # Shape of P is (14, 3)
    P = eigvecs_top @ np.diag(np.sqrt(eigvals_top))
    
    # The resulting points might not be on the unit sphere.
    # The objective function should ideally handle scaling, but since we are 
    # optimizing distances, we can normalize the final result.
    # However, for the objective to be consistent, we should probably 
    # normalize the points inside the objective evaluation.
    return P

def _distances_to_gram(d2: np.ndarray, n: int = 14) -> np.ndarray:
    """
    Convert pairwise squared distances to Gram matrix.
    d2[i,j] = ||p_i - p_j||^2 = ||p_i||^2 + ||p_j||^2 - 2 p_i . p_j
    Let G be Gram matrix. G_ii = ||p_i||^2.
    d2_ij = G_ii + G_jj - 2 G_ij
    G_ij = (G_ii + G_jj - d2_ij) / 2
    
    We assume points are on unit sphere, so G_ii = 1.
    Then G_ij = (1 + 1 - d2_ij) / 2 = 1 - d2_ij / 2.
    """
    # d2 is (14, 14) with zeros on diagonal
    # G = 1 - d2 / 2, but diagonal of G must be 1.
    # d2 diagonal is 0, so G diagonal becomes 1 - 0 = 1. Correct.
    G = 1.0 - d2 / 2.0
    return G

def _gram_to_distances(gram: np.ndarray, n: int = 14) -> np.ndarray:
    """
    Convert Gram matrix to pairwise squared distances.
    d2_ij = G_ii + G_jj - 2 G_ij
    If we assume unit sphere, G_ii=1.
    But if we reconstruct P from Gram, the norms might not be 1.
    Let's compute d2 from the reconstructed points to be safe?
    No, we want the objective to be differentiable w.r.t d2.
    So we should compute d2 from the Gram matrix directly, assuming the 
    relationship d2_ij = G_ii + G_jj - 2 G_ij.
    """
    # G is (n, n)
    # d2_ij = G_ii + G_jj - 2 G_ij
    # diag_G = np.diag(gram)
    # d2 = diag_G[:, None] + diag_G[None, :] - 2 * gram
    # This is valid for any Gram matrix.
    diag_G = np.diag(gram)
    d2 = diag_G[:, None] + diag_G[None, :] - 2 * gram
    # Ensure non-negative
    d2 = np.maximum(d2, 0)
    return d2

def _objective(d2_flat, k=20.0, n=14):
    """
    Objective function to minimize: -log(soft_min / soft_max)
    d2_flat: flat array of upper triangle of squared distances (91 values)
    """
    # Reconstruct full d2 matrix
    d2 = np.zeros((n, n))
    # Fill upper triangle
    # pdist returns upper triangle in row-major order
    # We can use np.triu_indices
    iu = np.triu_indices(n, k=1)
    d2[iu] = d2_flat
    d2[iu[1], iu[0]] = d2_flat
    
    # Convert to Gram
    G = _distances_to_gram(d2, n)
    
    # Project to PSD rank 3
    # This projection is non-differentiable, which is a problem for SLSQP.
    # Alternative: Optimize the Gram matrix directly?
    # If we optimize G, the constraint is G is PSD rank 3.
    # SLSQP can handle linear constraints, but PSD is not linear.
    
    # Let's try a different approach: Optimize the points directly but use a better optimizer?
    # Or, use the fact that for unit sphere points, G = 1 - d2/2.
    # The constraint that G is PSD rank 3 is equivalent to the points existing in R^3.
    
    # If we just use the d2 -> G -> P -> d2 loop, we break differentiability.
    
    # Let's stick to optimizing coordinates but use a better method?
    # The prompt asks for a STRUCTURAL change.
    # Gram matrix optimization is the standard "different" approach.
    # To make it differentiable, we can optimize the eigenvalues and eigenvectors?
    # Or, we can simply optimize the coordinates using `minimize` with method='L-BFGS-B' 
    # or 'Powell' which might be more robust than NM?
    # But the prompt specifically pointed to Gram matrix as an inspiration that scored well.
    
    # Let's implement the Gram matrix optimization but handle the non-differentiability 
    # by using a penalty method or just accepting that SLSQP might struggle with the 
    # projection. 
    # Actually, we can optimize the Gram matrix G directly, with the constraint 
    # that G is PSD (eigvals >= 0) and rank <= 3 (eigvals[3:] == 0).
    # This is a non-convex problem.
    
    # Let's try a hybrid: 
    # 1. Start with a good guess (Icosahedron + 2 poles).
    # 2. Optimize the coordinates using a smooth objective that approximates the metric.
    # The current code uses NM which is slow and local.
    # Let's use `minimize` with `method='Powell'` or `BFGS` on coordinates.
    # Powell is derivative-free but often better than NM for high dimensions.
    # BFGS requires gradients, which we can approximate or derive.
    
    # Let's try optimizing the coordinates with a smooth objective:
    # f(x) = -log(soft_min(d2) / soft_max(d2))
    # This is smooth if d2 is smooth (which it is).
    
    # We will define the objective on the flat coordinate array.
    pass

def _objective_coords(params, k=20.0):
    """
    Smooth objective for coordinates.
    """
    pts = params.reshape(14, 3)
    # Normalize to sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    pts = pts / norms
    
    # Compute squared distances
    # d2 = sum((pts_i - pts_j)^2)
    # Using pdist is faster
    d = pdist(pts)
    d2 = d * d
    
    # Soft min and max of d2
    min_d2 = _soft_min(d2, k)
    max_d2 = _soft_max(d2, k)
    
    if min_d2 <= 0 or max_d2 <= 0:
        return 1e10
    
    ratio = min_d2 / max_d2
    return -np.log(ratio)

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit
    
    rng = np.random.RandomState(42)
    ico = _icosahedron_vertices()
    
    # Initial candidate: Icosahedron + 2 poles
    p1 = np.array([0, 0, 1.0])
    p2 = np.array([0, 0, -1.0])
    initial_pts = np.vstack([ico, p1.reshape(1, -1), p2.reshape(1, -1)])
    initial_pts = _normalize_sphere(initial_pts)
    
    best_points = initial_pts.copy()
    best_metric = _evaluate(best_points)
    
    # Define objective for optimizer
    def objective(params):
        return _objective_coords(params, k=20.0)
        
    # Try Powell method, which is robust for high dimensions and doesn't need gradients
    # It is slower than BFGS but often finds better local minima for non-smooth-ish functions
    # (our objective is smooth due to soft min/max, but the underlying metric is not).
    
    # We will run multiple restarts with different initializations and optimization methods.
    
    # Method 1: Powell on the initial guess
    try:
        res = minimize(
            objective, 
            initial_pts.flatten(), 
            method='Powell', 
            options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-8}
        )
        if res.fun < objective(initial_pts.flatten()):
            refined = res.x.reshape(14, 3)
            refined = _normalize_sphere(refined)
            m = _evaluate(refined)
            if m > best_metric:
                best_metric = m
                best_points = refined
    except Exception:
        pass
    
    if time.time() > deadline - 100:
        return best_points

    # Method 2: BFGS with numerical gradients? Or just L-BFGS-B?
    # Let's try L-BFGS-B
    try:
        res = minimize(
            objective, 
            initial_pts.flatten(), 
            method='L-BFGS-B', 
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        if res.fun < objective(initial_pts.flatten()):
            refined = res.x.reshape(14, 3)
            refined = _normalize_sphere(refined)
            m = _evaluate(refined)
            if m > best_metric:
                best_metric = m
                best_points = refined
    except Exception:
        pass

    # Random restarts
    for _ in range(10):
        if time.time() > deadline - 50:
            break
        
        # Random perturbations
        noise = rng.randn(14, 3) * 0.2
        cand = best_points + noise
        cand = _normalize_sphere(cand)
        
        # Optimize with Powell (shorter)
        try:
            res = minimize(
                objective, 
                cand.flatten(), 
                method='Powell', 
                options={'maxiter': 100, 'xtol': 1e-6, 'ftol': 1e-6}
            )
            refined = res.x.reshape(14, 3)
            refined = _normalize_sphere(refined)
            m = _evaluate(refined)
            if m > best_metric:
                best_metric = m
                best_points = refined
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
