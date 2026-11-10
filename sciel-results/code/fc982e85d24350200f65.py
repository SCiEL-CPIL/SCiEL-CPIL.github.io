# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
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

def _get_gram_matrix(points: np.ndarray) -> np.ndarray:
    """Compute the Gram matrix of the points."""
    return points @ points.T

def _reconstruct_points(gram: np.ndarray) -> np.ndarray:
    """
    Reconstruct points from a Gram matrix.
    The Gram matrix G = P P^T.
    We want P such that P P^T = G.
    Since we are in 3D, G should have rank <= 3.
    We perform eigendecomposition G = V D V^T.
    If G is PSD, D has non-negative eigenvalues.
    P = V_k D_k^{1/2} where V_k are eigenvectors corresponding to the largest 3 eigenvalues.
    """
    # Symmetrize to remove numerical noise
    gram = (gram + gram.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(gram)
    
    # Sort eigenvalues in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Take the top 3 eigenvalues and eigenvectors
    # Ensure eigenvalues are non-negative
    top_vals = np.maximum(eigvals[:3], 0)
    top_vecs = eigvecs[:, :3]
    
    # P = top_vecs * sqrt(top_vals)
    # top_vecs is (14, 3), top_vals is (3,)
    # We need to scale columns of top_vecs by sqrt(top_vals)
    sqrt_vals = np.sqrt(top_vals)
    points = top_vecs * sqrt_vals
    
    return points

def _gram_to_dist(gram: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distances from the Gram matrix.
    d_ij^2 = G_ii + G_jj - 2 G_ij
    """
    n = gram.shape[0]
    # Extract diagonal
    diag = np.diag(gram)
    # Compute squared distances
    # dist_sq[i, j] = diag[i] + diag[j] - 2 * gram[i, j]
    dist_sq = diag[:, None] + diag[None, :] - 2 * gram
    # Numerical stability: clip negative values
    dist_sq = np.maximum(dist_sq, 0)
    # Take upper triangle
    iu = np.triu_indices(n, k=1)
    return np.sqrt(dist_sq[iu])

def _soft_min(dists, beta):
    """Smooth approximation of the minimum."""
    # log(sum(exp(-beta * d))) / -beta
    # For stability, subtract max
    x = -beta * dists
    max_x = np.max(x)
    log_sum_exp = max_x + np.log(np.sum(np.exp(x - max_x)))
    return -log_sum_exp / beta

def _soft_max(dists, beta):
    """Smooth approximation of the maximum."""
    # log(sum(exp(beta * d))) / beta
    x = beta * dists
    max_x = np.max(x)
    log_sum_exp = max_x + np.log(np.sum(np.exp(x - max_x)))
    return log_sum_exp / beta

def _objective_gram(gram_params, beta):
    """
    Objective function for Gram matrix optimization.
    We optimize over the entries of the Gram matrix.
    To ensure PSD, we can parameterize G = L L^T + epsilon * I, or just use the fact that
    if we start from a PSD matrix and use small steps, it might stay PSD.
    However, standard minimizers don't enforce PSD.
    
    Alternative: Parameterize G as M M^T where M is 14x14? No, rank constraint.
    Better: Optimize the points directly but use the Gram structure implicitly?
    No, the directive says "optimize the Gram matrix".
    
    Let's parameterize the Gram matrix G.
    We want to maximize (d_min / d_max)^2.
    This is equivalent to maximizing log(d_min) - log(d_max).
    Using soft approximations:
    f = _soft_min(dists, beta) - _soft_max(dists, beta)
    
    We need to ensure G is PSD.
    If we just minimize -f(G), the optimizer might go to non-PSD regions.
    We can add a penalty for negative eigenvalues.
    
    Or, we can parameterize G = W W^T where W is 14x3?
    If we use W (14x3), then G = W W^T is automatically PSD and rank <= 3.
    This is equivalent to optimizing the points W directly!
    Wait, if G = W W^T, then W are the points.
    So "optimizing the Gram matrix" with the constraint rank <= 3 and PSD is exactly
    equivalent to optimizing the points in R^3.
    
    So why is this different?
    The current program optimizes points using Nelder-Mead on the raw metric.
    The inspiration suggests using a smooth log-sum-exp objective.
    The current program uses Nelder-Mead (nonsmooth, derivative-free).
    I will use L-BFGS-B or similar on the points, but with a SMOOTH objective function
    that approximates the min/max ratio.
    
    The key difference from the current program is the OBJECTIVE FUNCTION and the OPTIMIZER.
    Current: Nelder-Mead on exact (d_min/d_max)^2.
    New: L-BFGS-B on smooth approximation log(soft_min) - log(soft_max).
    
    This allows for gradient-based optimization which can be much faster and more effective
    if the smooth landscape guides us to the right basin.
    """
    pts = gram_params.reshape(14, 3)
    
    # Calculate distances
    dists = pdist(pts)
    
    # Calculate smooth min and max
    d_min_soft = _soft_min(dists, beta)
    d_max_soft = _soft_max(dists, beta)
    
    # Avoid log(0)
    eps = 1e-12
    d_min_soft = np.maximum(d_min_soft, eps)
    d_max_soft = np.maximum(d_max_soft, eps)
    
    # Maximize log(d_min) - log(d_max)
    # Minimize -(log(d_min) - log(d_max))
    obj = -(np.log(d_min_soft) - np.log(d_max_soft))
    
    return obj

def _refine_smooth(points: np.ndarray, beta_schedule: list, time_limit: float) -> np.ndarray:
    """
    Refine points using L-BFGS-B with increasing beta (annealing).
    """
    start_time = time.time()
    best_points = points.copy()
    best_metric = _evaluate(points)
    
    current_points = points.copy()
    
    for beta in beta_schedule:
        if time.time() > start_time + time_limit:
            break
            
        def obj_func(params):
            return _objective_gram(params, beta)
            
        try:
            result = minimize(
                obj_func,
                current_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            refined_pts = result.x.reshape(14, 3)
            
            # Evaluate on the exact metric
            refined_metric = _evaluate(refined_pts)
            
            if refined_metric > best_metric:
                best_metric = refined_metric
                best_points = refined_pts.copy()
                current_points = refined_pts.copy()
            else:
                # If it got worse, keep the old points but maybe try to nudge?
                # For now, just keep current_points as is for the next beta
                pass
                
        except Exception:
            pass
            
    return best_points

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

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses smooth gradient-based optimization with annealing.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit
    
    rng = np.random.RandomState(42)
    ico = _icosahedron_vertices()
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Icosahedron + Poles
    p1 = np.array([0, 0, 1.0])
    p2 = np.array([0, 0, -1.0])
    cand = np.vstack([ico, p1.reshape(1, -1), p2.reshape(1, -1)])
    candidates.append(_normalize_sphere(cand))
    
    # 2. Random restarts
    for _ in range(30):
        cand = rng.randn(14, 3)
        candidates.append(_normalize_sphere(cand))
        
    # 3. Greedy construction
    greedy_pts = np.zeros((14, 3))
    greedy_pts[0] = rng.randn(3)
    greedy_pts[0] /= np.linalg.norm(greedy_pts[0])
    for i in range(1, 14):
        best_pos = None
        best_min_dist = -1
        samples = rng.randn(2000, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        for s in samples:
            dists = np.linalg.norm(greedy_pts[:i] - s, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = s
        greedy_pts[i] = best_pos
    candidates.append(greedy_pts)
    
    # Evaluate and sort candidates
    scored_candidates = []
    for cand in candidates:
        m = _evaluate(cand)
        scored_candidates.append((m, cand))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_points = scored_candidates[0][1].copy()
    best_metric = scored_candidates[0][0]
    
    # Refine top candidates using smooth gradient optimization
    # Use a beta schedule: start low (very smooth) and increase (sharper)
    beta_schedule = [10, 50, 100, 500, 1000, 5000]
    
    for i, (m, cand) in enumerate(scored_candidates[:15]):
        if time.time() > deadline - 30:
            break
            
        # Time limit per refinement
        time_per_refine = max(5, (deadline - 30 - time.time()) / (15 - i))
        
        refined = _refine_smooth(cand, beta_schedule, time_per_refine)
        refined_metric = _evaluate(refined)
        
        if refined_metric > best_metric:
            best_metric = refined_metric
            best_points = refined.copy()
            
    # Final polish with Nelder-Mead on the exact metric to escape smooth minima
    if best_points is not None and time.time() < deadline - 10:
        def exact_obj(params):
            pts = params.reshape(14, 3)
            return -_evaluate(pts)
            
        try:
            result = minimize(
                exact_obj,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            polished = result.x.reshape(14, 3)
            polished_metric = _evaluate(polished)
            
            if polished_metric > best_metric:
                best_metric = polished_metric
                best_points = polished.copy()
        except Exception:
            pass

    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        best_points = ico

    return best_points

# EVOLVE-BLOCK-END
