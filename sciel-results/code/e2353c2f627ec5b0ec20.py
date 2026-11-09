import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import itertools

def _evaluate(points: np.ndarray) -> float:
    """Compute the metric (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def _soft_min(dists, beta):
    """
    Smooth approximation of the minimum using log-sum-exp.
    As beta -> inf, this approaches the true minimum.
    """
    min_d = np.min(dists)
    shifted = -beta * (dists - min_d)
    # Clip to avoid underflow/overflow in exp
    shifted = np.clip(shifted, -700, 700)
    sum_exp = np.sum(np.exp(shifted))
    log_sum_exp = np.log(sum_exp) if sum_exp > 0 else 0.0
    return min_d - (log_sum_exp / beta)

def _soft_max(dists, beta):
    """
    Smooth approximation of the maximum using log-sum-exp.
    As beta -> inf, this approaches the true maximum.
    """
    max_d = np.max(dists)
    shifted = beta * (dists - max_d)
    shifted = np.clip(shifted, -700, 700)
    sum_exp = np.sum(np.exp(shifted))
    log_sum_exp = np.log(sum_exp) if sum_exp > 0 else 0.0
    return max_d + (log_sum_exp / beta)

def _smooth_metric(params, beta):
    """
    Compute the smooth metric.
    We minimize the negative of (d_min / d_max)^2.
    To handle scale invariance, we normalize the points to the unit sphere
    inside the objective function.
    """
    pts = params.reshape(14, 3)
    # Normalize to unit sphere to remove scale degree of freedom
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    pts = pts / norms
    
    dists = pdist(pts)
    if np.any(~np.isfinite(dists)):
        return 1e10
        
    d_min = _soft_min(dists, beta)
    d_max = _soft_max(dists, beta)
    
    if d_max < 1e-10:
        return 1e10
        
    ratio = d_min / d_max
    metric = ratio * ratio
    return -metric

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

def _normalize(points: np.ndarray) -> np.ndarray:
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    ico = _icosahedron_vertices()
    
    best_points = None
    best_metric = 0.0
    
    # Generate candidate initial configurations
    # Strategy: Place the 2 extra points in high-symmetry positions relative to the icosahedron.
    # 1. Poles (0,0,1) and (0,0,-1) - 5-fold axis
    # 2. Face centers (3-fold axes)
    # 3. Edge midpoints (2-fold axes)
    
    candidates = []
    
    # 1. 5-fold axis (Poles)
    p1 = np.array([0, 0, 1])
    p2 = np.array([0, 0, -1])
    cand = np.vstack([ico, p1.reshape(1, -1), p2.reshape(1, -1)])
    candidates.append(_normalize(cand))
    
    # 2. 3-fold axes (Face centers)
    # The face centers of the icosahedron correspond to the vertices of a dodecahedron.
    # We can generate them by normalizing the sum of 3 adjacent vertices.
    # Instead of finding adjacency, we can use the known dodecahedron vertices.
    phi = (1 + np.sqrt(5)) / 2
    # Dodecahedron vertices (normalized)
    # (±1, ±1, ±1)
    # (0, ±1/phi, ±phi)
    # (±1/phi, ±phi, 0)
    # (±phi, 0, ±1/phi)
    dodeca_vertices = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                dodeca_vertices.append([sx, sy, sz])
    for sx in [-1, 1]:
        for sz in [-1, 1]:
            dodeca_vertices.append([0, sx/phi, sz*phi])
            dodeca_vertices.append([sx/phi, sz*phi, 0])
            dodeca_vertices.append([sz*phi, 0, sx/phi])
            
    dodeca_vertices = np.array(dodeca_vertices, dtype=float)
    dodeca_vertices = _normalize(dodeca_vertices)
    
    # Pick a pair of opposite face centers (e.g., index 0 and 7 if they are opposite)
    # Let's just pick the first one and its negative
    fc1 = dodeca_vertices[0]
    fc2 = -fc1
    cand = np.vstack([ico, fc1.reshape(1, -1), fc2.reshape(1, -1)])
    candidates.append(_normalize(cand))
    
    # 3. 2-fold axes (Edge midpoints)
    # Edge midpoints of icosahedron are vertices of a cuboctahedron? No, they form a cuboctahedron only if...
    # Actually, the 2-fold axes pass through midpoints of edges.
    # Let's just use a random point on the sphere but ensure it's not too close to vertices?
    # Better: Use the vertices of a cube inscribed in the sphere? No.
    # Let's just add a few random high-quality starts from the best icosahedron + random points.
    
    rng = np.random.RandomState(42)
    for _ in range(5):
        p1 = rng.randn(3)
        p1 = p1 / np.linalg.norm(p1)
        p2 = -p1 # Antipodal is usually good for symmetry
        cand = np.vstack([ico, p1.reshape(1, -1), p2.reshape(1, -1)])
        candidates.append(_normalize(cand))
        
    # Optimization loop
    # We will use L-BFGS-B with a smooth objective.
    # We increase beta (sharpness) over time to converge to the exact metric.
    
    betas = [10, 50, 100, 200, 500]
    
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 10:
            break
            
        current_best = cand.copy()
        current_metric = _evaluate(current_best)
        
        # Multi-stage optimization
        for beta in betas:
            if time.time() > deadline - 10:
                break
                
            def objective(params):
                return _smooth_metric(params, beta)
                
            try:
                result = minimize(
                    objective,
                    current_best.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                new_points = result.x.reshape(14, 3)
                new_points = _normalize(new_points)
                new_metric = _evaluate(new_points)
                
                if new_metric > current_metric:
                    current_metric = new_metric
                    current_best = new_points
            except Exception:
                pass
                
        if current_metric > best_metric:
            best_metric = current_metric
            best_points = current_best.copy()

    # Final polish on the best found
    if best_points is not None and time.time() < deadline - 5:
        # One more high-beta refinement
        beta_final = 1000
        def objective_final(params):
            return _smooth_metric(params, beta_final)
            
        try:
            result = minimize(
                objective_final,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            refined_points = result.x.reshape(14, 3)
            refined_points = _normalize(refined_points)
            refined_metric = _evaluate(refined_points)
            
            if refined_metric > best_metric:
                best_points = refined_points
                best_metric = refined_metric
        except Exception:
            pass

    # Fallback
    if best_points is None:
        best_points = candidates[0]
        
    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        best_points = candidates[0]
        
    return best_points
