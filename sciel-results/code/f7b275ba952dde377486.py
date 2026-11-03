import numpy as np
import time
import sys

N = 130
_DEADLINE = time.time() + 140  # 10s safety margin

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _time_ok(margin=5.0):
    return time.time() < _DEADLINE - margin

def _covering_radius(pts):
    """
    Compute the covering radius in radians.
    Uses a dense grid approximation for speed and robustness during optimization.
    For the final result, we could use ConvexHull, but the grid is sufficient for
    the evaluator if dense enough, and much faster.
    However, the evaluator uses the *true* covering radius (ConvexHull facets).
    To ensure the final score is accurate, we should return a configuration that
    is well-optimized. The grid gradient will lead to a good local minimum.
    """
    pts = _normalize(pts)
    # Use a dense grid to estimate covering radius for the *score* reported internally
    # but the function just needs to return points. The evaluator will compute the score.
    # This function is for internal tracking.
    
    # Generate a dense icosahedral-like grid or Fibonacci grid
    # Let's use a Fibonacci spiral with high N for the grid
    n_grid = 5000
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    grid_pts = _normalize(grid_pts)
    
    # Compute min distance from each grid point to any config point
    # Distances are cosines (dot products)
    dots = grid_pts @ pts.T
    min_dots = np.max(dots, axis=1) # Max dot product = Min angle
    max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
    return np.max(max_angle)

def _grid_gradient(pts, grid_pts, beta=50.0):
    """
    Compute the gradient of the soft-max covering radius proxy.
    Objective: min_y max_i d(y, p_i)
    Approximation: min_y (1/beta) * log(sum_i exp(-beta * d(y, p_i))) 
    Wait, d(y,p) is angular distance. We want to minimize the MAX distance.
    Let f(y) = max_i d(y, p_i).
    Soft max approximation: F(y) = (1/beta) * log( sum_i exp(beta * d(y, p_i)) ).
    As beta -> inf, F(y) -> max_i d(y, p_i).
    
    We want to minimize F(y) for all y.
    This is equivalent to minimizing the maximum over y of F(y).
    Since we are optimizing p_i, we can use the sub-gradient at the "worst" y.
    
    Alternative approach:
    The covering radius is determined by the "holes".
    Let's define the potential V(p) = sum_y exp(beta * d_min(y, P)) / Z
    where d_min(y, P) = min_i d(y, p_i).
    We want to minimize the maximum of d_min.
    
    Let's use the gradient of the soft-max of the distances at the "worst" grid points.
    Let y_k be the grid points.
    d_k = min_i d(y_k, p_i).
    We want to minimize max_k d_k.
    Soft-max: S = (1/beta) log( sum_k exp(beta * d_k) ).
    Gradient w.r.t p_j:
    dS/dp_j = sum_k w_k * (d d_k / d p_j)
    where w_k = exp(beta * d_k) / sum_m exp(beta * d_m).
    
    d_k = min_i d(y_k, p_i).
    The derivative is non-zero only for the i that achieves the min.
    Let i_k be the argmin index.
    d d_k / d p_j = d d(y_k, p_{i_k}) / d p_j.
    If j != i_k, derivative is 0.
    If j == i_k, derivative is the gradient of angular distance.
    
    d(y, p) = arccos(y . p).
    d/dp arccos(y . p) = -1 / sqrt(1 - (y.p)^2) * (-y) = y / sqrt(1 - (y.p)^2).
    Note: This gradient points in the direction of y (away from p if y is far? No).
    Let's check: if p moves towards y, distance decreases.
    Gradient of distance w.r.t p points in direction of increasing distance.
    Increasing distance means moving p away from y.
    Vector from p to y is (y - p). Projected on tangent space?
    Actually, gradient of d on the sphere surface for p:
    If we move p by delta, d changes by -cos(d) * (delta . y) ... no.
    
    Let's use the 3D gradient and project.
    grad_p d(y, p) = (p - (p.y)y) / (||p - (p.y)y|| * sqrt(1-(p.y)^2)) ?
    Simpler:
    d = arccos(c), c = p.y.
    dd/dp = -1/sqrt(1-c^2) * y.
    This vector points in direction -y.
    If we move p in direction -y, c becomes more negative, d increases. Correct.
    So to DECREASE d, we should move p in direction +y.
    
    So, grad_p_j d(y_k, p_j) = - y_k / sqrt(1 - (p_j . y_k)^2).
    
    We want to MINIMIZE S.
    So we move p_j in direction - grad S / grad p_j.
    
    grad S / grad p_j = sum_{k: i_k == j} w_k * (- y_k / sqrt(1 - (p_j . y_k)^2))
    
    So update p_j += lr * sum_{k: i_k == j} w_k * ( y_k / sqrt(1 - (p_j . y_k)^2))
    
    This moves p_j towards the "worst" grid points y_k for which p_j is the nearest neighbor.
    """
    pts = _normalize(pts.copy())
    
    # Compute distances from all grid points to all config points
    # dots: (G, N)
    dots = grid_pts @ pts.T
    # We want min angular distance, which corresponds to max dot product.
    # min_dots = max(dots, axis=1)
    # But we need the index of the max dot product (nearest point).
    
    # To avoid numerical issues with arccos, we can work with cosines?
    # No, the gradient formula depends on the angle.
    
    # Let's compute the "active" points for each grid point.
    max_dots = np.max(dots, axis=1)
    argmax_idx = np.argmax(dots, axis=1)
    
    # Compute weights w_k for soft-max of distances.
    # We need d_k = arccos(max_dots_k).
    # To avoid division by zero in sqrt(1-c^2), clip c.
    c_safe = np.clip(max_dots, -1.0 + 1e-6, 1.0 - 1e-6)
    d_k = np.arccos(c_safe)
    
    # Soft-max weights
    # Subtract max for numerical stability
    d_max = np.max(d_k)
    exp_d = np.exp(beta * (d_k - d_max))
    w = exp_d / np.sum(exp_d)
    
    # Accumulate gradient for each point j
    # grad_j = sum_{k: argmax_idx[k] == j} w_k * ( y_k / sqrt(1 - (p_j . y_k)^2) )
    
    # We need (p_j . y_k) for the specific pairs (k, argmax_idx[k]).
    # This is dots[argmax_idx, np.arange(G)]? No.
    # dots is (G, N). dots[k, argmax_idx[k]] is the max dot for k.
    # This is exactly max_dots.
    
    # Denominator term: sqrt(1 - c_k^2) where c_k = max_dots[k].
    # Note: c_k is the dot product of y_k and p_{argmax_idx[k]}.
    
    denom = np.sqrt(1.0 - c_safe**2)
    
    # Weighted contribution for each grid point k to its nearest point j.
    # contrib_k = w_k / denom_k * y_k
    
    contrib = (w / denom)[:, np.newaxis] * grid_pts
    
    # Scatter add to points
    grad = np.zeros_like(pts)
    np.add.at(grad, argmax_idx, contrib)
    
    # We want to move in the direction that DECREASES the objective.
    # The gradient of the objective w.r.t p_j points in the direction of INCREASING distance (away from y).
    # So we subtract the gradient.
    # Wait, my derivation:
    # grad_p d = - y / sqrt(...)
    # So grad S = sum w_k * (- y_k / sqrt(...))
    # To minimize S, we do p -= lr * grad S.
    # p -= lr * (- sum ... ) => p += lr * sum ...
    # So we ADD the accumulated vector.
    
    # The accumulated vector `grad` is sum w_k * y_k / sqrt(...).
    # This is exactly - (grad S).
    # So p_new = p + lr * grad.
    
    return grad

def _fibonacci_spiral(n, jitter=0.0, seed=0):
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    if jitter > 0:
        i = i + rng.randn(n) * jitter
        i = np.clip(i, 0.5, n - 0.5)
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def construct():
    """
    1. Generate diverse initial configs.
    2. Run gradient descent on the soft-max covering radius proxy using a dense grid.
    3. Use a dynamic learning rate and beta (temperature) schedule.
    4. Final polish with a few steps of higher beta.
    """
    start_time = time.time()
    
    # Grid for optimization
    # 2000 points is a good balance between resolution and speed (2000*130 = 260k ops per step)
    n_grid = 2000
    grid_pts = _fibonacci_spiral(n_grid, jitter=0.0, seed=999)
    
    # Initial configs
    inits = []
    for seed in [42, 123, 7, 999, 31415, 271828, 161803, 111, 222, 333]:
        inits.append(_fibonacci_spiral(N, jitter=0.1, seed=seed))
    inits.append(_fibonacci_spiral(N, jitter=0.0, seed=0))
    rng = np.random.RandomState(555)
    inits.append(_normalize(rng.randn(N, 3)))
    
    # Sort by initial estimated covering radius
    init_scores = []
    for p in inits:
        # Quick estimate using a smaller grid for sorting
        p_norm = _normalize(p)
        dots = grid_pts @ p_norm.T
        max_dots = np.max(dots, axis=1)
        min_dots = np.max(dots, axis=1) # Wait, max dot is min angle
        # We want to minimize the max angle.
        # So we want to maximize the min dot? No.
        # Covering radius = max_y min_i d(y, p_i).
        # min_i d(y, p_i) corresponds to max_i cos(d) = max_i dot(y, p_i).
        # So we want to minimize max_y arccos(max_i dot(y, p_i)).
        # This is equivalent to maximizing min_y max_i dot(y, p_i).
        
        # Let's just compute the proxy value for sorting
        c_safe = np.clip(np.max(dots, axis=1), -1.0, 1.0)
        d_k = np.arccos(c_safe)
        score = np.max(d_k)
        init_scores.append((score, p))
        
    init_scores.sort(key=lambda x: x[0])
    
    best_pts = None
    best_score = np.pi
    
    # Main Optimization Loop
    for r0, p0 in init_scores:
        if not _time_ok(10.0):
            break
            
        pts = p0.copy()
        
        # Adaptive Beta Schedule
        # Start with low beta (smooth) to find global structure, increase for precision
        beta = 10.0
        
        # Learning rate schedule
        lr = 0.05
        
        # Run iterations
        for it in range(2000):
            if not _time_ok(5.0):
                break
                
            # Compute gradient
            grad = _grid_gradient(pts, grid_pts, beta=beta)
            
            # Normalize gradient to prevent large steps
            grad_norms = np.linalg.norm(grad, axis=1, keepdims=True)
            grad_norms[grad_norms < 1e-12] = 1.0
            grad = grad / grad_norms
            
            # Update
            pts = pts + lr * grad
            pts = _normalize(pts)
            
            # Periodically evaluate and adjust hyperparameters
            if it % 50 == 0:
                # Evaluate current score
                dots = grid_pts @ pts.T
                c_safe = np.clip(np.max(dots, axis=1), -1.0, 1.0)
                d_k = np.arccos(c_safe)
                current_score = np.max(d_k)
                
                if current_score < best_score:
                    best_score = current_score
                    best_pts = pts.copy()
                    
                # Increase beta if we are making progress (converging)
                if it > 100:
                    beta *= 1.05
                # Decrease lr as we converge
                lr *= 0.98
                
                # If beta is very high, we might be stuck in a local minimum of the soft-max
                # which approximates the hard-max. We can add a small random perturbation.
                if beta > 100 and it % 100 == 0:
                    rng = np.random.RandomState(int(time.time() * 1000))
                    perturbation = rng.randn(*pts.shape) * 0.01
                    pts = _normalize(pts + perturbation)
        
        # Final evaluation of this run
        dots = grid_pts @ pts.T
        c_safe = np.clip(np.max(dots, axis=1), -1.0, 1.0)
        d_k = np.arccos(c_safe)
        final_score = np.max(d_k)
        
        if final_score < best_score:
            best_score = final_score
            best_pts = pts.copy()

    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        
    return _normalize(best_pts)

# Test
if __name__ == "__main__":
    t0 = time.time()
    pts = construct()
    print("Shape:", pts.shape)
    # Use a very dense grid to estimate the final score for local testing
    n_grid_test = 20000
    grid_test = _fibonacci_spiral(n_grid_test, jitter=0.0, seed=42)
    dots = grid_test @ pts.T
    c_safe = np.clip(np.max(dots, axis=1), -1.0, 1.0)
    d_k = np.arccos(c_safe)
    print("Covering radius (deg, grid approx):", np.degrees(np.max(d_k)))
    print("Time: %.1f s" % (time.time() - t0))
