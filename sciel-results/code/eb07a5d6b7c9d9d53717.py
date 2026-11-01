import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 14
D = 3
M = N * (N - 1) // 2

# Precompute pair indices for efficiency
pair_indices = list(itertools.combinations(range(N), 2))
pair_i = np.array([p[0] for p in pair_indices])
pair_j = np.array([p[1] for p in pair_indices])

def _compute_sq_dists(X):
    """Compute squared pairwise distances for N points in D dimensions."""
    diff = X[pair_i] - X[pair_j]
    return np.sum(diff**2, axis=1)

def _raw_score(points):
    """Calculate the actual score: (max_dist / min_dist)^2."""
    sq_dists = _compute_sq_dists(points)
    min_sq = np.min(sq_dists)
    if min_sq < 1e-12:
        return np.inf
    max_sq = np.max(sq_dists)
    return (max_sq / min_sq)

def _smooth_obj_and_grad(x_flat, tau):
    """
    Smooth objective in log-space with analytic gradient.
    Minimizes log(soft_max(log d2)) - log(soft_min(log d2)).
    
    x_flat: (N*D,) flattened centered points
    tau: temperature parameter
    Returns: (objective, gradient)
    """
    X = x_flat.reshape(N, D)
    # Center of mass constraint
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    
    sq_dists = _compute_sq_dists(X_centered)
    # Ensure positive for log
    sq_dists_safe = np.maximum(sq_dists, 1e-15)
    log_d2 = np.log(sq_dists_safe)
    
    if tau <= 0:
        tau = 1e-10
    
    # Soft max in log-space
    max_log = np.max(log_d2)
    term_max = tau * (log_d2 - max_log)
    term_max = np.clip(term_max, -50, 50)
    sum_exp_max = np.sum(np.exp(term_max))
    soft_max = max_log + (1.0 / tau) * np.log(sum_exp_max)
    p_max = np.exp(term_max) / sum_exp_max
    
    # Soft min in log-space
    min_log = np.min(log_d2)
    term_min = -tau * (log_d2 - min_log)
    term_min = np.clip(term_min, -50, 50)
    sum_exp_min = np.sum(np.exp(term_min))
    soft_min = min_log - (1.0 / tau) * np.log(sum_exp_min)
    p_min = np.exp(term_min) / sum_exp_min
    
    obj = soft_max - soft_min
    
    # Gradient computation
    # d(log d2)/dX = 2*(Xi - Xj)/d2 for pair (i,j)
    diff = X_centered[pair_i] - X_centered[pair_j]
    
    # Gradient of soft_max w.r.t. X
    grad_soft_max = np.zeros_like(X_centered)
    # d(soft_max)/d(log_d2_k) = p_max[k]
    # d(log_d2_k)/dX = 2*diff_k / d2_k
    inv_d2 = 1.0 / sq_dists_safe
    weight_max = 2.0 * p_max * inv_d2
    
    np.add.at(grad_soft_max, pair_i, weight_max[:, np.newaxis] * diff)
    np.add.at(grad_soft_max, pair_j, -weight_max[:, np.newaxis] * diff)
    
    # Gradient of soft_min w.r.t. X
    grad_soft_min = np.zeros_like(X_centered)
    weight_min = 2.0 * p_min * inv_d2
    
    np.add.at(grad_soft_min, pair_i, weight_min[:, np.newaxis] * diff)
    np.add.at(grad_soft_min, pair_j, -weight_min[:, np.newaxis] * diff)
    
    grad = grad_soft_max - grad_soft_min
    
    # Apply centering: gradient should be projected to remove mean
    # The centering constraint means we only care about variations with zero mean
    grad_centered = grad - np.mean(grad, axis=0, keepdims=True)
    
    return obj, grad_centered.flatten()

def _fibonacci_sphere(n):
    """Generate n points on a sphere using the Fibonacci sphere method."""
    points = np.zeros((n, 3))
    golden_angle = np.pi * (3. - np.sqrt(5.))
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2
        radius = np.sqrt(1 - y**2)
        theta = golden_angle * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points[i] = [x, y, z]
    return points

def _normalize_scale(X):
    """Normalize so that mean squared distance is 1."""
    sq_dists = _compute_sq_dists(X)
    mean_sq = np.mean(sq_dists)
    if mean_sq > 1e-15:
        X = X * np.sqrt(1.0 / mean_sq)
    return X

def _center(X):
    """Center points at origin."""
    return X - np.mean(X, axis=0, keepdims=True)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, D):
            return
        if not np.all(np.isfinite(pts)):
            return
        score = _raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    coord_candidates = []
    
    # 1. Fibonacci sphere
    fib = _fibonacci_sphere(N)
    coord_candidates.append(_center(fib))
    
    # 2. Fibonacci sphere with noise
    for scale in [0.05, 0.1, 0.2]:
        noisy_fib = fib + rng.normal(scale=scale, size=(N, D))
        coord_candidates.append(_center(noisy_fib))
    
    # 3. Random points on sphere with varying radii
    for i in range(15):
        pts = rng.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        # Vary radii to create different structures
        radii = rng.uniform(0.3, 1.0, size=(N, 1))
        pts = pts * radii
        coord_candidates.append(_center(pts))
    
    # 4. Random points in cube
    for i in range(10):
        pts = rng.uniform(-1, 1, size=(N, D))
        coord_candidates.append(_center(pts))
    
    # 5. Icosahedron + 2 extra points
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    extra = rng.normal(size=(2, D)) * 0.3
    ico_extra = np.vstack([ico, extra])
    coord_candidates.append(_center(ico_extra))
    
    # 6. Cube vertices + center + 2 extra
    cube = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]], dtype=float)
    center_pt = np.array([[0, 0, 0]])
    extra2 = rng.normal(size=(2, D)) * 0.5
    cube_pts = np.vstack([cube, center_pt, extra2])
    coord_candidates.append(_center(cube_pts))
    
    # Normalize all candidates
    for i in range(len(coord_candidates)):
        coord_candidates[i] = _normalize_scale(coord_candidates[i])
    
    # Coordinate-space optimization with multi-stage tau and analytic gradients
    n_coord_restarts = 80
    for i in range(n_coord_restarts):
        if time.time() - start_time > deadline - 15:
            break
            
        if i < len(coord_candidates):
            x0 = coord_candidates[i].flatten()
        else:
            # Random restart
            x0 = rng.uniform(-1, 1, size=(N * D))
            x0 = x0.reshape(N, D)
            x0 = _center(x0)
            x0 = _normalize_scale(x0)
            x0 = x0.flatten()
        
        try:
            current_x = x0
            
            # Multi-stage optimization with increasing tau
            for tau in [5.0, 20.0, 100.0, 500.0, 2000.0]:
                if time.time() - start_time > deadline - 10:
                    break
                    
                res = minimize(
                    _smooth_obj_and_grad, 
                    current_x, 
                    args=(tau,),
                    jac=True,
                    method='L-BFGS-B', 
                    options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.x is not None and np.all(np.isfinite(res.x)):
                    current_x = res.x
                    pts = current_x.reshape(N, D)
                    pts = _center(pts)
                    update_best(pts)
                    
        except Exception:
            pass

    # Final polish: high-precision optimization on best points
    if best_points is not None and time.time() - start_time < deadline - 3:
        try:
            x_final = _normalize_scale(_center(best_points)).flatten()
            
            # High-tau polish
            for tau in [5000.0, 10000.0]:
                if time.time() - start_time > deadline - 2:
                    break
                    
                res_final = minimize(
                    _smooth_obj_and_grad, 
                    x_final, 
                    args=(tau,),
                    jac=True,
                    method='L-BFGS-B', 
                    options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res_final.x is not None and np.all(np.isfinite(res_final.x)):
                    x_final = res_final.x
                    pts_final = x_final.reshape(N, D)
                    pts_final = _center(pts_final)
                    update_best(pts_final)
                    
        except Exception:
            pass

    # SLSQP polish on exact objective
    if best_points is not None and time.time() - start_time < deadline - 1:
        try:
            def exact_obj(x_flat):
                X = x_flat.reshape(N, D)
                X = _center(X)
                return _raw_score(X)
            
            x_slqp = _normalize_scale(_center(best_points)).flatten()
            res_slqp = minimize(
                exact_obj, 
                x_slqp, 
                method='SLSQP', 
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if res_slqp.x is not None and np.all(np.isfinite(res_slqp.x)):
                pts_slqp = res_slqp.x.reshape(N, D)
                pts_slqp = _center(pts_slqp)
                update_best(pts_slqp)
        except Exception:
            pass

    if best_points is None:
        # Fallback
        best_points = _center(_normalize_scale(fib))
        best_score = _raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = _raw_score(pts)
    print(f"Score: {score}")
