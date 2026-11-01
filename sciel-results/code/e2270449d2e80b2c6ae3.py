import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(rng):
    """Generate random initial points."""
    pts = rng.standard_normal((N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _get_grid(res=200):
    """
    Generate a dense spherical grid for the surrogate objective.
    Uses a Fibonacci spiral for uniform distribution.
    res controls the number of grid points (res^2 approx).
    """
    # We want roughly res^2 points. Let's just use a fixed high number for consistency.
    # 10,000 to 20,000 points is usually fast enough for matrix ops.
    n_grid = res * res 
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _surrogate_objective_and_grad(pts, grid, temperature=0.05):
    """
    Compute a smooth surrogate for the covering radius:
    f(pts) = (1/|G|) * sum_{g in G} max_{p in P} (arccos(g.p))^2
    Approximated by soft-max:
    M(g) = -temperature * log(sum_p exp(-dist(g, p)^2 / temperature))
    Objective = mean(M(g))
    
    Gradient is computed analytically.
    """
    # dist^2 = 2 * (1 - dot)
    dots = pts @ grid.T  # (N, G)
    dots = np.clip(dots, -1.0, 1.0)
    d2 = 2.0 * (1.0 - dots)  # (N, G)
    
    # Soft max of d2 (we want to minimize the max distance, so we look at large d2)
    # Standard softmax for max: max(x) ~ -T log(sum exp(-x/T))
    # Here x = d2.
    # To avoid overflow, shift by max per column (grid point)
    max_d2 = np.max(d2, axis=0, keepdims=True)
    exp_terms = np.exp(-(d2 - max_d2) / temperature)
    sum_exp = np.sum(exp_terms, axis=0, keepdims=True)
    
    # M_g = max_d2 + T * log(sum_exp) ... wait.
    # Let S = sum exp(-d2/T). 
    # soft_max = -T log(S).
    # d2 - max_d2 is <= 0.
    # exp(-(d2 - max_d2)/T) = exp(max_d2/T) * exp(-d2/T).
    # S = exp(max_d2/T) * sum_exp.
    # -T log(S) = -T (max_d2/T + log(sum_exp)) = -max_d2 - T log(sum_exp).
    # This is the *negative* of the distance if we treat distance as positive.
    # We want the "effective distance" D_g such that D_g approx max_p d2_gp.
    # D_g = max_d2 + T * log(sum_exp) is NOT correct sign-wise for "max".
    # Let's re-derive.
    # We want approx max(d2).
    # Let f(x) = T log(sum exp(x_i/T)). Then f(x) approx max(x_i).
    # So D_g = T * log(sum exp(d2_gp / T)).
    
    # Let's restart the soft-max calculation for max(d2).
    # d2 values are positive.
    max_d2 = np.max(d2, axis=0, keepdims=True)
    exp_terms = np.exp((d2 - max_d2) / temperature)
    sum_exp = np.sum(exp_terms, axis=0, keepdims=True)
    
    # D_g = max_d2 + T * log(sum_exp)
    D = max_d2[0] + temperature * np.log(sum_exp[0])
    
    # Objective is mean(D^2) or mean(D)? 
    # Covering radius is max angular dist. Angular dist = arccos(dot) = sqrt(2(1-dot)) approx.
    # Minimizing max d2 is equivalent to minimizing max dist.
    # Let's minimize mean(D^2) to penalize outliers more, or just mean(D).
    # The task asks for "largest angular distance". 
    # Using sum of squares of nearest neighbor distances is a common surrogate.
    # Let's use mean(D^2).
    
    obj = np.mean(D**2)
    
    # Gradient w.r.t. pts
    # D_g = max_d2_g + T log(sum_p exp((d2_gp - max_d2_g)/T))
    # This is complex to differentiate directly due to max_d2.
    # Alternative: Use the standard softmax attention gradient.
    # Let weights w_gp = exp(d2_gp / T) / sum_k exp(d2_gk / T).
    # D_g = sum_p w_gp * d2_gp. (This is the soft-max approximation).
    # Actually, T log(sum exp(x/T)) is the log-sum-exp, which approximates max.
    # The gradient of LSE(x)/T w.r.t x_i is softmax(x)_i.
    # So d(D_g)/d(d2_gp) = softmax(d2_g / T)_p.
    
    # Let's compute softmax weights.
    # To be numerically stable:
    # w_gp = exp((d2_gp - max_d2_g)/T) / sum_k exp((d2_gk - max_d2_g)/T)
    # We already have exp_terms and sum_exp.
    w = exp_terms / sum_exp  # (N, G)
    
    # d(D_g)/d(d2_gp) = w_gp
    # d(d2_gp)/d(pts_p) = d(2(1 - pts_p . g_g))/d(pts_p) = -2 * g_g
    # So d(D_g)/d(pts_p) = sum_g w_gp * (-2 * g_g) = -2 * (w[:, g].T @ g_g)? No.
    # Sum over g: d(D_g)/d(pts_p) = -2 * sum_g w_gp * g_g
    
    # grad_obj = d(mean(D^2))/d(pts) = (1/G) * sum_g 2 * D_g * d(D_g)/d(pts)
    # = (2/G) * sum_g D_g * (-2 * sum_g' w_gp g_g') ... indices mixed.
    
    # Let's do it with matrix multiplication.
    # grad_pts_p = (2/G) * sum_g D_g * (-2 * sum_k w_pk g_k)
    # Let M = w @ grid.T  -> (N, 3). M[p, :] = sum_g w_pg g_g.
    # Then d(D_g)/d(pts_p) = -2 * M[p, :]
    # grad_pts_p = (2/G) * sum_g D_g * (-2 * M[p, :])
    #             = (-4/G) * M[p, :] * (sum_g D_g * w_pg)
    
    # Let S_p = sum_g D_g * w_pg. (N,)
    S = w.T @ D  # (N,)
    
    M = w @ grid.T  # (N, 3)
    
    # grad_pts = (-4 / G) * M * S[:, np.newaxis]
    grad_pts = (-4.0 / len(grid)) * M * S[:, np.newaxis]
    
    return obj, grad_pts

def _hull_radius(pts):
    """Compute the true covering radius using convex hull."""
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.pi
            
        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        return np.max(radii)
    except Exception:
        return np.pi

def _lloyd_relax(pts, grid, n_iters=50, deadline=None):
    """
    Lloyd's algorithm: Move each point to the centroid of its Voronoi cell.
    Voronoi cell on sphere approximated by assigning grid points to nearest pt.
    """
    for i in range(n_iters):
        if deadline and time.time() > deadline:
            break
            
        # Assign each grid point to nearest pt
        dots = pts @ grid.T  # (N, G)
        nearest = np.argmax(dots, axis=0)  # (G,)
        
        # Compute centroid for each pt
        new_pts = np.zeros_like(pts)
        for k in range(N):
            mask = nearest == k
            if np.any(mask):
                # Spherical centroid: normalize sum of vectors
                c = np.sum(grid[mask], axis=0)
                norm = np.linalg.norm(c)
                if norm > _EPS:
                    new_pts[k] = c / norm
                else:
                    new_pts[k] = pts[k]
            else:
                new_pts[k] = pts[k]
                
        # Normalize
        new_pts /= (np.linalg.norm(new_pts, axis=1, keepdims=True) + _EPS)
        
        # Check for collapse
        if np.allclose(new_pts, pts, atol=1e-6):
            break
            
        pts = new_pts
        
    return pts

def construct():
    deadline = time.time() + 145.0  # Safety margin
    rng = np.random.default_rng(42)
    
    # 1. Initial points
    best_pts = _initial_fib()
    
    # 2. Smooth Surrogate Optimization (Gradient Descent)
    # Use a grid of ~4000 points for speed during gradient steps
    grid_opt = _get_grid(63) # ~4000 points
    
    # Gradient Descent
    pts = best_pts.copy()
    lr = 0.01
    for step in range(200):
        if time.time() > deadline - 30.0:
            break
            
        obj, grad = _surrogate_objective_and_grad(pts, grid_opt, temperature=0.05)
        
        # Normalize gradient to prevent explosion
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad = grad / (grad_norm + _EPS)
        
        cand = pts + lr * grad
        cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + _EPS)
        
        # Simple acceptance: always accept in smooth phase, or check improvement
        # Since it's a surrogate, just take the step
        pts = cand
        
        # Adaptive learning rate
        if step % 20 == 19:
            lr *= 0.9
            
    # 3. Lloyd's Relaxation (Voronoi)
    # Use a denser grid for better Voronoi approximation
    grid_lloyd = _get_grid(100) # ~10000 points
    local_deadline = min(deadline - 10.0, time.time() + 60.0)
    pts = _lloyd_relax(pts, grid_lloyd, n_iters=100, deadline=local_deadline)
    
    # 4. Final Polish with High-Resolution Grid
    # Use an even denser grid for the final few iterations to fix small gaps
    if time.time() < deadline - 5.0:
        grid_polish = _get_grid(150) # ~22500 points
        # Run a few more Lloyd iterations
        pts = _lloyd_relax(pts, grid_polish, n_iters=20, deadline=deadline - 2.0)
        
    # Final normalization
    best_pts = np.array(pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
