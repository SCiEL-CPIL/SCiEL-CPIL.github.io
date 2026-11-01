import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """Calculate squared pairwise distances for N points in D dimensions."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Calculate the actual score: (max_dist / min_dist)^2."""
    sq_dists = get_pairwise_sq_dists(points)
    min_sq = np.min(sq_dists)
    if min_sq < 1e-12:
        return np.inf
    max_sq = np.max(sq_dists)
    return (max_sq / min_sq)

def objective_smooth(x, tau=100.0):
    """
    Smoothed objective function using log-sum-exp for max and min.
    Minimizes log(max_sq_dist) - log(min_sq_dist).
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    
    # Avoid log(0)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max of squared distances
    m_max = np.max(sq_dists)
    log_sum_exp_max = m_max + np.log(np.sum(np.exp(tau * (sq_dists - m_max)))) / tau
    
    # Soft min of squared distances
    m_min = np.min(sq_dists)
    log_sum_exp_min = m_min - np.log(np.sum(np.exp(-tau * (sq_dists - m_min)))) / tau
    
    # Objective is log(max) - log(min)
    return np.log(log_sum_exp_max) - np.log(log_sum_exp_min)

def fibonacci_sphere(n):
    """Generate n points on a sphere using the Fibonacci sphere method."""
    points = []
    golden_angle = np.pi * (3. - np.sqrt(5.))
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2
        radius = np.sqrt(1 - y**2)
        theta = golden_angle * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s limit
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()
    
    rng = np.random.default_rng(42)
    
    # --- Strategy 1: Coordinate-space optimization with multiple initializations ---
    
    # Initial candidates
    # 1. Fibonacci sphere
    fib = fibonacci_sphere(N)
    coord_candidates = [fib.flatten()]
    
    # 2. Random points on a sphere
    for i in range(10):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        coord_candidates.append(pts.flatten())
        
    # 3. Random points in a cube
    for i in range(10):
        pts = rng.uniform(-1, 1, size=(N, D))
        coord_candidates.append(pts.flatten())

    # 4. Icosahedron + 2 extra points
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    extra = np.array([[0, 0, 1.5], [0, 0, -1.5]])
    ico_extra = np.vstack([ico, extra])
    coord_candidates.append(ico_extra.flatten())
    
    # 5. Cube vertices + center + 2 extra
    cube = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]], dtype=float)
    center = np.array([[0, 0, 0]])
    extra2 = rng.normal(size=(2, D)) * 0.5
    cube_pts = np.vstack([cube, center, extra2])
    coord_candidates.append(cube_pts.flatten())
    
    # Coordinate-space optimization with multi-stage tau
    n_coord_restarts = 40
    for i in range(n_coord_restarts):
        if time.time() - start_time > timeout - 20:
            break
            
        if i < len(coord_candidates):
            x0 = coord_candidates[i]
        else:
            x0 = rng.uniform(-1, 1, size=(N * D))
            
        try:
            # Multi-stage optimization with increasing tau
            current_x = x0
            
            for tau in [5.0, 20.0, 100.0, 500.0]:
                if time.time() - start_time > timeout - 10:
                    break
                    
                res = minimize(
                    objective_smooth, 
                    current_x, 
                    method='L-BFGS-B', 
                    args=(tau,),
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.x is not None:
                    current_x = res.x
                    pts = current_x.reshape(N, D)
                    update_best(pts)
                    
        except Exception:
            pass

    # --- Final polish: use the best points found and do a high-precision coordinate optimization ---
    if best_points is not None and time.time() - start_time < timeout - 2:
        try:
            res_final = minimize(
                objective_smooth, 
                best_points.flatten(), 
                method='L-BFGS-B', 
                args=(1000.0,),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res_final.x is not None:
                pts_final = res_final.x.reshape(N, D)
                update_best(pts_final)
        except Exception:
            pass

    if best_points is None:
        # Fallback
        best_points = fib
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
