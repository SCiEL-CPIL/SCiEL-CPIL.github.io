import numpy as np
import time
from scipy.optimize import minimize, basinhopping
from functools import partial

N = 14
D = 3
NUM_PAIRS = N * (N - 1) // 2

def get_pairwise_sq_dists(points):
    """
    Returns a flat array of squared pairwise distances for N points.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Compute the true (non-smooth) score: max/min squared distance."""
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-15:
        return np.inf
    return np.max(sq_dists) / np.min(sq_dists)

def normalize_points(points):
    """Center and scale points to have unit mean squared norm."""
    pts = points - np.mean(points, axis=0)
    norm = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
    if norm < 1e-15:
        return pts
    return pts / norm

def objective_smooth(x, tau=10.0):
    """
    Smooth log-ratio objective using soft-max and soft-min.
    x is flattened array of shape (N*D,)
    """
    points = x.reshape(N, D)
    # Center and normalize to handle invariances during optimization
    # This helps the optimizer stay in a well-conditioned region
    points = points - np.mean(points, axis=0, keepdims=True)
    mean_sq_norm = np.mean(np.sum(points**2, axis=1))
    if mean_sq_norm < 1e-15:
        return 1e10
    points = points / np.sqrt(mean_sq_norm)
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Soft max and soft min via log-sum-exp on the squared distances
    # Note: Minimizing log(max) - log(min) is equivalent to minimizing log(max/min)
    # which is equivalent to minimizing max/min
    m_max = np.max(sq_dists)
    m_min = np.min(sq_dists)
    
    # Soft Max
    exp_max_terms = np.exp(tau * (sq_dists - m_max))
    soft_max = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    # Soft Min
    exp_min_terms = np.exp(-tau * (sq_dists - m_min))
    soft_min = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    soft_min = np.maximum(soft_min, 1e-15)
    
    return np.log(soft_max) - np.log(soft_min)

def fibonacci_sphere(n_points):
    """Generate n_points on a sphere using Fibonacci lattice."""
    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_points):
        y = 1.0 - (i / (n_points - 1)) * 2.0
        radius = np.sqrt(1.0 - y**2)
        theta = golden_angle * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

def icosahedron_with_extras(n_points, rng):
    """
    Generate points based on an icosahedron structure plus random extras.
    Icosahedron has 12 vertices. We need 14.
    """
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Add 2 random points
    p1 = rng.normal(size=D)
    p1 = p1 / np.linalg.norm(p1)
    p2 = rng.normal(size=D)
    p2 = p2 / np.linalg.norm(p2)
    
    pts = np.vstack([ico, p1, p2])
    return pts

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s kill
    
    best_score = np.inf
    best_points = None
    
    rng = np.random.default_rng(12345)
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        if not np.all(np.isfinite(pts)):
            return
        sq_dists = get_pairwise_sq_dists(pts)
        if np.min(sq_dists) < 1e-10:
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    # 1. Generate diverse initial candidates
    candidates = []
    
    # Fibonacci sphere
    fib_pts = fibonacci_sphere(N)
    candidates.append(fib_pts)
    
    # Icosahedron based
    for i in range(3):
        ico_pts = icosahedron_with_extras(N, rng)
        candidates.append(ico_pts)
        
    # Random sphere points
    for i in range(5):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts)
        
    # Random ball points
    for i in range(5):
        pts = rng.normal(size=(N, D)) * 2.0
        candidates.append(pts)
        
    # Normalize all candidates
    normalized_candidates = [normalize_points(c) for c in candidates]
    
    # Evaluate initial candidates
    for c in normalized_candidates:
        update_best(c)
        
    if best_points is None:
        # Fallback
        return normalize_candidates[0] if normalized_candidates else np.zeros((N, D))
        
    # 2. Basin Hopping for Global Optimization
    # Use the best candidate found so far as the starting point for BH
    x0 = best_points.flatten()
    
    # Define the objective for Basin Hopping (using a moderate tau)
    tau_bh = 20.0
    def bh_objective(x):
        return objective_smooth(x, tau=tau_bh)
        
    bh_minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'options': {'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
    }
    
    bh_best_x = x0.copy()
    bh_best_f = bh_objective(x0)
    
    def bh_callback(x, f, accept):
        nonlocal bh_best_x, bh_best_f
        if f < bh_best_f:
            bh_best_f = f
            bh_best_x = x.copy()
        # Time check
        if time.time() - start_time > timeout - 40:
            raise Exception("Time to stop Basin Hopping")
            
    try:
        bh_res = basinhopping(
            bh_objective,
            x0,
            minimizer_kwargs=bh_minimizer_kwargs,
            stepsize=0.5,
            niter=100, # Will be cut by time
            callback=bh_callback,
            seed=42
        )
        if bh_res.x is not None:
            update_best(bh_res.x.reshape(N, D))
            bh_best_x = bh_res.x
    except Exception:
        # Use best found during BH
        if bh_best_x is not None:
            update_best(bh_best_x.reshape(N, D))
            
    # 3. Local Refinement with Multi-Stage Tau
    # Take the best point found and polish it
    x_current = best_points.flatten()
    
    tau_schedule = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
    
    for tau in tau_schedule:
        if time.time() - start_time > timeout - 20:
            break
        try:
            res = minimize(
                objective_smooth, 
                x_current, 
                method='L-BFGS-B', 
                args=(tau,),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                x_current = res.x
        except Exception:
            pass

    # 4. Final Exact Local Search using SLSQP on high-tau smooth objective
    # Using a very high tau makes the smooth objective very close to the true one
    if time.time() - start_time < timeout - 5:
        try:
            res = minimize(
                objective_smooth, 
                x_current, 
                method='SLSQP',
                args=(50000.0,),
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
        except Exception:
            pass

    # 5. Try a few more random restarts if time remains
    # This helps if Basin Hopping got stuck in a bad basin
    restart_count = 0
    while time.time() - start_time < timeout - 5:
        x0_rand = rng.uniform(-1, 1, size=(N * D))
        # Quick local opt
        try:
            res = minimize(
                objective_smooth, 
                x0_rand, 
                method='L-BFGS-B', 
                args=(50.0,),
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                
                # If this is good, do a high-precision polish
                if raw_score(pts) < best_score * 1.05 and time.time() - start_time < timeout - 3:
                    try:
                        res2 = minimize(
                            objective_smooth, 
                            res.x, 
                            method='L-BFGS-B', 
                            args=(1000.0,),
                            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                        )
                        if res2.x is not None:
                            pts2 = res2.x.reshape(N, D)
                            update_best(pts2)
                    except Exception:
                        pass
        except Exception:
            pass
        restart_count += 1
        if restart_count > 10: break # Cap restarts

    if best_points is None:
        return np.zeros((N, D))
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    sq_dists = get_pairwise_sq_dists(pts)
    score = (np.max(sq_dists) / np.min(sq_dists))
    print(f"Score: {score}")
