import numpy as np
import time
from scipy.optimize import minimize, basinhopping

N = 14
D = 3

# Precompute pairwise indices for efficiency
_pair_i, _pair_j = np.triu_indices(N, k=1)
_n_pairs = N * (N - 1) // 2

def get_pairwise_sq_dists(points):
    """
    Compute squared pairwise distances for N points in D dimensions.
    Returns a flat array of length N*(N-1)/2.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    return sq_dists[_pair_i, _pair_j]

def raw_score(points):
    """
    Calculate the actual score: (max_dist / min_dist)^2.
    """
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-12:
        return np.inf
    return (np.max(sq_dists) / np.min(sq_dists))

def generate_D3d_points(params):
    """
    Generate 14 points with D3d symmetry.
    params: [x, y, z, a]
    """
    x, y, z, a = params
    
    c = 0.5
    s = np.sqrt(3.0) / 2.0
    
    # C3 orbit of (x, y, z)
    p1 = np.array([x, y, z])
    p2 = np.array([-c*x - s*y, s*x - c*y, z])
    p3 = np.array([-c*x + s*y, -s*x - c*y, z])
    
    # C2' orbit (rotation by 180 around x-axis)
    q1 = np.array([x, -y, -z])
    q2 = np.array([-c*x - s*y, -s*x + c*y, -z])
    q3 = np.array([-c*x + s*y, s*x + c*y, -z])
    
    # Inversion orbit
    r1 = -p1
    r2 = -p2
    r3 = -p3
    s1 = -q1
    s2 = -q2
    s3 = -q3
    
    # Axis points
    t1 = np.array([0.0, 0.0, a])
    t2 = np.array([0.0, 0.0, -a])
    
    points = np.array([p1, p2, p3, q1, q2, q3, r1, r2, r3, s1, s2, s3, t1, t2])
    return points

def objective_D3d(params, tau=50.0):
    """
    Objective function for D3d symmetric configuration.
    Minimizes log(max_sq_dist) - log(min_sq_dist) with soft approximation.
    """
    points = generate_D3d_points(params)
    
    # Center points
    points = points - np.mean(points, axis=0, keepdims=True)
    
    sq_dists = get_pairwise_sq_dists(points)
    mean_sq_dist = np.mean(sq_dists)
    if mean_sq_dist < 1e-12:
        return 1e10
    
    # Normalize
    points = points / np.sqrt(mean_sq_dist)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    log_sq_dists = np.log(sq_dists)
    
    # Soft max/min
    m_max = np.max(log_sq_dists)
    exp_max = np.exp(tau * (log_sq_dists - m_max))
    soft_max = m_max + np.log(np.sum(exp_max)) / tau
    
    m_min = np.min(log_sq_dists)
    exp_min = np.exp(-tau * (log_sq_dists - m_min))
    soft_min = m_min - np.log(np.sum(exp_min)) / tau
    
    return soft_max - soft_min

def construct():
    start_time = time.time()
    timeout = 120.0  # Safety margin
    
    best_score = np.inf
    best_params = None
    
    def update_best(params):
        nonlocal best_score, best_params
        if params is None:
            return
        pts = generate_D3d_points(params)
        # Check for duplicates
        sq_dists = get_pairwise_sq_dists(pts)
        if np.min(sq_dists) < 1e-10:
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_params = params.copy()
    
    rng = np.random.default_rng(42)
    
    # Initial candidates for (x, y, z, a)
    # x, y, z define the generic orbit, a defines the axis points
    # We want x, y not both 0.
    inits = [
        [1.0, 0.5, 0.5, 0.5],
        [1.0, 0.2, 0.8, 0.3],
        [0.8, 0.8, 0.2, 0.6],
        [1.0, 0.0, 1.0, 0.0], # Degenerate? x!=0 so ok.
        [0.5, 0.5, 0.5, 1.0]
    ]
    
    for init in inits:
        if time.time() - start_time > timeout - 20:
            break
        update_best(init)
        
        # Local optimization from this start
        try:
            res = minimize(
                objective_D3d, 
                init, 
                method='L-BFGS-B', 
                args=(50.0,),
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.x is not None:
                update_best(res.x)
        except Exception:
            pass
        
    # Basin Hopping on the 4D parameter space
    if best_params is None:
        best_params = np.array([1.0, 0.5, 0.5, 0.5])
        
    def bh_objective(params):
        return objective_D3d(params, tau=20.0)
    
    bh_minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'options': {'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
    }
    
    try:
        bh_res = basinhopping(
            bh_objective,
            best_params,
            minimizer_kwargs=bh_minimizer_kwargs,
            stepsize=0.5,
            niter=100, # More iterations since dim is low
            seed=42
        )
        if bh_res.x is not None:
            update_best(bh_res.x)
    except Exception:
        pass
        
    # Final polish with higher tau
    if time.time() - start_time < timeout - 5:
        try:
            res = minimize(
                objective_D3d, 
                best_params, 
                method='L-BFGS-B', 
                args=(500.0,),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                update_best(res.x)
        except Exception:
            pass
            
    # Exact objective polish
    if time.time() - start_time < timeout - 2:
        def exact_obj(params):
            pts = generate_D3d_points(params)
            sq_dists = get_pairwise_sq_dists(pts)
            min_sq = np.min(sq_dists)
            if min_sq < 1e-12:
                return 1e10
            return np.max(sq_dists) / min_sq
        
        try:
            res = minimize(
                exact_obj, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res.x is not None:
                update_best(res.x)
        except Exception:
            pass

    if best_params is None:
        best_params = np.array([1.0, 0.5, 0.5, 0.5])

    return generate_D3d_points(best_params)

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
