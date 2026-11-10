import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes (d_min / d_max)^2 for 14 points in 3D.
    
    Strategy:
    1. Initialize using Fekete points (maximize product of distances) to ensure a 
       well-distributed starting configuration.
    2. Use Simulated Annealing (SA) to optimize the configuration.
       - Objective: Maximize d_min subject to d_max <= 1.
       - To handle the non-smooth min/max and constraints, we use a penalty-based 
         objective for SA: 
           f(x) = d_min - lambda * max(0, d_max - 1)^2
         This encourages increasing the minimum distance while penalizing 
         any maximum distance exceeding 1.
       - The temperature schedule allows escaping local minima where a few 
         pairs are too close, by occasionally accepting worse configurations 
         to explore the space.
    3. Polish the best result found by SA using BFGS on a smoothed 
       log-barrier objective to fine-tune the coordinates.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 600  # Safety margin

    def get_score(pts):
        """Compute the actual metric: (d_min / d_max)^2"""
        dists = pdist(pts, metric='sqeuclidean')
        if len(dists) == 0:
            return 0.0
        d_min_sq = np.min(dists)
        d_max_sq = np.max(dists)
        if d_max_sq < 1e-12:
            return 0.0
        return (d_min_sq / d_max_sq)

    def sa_objective(pts, lam=10.0):
        """
        Objective for Simulated Annealing.
        We want to maximize d_min, but keep d_max close to 1 (or <= 1).
        Since SA minimizes, we return negative of the desired value.
        We use a penalty if d_max > 1.
        """
        dists = pdist(pts, metric='sqeuclidean')
        if len(dists) == 0:
            return 0.0
        d_min_sq = np.min(dists)
        d_max_sq = np.max(dists)
        
        # Penalty for exceeding max distance
        penalty = 0.0
        if d_max_sq > 1.0:
            penalty = lam * (d_max_sq - 1.0)**2
        
        # We want to maximize d_min_sq - penalty
        # So we minimize -(d_min_sq - penalty)
        return -(d_min_sq - penalty)

    def polish_objective(pts, tau=0.01):
        """
        Smoothed objective for BFGS polishing.
        Maximize softmin(d^2) / softmax(d^2) with a barrier on d_max.
        """
        dists = pdist(pts, metric='sqeuclidean')
        if len(dists) == 0:
            return 0.0
        
        # Log-sum-exp for stability
        m_min = np.min(dists)
        softmin_val = m_min - tau * np.log(np.sum(np.exp(-(dists - m_min) / tau)))
        
        m_max = np.max(dists)
        softmax_val = m_max + tau * np.log(np.sum(np.exp((dists - m_max) / tau)))
        
        if softmax_val < 1e-12:
            return 0.0
            
        ratio = softmin_val / softmax_val
        
        # Barrier to keep d_max <= 1 (actually, since it's scale invariant, 
        # we just want the ratio. But adding a slight barrier helps convergence 
        # to a specific scale if needed, though not strictly necessary for ratio).
        # Let's stick to just the ratio for the polish, as scaling is free.
        return -ratio

    def normalize_points(pts):
        """Center and scale such that d_max = 1."""
        pts = pts - np.mean(pts, axis=0)
        dists = pdist(pts, metric='sqeuclidean')
        d_max_sq = np.max(dists)
        if d_max_sq > 1e-12:
            pts = pts / np.sqrt(d_max_sq)
        return pts

    def generate_fekete_init():
        """
        Generate initial points by maximizing the product of distances (Fekete points).
        This is done via a few iterations of gradient ascent on log(product) = sum(log(d)).
        """
        # Start with random points on sphere
        pts = np.random.randn(n, d)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        def fekete_obj(x):
            p = x.reshape(n, d)
            dists = pdist(p, metric='sqeuclidean')
            # Avoid log(0)
            dists = np.maximum(dists, 1e-10)
            return -np.sum(np.log(dists))
        
        # Quick optimization to get a good Fekete-like distribution
        res = minimize(fekete_obj, pts.flatten(), method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        return res.x.reshape(n, d)

    # Main Loop
    best_pts = None
    best_score = 0.0
    
    # 1. Generate a good initial guess using Fekete points
    init_pts = generate_fekete_init()
    init_pts = normalize_points(init_pts)
    
    # 2. Simulated Annealing
    # Parameters
    T_start = 0.1
    T_end = 1e-5
    alpha = 0.995  # Cooling rate
    n_iter_per_temp = 50
    
    current_pts = init_pts.copy()
    current_obj = sa_objective(current_pts)
    best_sa_pts = current_pts.copy()
    best_sa_obj = current_obj
    
    T = T_start
    while T > T_end:
        for _ in range(n_iter_per_temp):
            if time.time() > deadline - 30:
                break
            
            # Perturb current points
            # Use a step size proportional to temperature
            step_size = T * 0.5
            noise = np.random.randn(n, d) * step_size
            new_pts = current_pts + noise
            
            # Normalize to keep scale consistent (d_max ~ 1)
            new_pts = normalize_points(new_pts)
            
            new_obj = sa_objective(new_pts)
            
            # Acceptance criterion
            delta_obj = new_obj - current_obj
            if delta_obj < 0 or np.random.rand() < np.exp(-delta_obj / T):
                current_pts = new_pts
                current_obj = new_obj
                
                if current_obj < best_sa_obj:
                    best_sa_obj = current_obj
                    best_sa_pts = current_pts.copy()
        
        T *= alpha
        
        if time.time() > deadline - 30:
            break

    # Evaluate best SA result
    best_score = get_score(best_sa_pts)
    best_pts = best_sa_pts.copy()

    # 3. Polish with BFGS using smoothed objective
    # We might do multiple polishing steps from slightly perturbed versions of best_sa_pts
    polish_starts = 5
    for i in range(polish_starts):
        if time.time() > deadline - 10:
            break
        
        # Start from best_sa_pts, maybe with tiny noise for i>0
        x0 = best_sa_pts.flatten()
        if i > 0:
            x0 += np.random.randn(n * d) * 1e-4
        
        res = minimize(
            polish_objective,
            x0,
            method='BFGS',
            options={'maxiter': 500, 'gtol': 1e-8}
        )
        
        if time.time() > deadline - 5:
            continue
            
        pts_polished = res.x.reshape(n, d)
        score_polished = get_score(pts_polished)
        
        if score_polished > best_score:
            best_score = score_polished
            best_pts = pts_polished.copy()

    # Final sanity check
    if best_pts is None:
        best_pts = init_pts
        
    # Ensure finite values
    if not np.all(np.isfinite(best_pts)):
        best_pts = init_pts
        
    return best_pts
