import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Returns 16 points in 2D that maximize the min/max distance ratio.
    Uses smooth soft-min/soft-max objective with adaptive beta, multiple initializations,
    L-BFGS-B optimization, and a final Nelder-Mead polish pass.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    n = 16
    m = n * (n - 1) // 2  # 120 pairwise distances
    
    # Precompute pairwise index pairs for vectorized distance computation
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def compute_dists(pts):
        """Compute all pairwise distances from points array."""
        pts = pts.reshape((n, 2))
        diff = pts[i_idx] - pts[j_idx]
        return np.sqrt(np.sum(diff ** 2, axis=1))
    
    def soft_min_max_ratio(pts, beta=100.0):
        """
        Compute soft min/soft max ratio of pairwise distances.
        Uses log-sum-exp style formulation.
        """
        dists = compute_dists(pts)
        if np.any(dists < 0):
            dists = np.abs(dists)
        
        d_max = np.max(dists)
        if d_max < 1e-15:
            return 0.0
        
        # Soft min: -logsumexp(-beta * d) / beta
        # Soft max: logsumexp(beta * d) / beta
        scaled_min = -beta * dists
        max_scaled_min = np.max(scaled_min)
        soft_min = -(max_scaled_min + np.log(np.sum(np.exp(scaled_min - max_scaled_min)))) / beta
        
        scaled_max = beta * dists
        max_scaled_max = np.max(scaled_max)
        soft_max = (max_scaled_max + np.log(np.sum(np.exp(scaled_max - max_scaled_max)))) / beta
        
        if soft_max < 1e-15:
            return 0.0
        
        return soft_min / soft_max
    
    def get_score(pts):
        dists = compute_dists(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2
    
    best_points = None
    best_score = -np.inf
    
    # Define diverse initial configurations
    
    # 1. 4x4 Grid
    x = np.arange(4)
    y = np.arange(4)
    xx, yy = np.meshgrid(x, y)
    grid_pts = np.column_stack((xx.ravel(), yy.ravel()))
    
    # 2. Hexagonal Lattice Subset
    u = np.array([1.0, 0.0])
    v = np.array([0.5, np.sqrt(3)/2])
    points_list = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * u + j * v
            dist_sq = np.dot(p, p)
            points_list.append((dist_sq, p))
    points_list.sort(key=lambda x: x[0])
    hex_pts = np.array([p[1] for p in points_list[:16]])
    
    # 3. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 1.0
    circle_pts = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    
    # 4. Random points (seeded for reproducibility)
    np.random.seed(42)
    rand_pts = np.random.randn(n, 2) * 2
    
    # 5. Two concentric circles (8+8)
    pts_cc = []
    for a in np.linspace(0, 2*np.pi, 8, endpoint=False):
        pts_cc.append([1.0 * np.cos(a), 1.0 * np.sin(a)])
    for a in np.linspace(np.pi/8, 2*np.pi + np.pi/8, 8, endpoint=False):
        pts_cc.append([0.5 * np.cos(a), 0.5 * np.sin(a)])
    concentric_circles_pts = np.array(pts_cc)
    
    # 6. Fibonacci spiral
    pts_fib = []
    golden_angle = np.pi * (3 - np.sqrt(5))
    for k in range(n):
        r = 0.8 * np.sqrt(k / n)
        theta = k * golden_angle
        pts_fib.append([r * np.cos(theta), r * np.sin(theta)])
    fibonacci_pts = np.array(pts_fib)
    
    # 7. Square ring + center cluster
    pts_ring = []
    for a in np.linspace(0, 2*np.pi, 12, endpoint=False):
        pts_ring.append([1.0 * np.cos(a), 1.0 * np.sin(a)])
    for a in np.linspace(np.pi/12, 2*np.pi + np.pi/12, 4, endpoint=False):
        pts_ring.append([0.4 * np.cos(a), 0.4 * np.sin(a)])
    ring_pts = np.array(pts_ring)
    
    # 8. Optimized two-ring: 10 outer + 6 inner
    pts_2ring = []
    for a in np.linspace(0, 2*np.pi, 10, endpoint=False):
        pts_2ring.append([1.0 * np.cos(a), 1.0 * np.sin(a)])
    for a in np.linspace(np.pi/10, 2*np.pi + np.pi/10, 6, endpoint=False):
        pts_2ring.append([0.55 * np.cos(a), 0.55 * np.sin(a)])
    two_ring_pts = np.array(pts_2ring)
    
    initial_guesses = [grid_pts, hex_pts, circle_pts, rand_pts, 
                       concentric_circles_pts, fibonacci_pts, ring_pts, two_ring_pts]
    
    # Optimization Loop with L-BFGS-B using adaptive beta
    num_guesses = len(initial_guesses)
    for i, init_pts in enumerate(initial_guesses):
        if time.time() > deadline - 120:  # Leave time for polish
            break
            
        # Normalize initial points to roughly unit scale to help optimizer
        scale = np.max(np.linalg.norm(init_pts, axis=1))
        if scale > 0:
            init_pts_scaled = init_pts / scale
        else:
            init_pts_scaled = init_pts
        
        # Adaptive beta: start moderate, increase for later iterations
        # Use different betas for different initializations
        beta_schedule = [50.0, 80.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0]
        beta = beta_schedule[i % len(beta_schedule)]
        
        try:
            # Use L-BFGS-B with smooth soft-min/soft-max objective
            def objective(params, b=beta):
                ratio = soft_min_max_ratio(params, b)
                return -ratio
            
            res = minimize(
                objective, 
                init_pts_scaled.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14, 'maxfun': 10000}
            )
            
            if time.time() > deadline - 60:
                if best_points is not None:
                    break
                candidate_points = res.x.reshape((n, 2))
                candidate_score = get_score(candidate_points)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_points.copy()
                break
                
            candidate_points = res.x.reshape((n, 2))
            
            # Calculate score for this candidate
            candidate_score = get_score(candidate_points)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points.copy()
                
        except Exception:
            continue
    
    # Additional refinement: try multiple Nelder-Mead passes with different starting points
    if best_points is not None and time.time() < deadline - 90:
        try:
            def objective_nm(params):
                pts = params.reshape((n, 2))
                dists = compute_dists(pts)
                d_min = np.min(dists)
                d_max = np.max(dists)
                if d_max < 1e-10:
                    return 0.0
                return -(d_min / d_max)
            
            # First NM polish from best
            res_nm = minimize(
                objective_nm,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'fatol': 1e-15, 'xatol': 1e-15, 'adaptive': True}
            )
            
            nm_points = res_nm.x.reshape((n, 2))
            nm_score = get_score(nm_points)
            
            if nm_score > best_score:
                best_score = nm_score
                best_points = nm_points.copy()
                
        except Exception:
            pass
    
    # Second round: perturb best and re-optimize with smooth objective
    if best_points is not None and time.time() < deadline - 30:
        try:
            np.random.seed(123)
            for pert_iter in range(5):
                if time.time() > deadline - 10:
                    break
                    
                # Add small random perturbation
                perturbation = np.random.randn(n, 2) * 0.01
                perturbed = best_points + perturbation
                
                # Re-optimize with high beta for precision
                def objective_high_beta(params):
                    ratio = soft_min_max_ratio(params, 500.0)
                    return -ratio
                
                res_pert = minimize(
                    objective_high_beta,
                    perturbed.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14, 'maxfun': 5000}
                )
                
                pert_points = res_pert.x.reshape((n, 2))
                pert_score = get_score(pert_points)
                
                if pert_score > best_score:
                    best_score = pert_score
                    best_points = pert_points.copy()
                    
        except Exception:
            pass
    
    # Final Nelder-Mead polish
    if best_points is not None and time.time() < deadline - 5:
        try:
            def objective_nm_final(params):
                pts = params.reshape((n, 2))
                dists = compute_dists(pts)
                d_min = np.min(dists)
                d_max = np.max(dists)
                if d_max < 1e-10:
                    return 0.0
                return -(d_min / d_max)
            
            res_nm_final = minimize(
                objective_nm_final,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'fatol': 1e-16, 'xatol': 1e-16, 'adaptive': True}
            )
            
            final_points = res_nm_final.x.reshape((n, 2))
            final_score = get_score(final_points)
            
            if final_score > best_score:
                best_score = final_score
                best_points = final_points.copy()
                
        except Exception:
            pass
    
    # If no valid points were found (should not happen), return grid
    if best_points is None:
        return grid_pts

    return best_points
