import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time

N = 14

# Precompute triple indices for vectorized area computation
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

    min_area = 0.5 * np.min(np.abs(cross))
    return min_area / hull_area

def _soft_min_value(areas, temp):
    max_a = np.max(areas)
    if max_a < 1e-12:
        return 0.0
    shifted = areas - max_a
    log_sum = np.log(np.sum(np.exp(shifted / temp)))
    return max_a - temp * log_sum

def _smooth_objective(x, temp=0.05):
    p = x.reshape(N, 2)
    if p.shape != (N, 2):
        return 1e6
    
    try:
        hull = ConvexHull(p)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 1e6
    except Exception:
        return 1e6

    p_i = p[TRIPLE_INDICES[:, 0]]
    p_j = p[TRIPLE_INDICES[:, 1]]
    p_k = p[TRIPLE_INDICES[:, 2]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area_soft = _soft_min_value(areas, temp)
    
    return -min_area_soft / hull_area

def _polish_points(points, time_limit=10.0):
    start_time = time.time()
    
    best_pts = points.copy()
    best_score = _calculate_score(points)
    
    # Temperature annealing schedule
    temp_schedule = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    
    for temp in temp_schedule:
        if time.time() - start_time > time_limit * 0.8:
            break
        
        def neg_smooth_score(x, t=temp):
            return _smooth_objective(x, t)

        try:
            res = minimize(neg_smooth_score, best_pts.flatten(), method='L-BFGS-B',
                          options={'maxiter':300, 'ftol':1e-14, 'gtol':1e-12, 'maxls':100})
            new_pts = res.x.reshape(N, 2)
            new_score = _calculate_score(new_pts)
            if new_score > best_score:
                best_pts = new_pts
                best_score = new_score
        except Exception:
            pass
    
    # Final fine-tuning with very low temperature
    if time.time() - start_time < time_limit * 0.95:
        def neg_smooth_score_final(x, t=0.001):
            return _smooth_objective(x, t)
        
        try:
            res = minimize(neg_smooth_score_final, best_pts.flatten(), method='L-BFGS-B',
                          options={'maxiter':200, 'ftol':1e-15, 'gtol':1e-13, 'maxls':80})
            new_pts = res.x.reshape(N, 2)
            new_score = _calculate_score(new_pts)
            if new_score > best_score:
                best_pts = new_pts
                best_score = new_score
        except Exception:
            pass
    
    return best_pts, best_score

def _generate_hull_interior_config(k_hull, rng, jitter=0.05):
    """
    Generate a configuration with k_hull points on a regular polygon (hull)
    and N-k_hull points inside.
    """
    # Hull vertices: regular k_hull-gon
    angles_hull = 2 * np.pi * np.arange(k_hull) / k_hull
    hull_pts = np.column_stack((np.cos(angles_hull), np.sin(angles_hull)))
    
    # Interior points: placed in a smaller regular polygon or grid
    n_interior = N - k_hull
    if n_interior == 0:
        return hull_pts
    
    # Place interior points in a regular polygon with smaller radius
    # and some random jitter to break symmetry
    r_in = 0.4 + 0.2 * rng.uniform()
    angles_in = 2 * np.pi * np.arange(n_interior) / n_interior + rng.uniform(0, 2*np.pi/n_interior)
    interior_pts = r_in * np.column_stack((np.cos(angles_in), np.sin(angles_in)))
    
    # Add small random jitter to all points to break exact symmetry
    all_pts = np.vstack((hull_pts, interior_pts))
    all_pts += rng.randn(N, 2) * jitter
    
    return all_pts

def _optimize_single_config(points, time_limit=20.0):
    """
    Optimize a single point configuration using annealed smooth optimization.
    """
    start_time = time.time()
    
    # Phase 1: Annealed L-BFGS-B optimization
    polished_pts, polished_score = _polish_points(points, time_limit=time_limit * 0.7)
    
    # Phase 2: Multiple restarts with small perturbations
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5.0:
        rng = np.random.RandomState(123)
        for restart in range(5):
            if time.time() - start_time > time_limit * 0.9:
                break
            
            noise_scale = 0.02 * (restart + 1)
            perturbed = polished_pts + rng.randn(N, 2) * noise_scale
            
            polished2, score2 = _polish_points(perturbed, time_limit=min(5.0, time_limit - (time.time() - start_time)))
            if score2 > polished_score:
                polished_pts = polished2
                polished_score = score2
    
    return polished_pts, polished_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 150s limit - 10s buffer
    
    best_score = 0
    best_points = None
    
    def try_update(score, points):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Strategy: Explore different hull topologies
    # For each number of hull vertices k, generate configurations and optimize
    
    rng = np.random.RandomState(42)
    
    # Try hull sizes from 3 to 14
    # Prioritize smaller hulls (3-7) as they may allow better interior packing
    hull_sizes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    # Allocate time based on hull size
    # Smaller hulls get more time as they're more promising
    time_weights = {}
    for k in hull_sizes:
        if k <= 5:
            time_weights[k] = 3.0
        elif k <= 8:
            time_weights[k] = 2.0
        else:
            time_weights[k] = 1.0
    
    total_weight = sum(time_weights.values())
    available_time = deadline - start_time - 15.0  # Reserve 15s for final polish
    
    for k in hull_sizes:
        if time.time() > deadline - 20:
            break
        
        # Generate multiple configurations for this hull size
        n_configs = 3 if k <= 6 else 2
        
        for config_idx in range(n_configs):
            if time.time() > deadline - 25:
                break
            
            # Generate initial configuration
            points = _generate_hull_interior_config(k, rng, jitter=0.03 + 0.02 * config_idx)
            
            # Optimize
            time_for_this = min(15.0, (deadline - time.time() - 20) * 0.5)
            if time_for_this < 3.0:
                continue
            
            optimized_pts, optimized_score = _optimize_single_config(points, time_limit=time_for_this)
            try_update(optimized_score, optimized_pts)
            
            # Also try a random interior placement
            if config_idx == 0 and k < N:
                # Random interior points
                hull_angles = 2 * np.pi * np.arange(k) / k
                hull_pts = np.column_stack((np.cos(hull_angles), np.sin(hull_angles)))
                
                n_interior = N - k
                interior_pts = rng.uniform(-0.5, 0.5, (n_interior, 2))
                # Ensure interior points are inside the hull
                # Simple approach: scale down
                interior_pts *= 0.6
                
                all_pts = np.vstack((hull_pts, interior_pts))
                all_pts += rng.randn(N, 2) * 0.02
                
                time_for_rand = min(10.0, (deadline - time.time() - 20) * 0.3)
                if time_for_rand > 2.0:
                    optimized_pts2, optimized_score2 = _optimize_single_config(all_pts, time_limit=time_for_rand)
                    try_update(optimized_score2, optimized_pts2)
    
    # Final intensive polish on the best configuration
    if best_points is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        final_pts, final_score = _polish_points(best_points, time_limit=max(5.0, remaining))
        try_update(final_score, final_pts)
    
    # Fallback: regular 14-gon if nothing worked
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _calculate_score(best_points)
        remaining = deadline - time.time() - 3
        if remaining > 5.0:
            polished_pts, polished_score = _polish_points(best_points, time_limit=min(10.0, remaining))
            best_points = polished_pts
            best_score = polished_score
    
    return best_points
