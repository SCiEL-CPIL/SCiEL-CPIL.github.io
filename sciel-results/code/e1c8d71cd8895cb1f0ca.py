import numpy as np
import scipy.spatial as spat
import time
import itertools

N = 14
DEADLINE = 145  # 5s safety margin before 150s kill

def _calc_min_tri_area(points):
    """
    Compute the minimum triangle area over all triples.
    Vectorized for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Create index arrays
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    # Only keep unique triples i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    # Get points
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    # Calculate cross product area
    # Area = 0.5 * | (p_j - p_i) x (p_k - p_i) |
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """Score = min triangle area / hull area. Higher is better."""
    hull_a = _hull_area(points)
    if hull_a < 1e-10:
        return 0.0
    min_tri = _calc_min_tri_area(points)
    return min_tri / hull_a

def _soft_min_score(points, beta=50.0):
    """
    Compute a smooth approximation of the minimum triangle area divided by hull area.
    Uses log-sum-exp for the soft minimum.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Create index arrays for unique triples i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # This approximates min(areas) as beta -> infinity
    # To avoid overflow, we can subtract the max (which is min of -areas)
    # Let z = -beta * areas. soft_min = -1/beta * log(sum(exp(z)))
    # z_max = max(z) = -beta * min(areas)
    # sum(exp(z)) = exp(z_max) * sum(exp(z - z_max))
    # log(sum(exp(z))) = z_max + log(sum(exp(z - z_max)))
    # soft_min = -1/beta * (z_max + log(sum(exp(z - z_max))))
    #           = min(areas) - 1/beta * log(sum(exp(z - z_max)))
    
    z = -beta * areas
    z_max = np.max(z)
    if not np.isfinite(z_max):
        return 0.0
        
    exp_z_minus_max = np.exp(z - z_max)
    sum_exp = np.sum(exp_z_minus_max)
    if sum_exp == 0:
        return 0.0
        
    log_sum_exp = z_max + np.log(sum_exp)
    soft_min_area = -1.0 / beta * log_sum_exp
    
    hull_a = _hull_area(points)
    if hull_a < 1e-10:
        return 0.0
        
    return soft_min_area / hull_a

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    """Generate n points on a regular polygon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return np.column_stack([xs, ys])

def _random_perturbed_polygon(n, radius=1.0, center=(0.0, 0.0), noise_scale=0.1, seed=42):
    """Generate a regular polygon with random radial and angular perturbations."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    # Perturb angles
    angle_noise = rng.normal(0, noise_scale * 0.1, n)
    angles += angle_noise
    
    # Perturb radii
    radii = np.ones(n) * radius
    radius_noise = rng.normal(0, noise_scale, n)
    radii += radius_noise
    
    xs = center[0] + radii * np.cos(angles)
    ys = center[1] + radii * np.sin(angles)
    return np.column_stack([xs, ys])

def construct():
    start_time = time.time()
    
    best_points = None
    best_score = -np.inf
    
    # Initial configurations: Regular polygon and perturbed versions
    initial_configs = []
    
    # 1. Regular polygon
    initial_configs.append(_regular_polygon(N))
    
    # 2. Perturbed regular polygons with different noise levels and seeds
    for seed in range(10):
        noise_scale = 0.05 + 0.01 * seed
        initial_configs.append(_random_perturbed_polygon(N, noise_scale=noise_scale, seed=seed))
        
    # 3. Random points in a circle (fallback)
    rng = np.random.RandomState(123)
    for _ in range(5):
        r = np.sqrt(rng.random(N))
        theta = rng.uniform(0, 2 * np.pi, N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        initial_configs.append(pts)
        
    # Simulated Annealing Parameters
    T_start = 0.1
    T_end = 0.001
    cooling_rate = 0.999
    max_iter_per_start = 2000
    
    for init_pts in initial_configs:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:  # Need 20s for final processing
            break
            
        # Normalize initial points to unit area hull
        hull_a = _hull_area(init_pts)
        if hull_a < 1e-10:
            continue
        init_pts = init_pts / np.sqrt(hull_a)
        
        current_pts = init_pts.copy()
        current_score = _score(current_pts)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_pts.copy()
            
        T = T_start
        rng = np.random.RandomState(int(elapsed * 1000) % 10000)
        
        for i in range(max_iter_per_start):
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 10:
                break
                
            # Generate a new point by perturbing a random point
            new_pts = current_pts.copy()
            idx = rng.randint(N)
            # Gaussian perturbation
            noise = rng.normal(0, 0.01, 2)
            new_pts[idx] += noise
            
            # Normalize new points to unit area hull
            new_hull_a = _hull_area(new_pts)
            if new_hull_a < 1e-10:
                T *= cooling_rate
                continue
            new_pts = new_pts / np.sqrt(new_hull_a)
            
            new_score = _score(new_pts)
            
            # Acceptance criterion
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_pts.copy()
            else:
                # Metropolis criterion
                delta = new_score - current_score
                if T > 0:
                    accept_prob = np.exp(delta / T)
                    if rng.random() < accept_prob:
                        current_pts = new_pts
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_points = current_pts.copy()
                            
            T *= cooling_rate
            
        # Local refinement with small steps after SA
        T_local = 0.001
        for i in range(1000):
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 5:
                break
            new_pts = current_pts.copy()
            idx = rng.randint(N)
            noise = rng.normal(0, 0.001, 2)
            new_pts[idx] += noise
            
            new_hull_a = _hull_area(new_pts)
            if new_hull_a < 1e-10:
                continue
            new_pts = new_pts / np.sqrt(new_hull_a)
            
            new_score = _score(new_pts)
            
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_pts.copy()
                    
    if best_points is None:
        best_points = initial_configs[0]
        
    return np.array(best_points)
