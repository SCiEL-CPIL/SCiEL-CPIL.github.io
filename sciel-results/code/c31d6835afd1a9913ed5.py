import numpy as np
import time
from scipy.optimize import differential_evolution, minimize, LinearConstraint
from scipy.spatial import ConvexHull

N = 14
DEADLINE = 120.0  # Safety margin before 150s kill

# Precompute triangle indices
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]

def compute_score(points):
    """Compute min triangle area / convex hull area."""
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    return min_area / hull_area

def neg_score(x):
    points = x.reshape((N, 2))
    return -compute_score(points)

def soft_min_score(x, tau=0.01):
    """
    Compute -soft_min(areas) / hull_area.
    soft_min(A) = -tau * log(sum(exp(-A/tau)))
    """
    points = x.reshape((N, 2))
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    areas = np.maximum(areas, 1e-12)
        
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    sum_exp = np.sum(np.exp(scaled - max_scaled))
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    
    return -soft_min_val / hull_area

def two_ring_configuration(params):
    """
    Generate points from two concentric regular heptagons.
    params: [r1, r2, th1, th2]
    r1: radius of outer heptagon
    r2: radius of inner heptagon
    th1: rotation of outer heptagon
    th2: rotation of inner heptagon
    """
    r1, r2, th1, th2 = params
    
    angles1 = th1 + np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles2 = th2 + np.linspace(0, 2*np.pi, 7, endpoint=False)
    
    points1 = np.column_stack((r1*np.cos(angles1), r1*np.sin(angles1)))
    points2 = np.column_stack((r2*np.cos(angles2), r2*np.sin(angles2)))
    
    return np.vstack([points1, points2])

def neg_two_ring_score(params):
    try:
        points = two_ring_configuration(params)
        return -compute_score(points)
    except Exception:
        return 1.0

def three_ring_configuration(params):
    """
    Generate points from three concentric regular polygons.
    Fixed split: 5, 4, 5 points.
    params: [r1, th1, r2, th2, r3, th3]
    """
    r1, th1, r2, th2, r3, th3 = params
    
    # Ring 1: 5 points
    angles1 = th1 + np.linspace(0, 2*np.pi, 5, endpoint=False)
    points1 = np.column_stack((r1*np.cos(angles1), r1*np.sin(angles1)))
    
    # Ring 2: 4 points
    angles2 = th2 + np.linspace(0, 2*np.pi, 4, endpoint=False)
    points2 = np.column_stack((r2*np.cos(angles2), r2*np.sin(angles2)))
    
    # Ring 3: 5 points
    angles3 = th3 + np.linspace(0, 2*np.pi, 5, endpoint=False)
    points3 = np.column_stack((r3*np.cos(angles3), r3*np.sin(angles3)))
    
    return np.vstack([points1, points2, points3])

def neg_three_ring_score(params):
    try:
        points = three_ring_configuration(params)
        return -compute_score(points)
    except Exception:
        return 1.0

def simulated_annealing(initial_points, rng, max_steps=10000, start_temp=1.0, end_temp=0.01, cooling_rate=0.99):
    """
    Run Simulated Annealing to refine the points.
    """
    current_points = initial_points.copy()
    current_score = compute_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    temp = start_temp
    
    for step in range(max_steps):
        # Generate a new candidate by adding Gaussian noise
        noise_scale = 0.05
        new_points = current_points + rng.normal(0, noise_scale, current_points.shape)
        
        new_score = compute_score(new_points)
        
        # Acceptance criterion
        if new_score > current_score or (temp > 0 and rng.random() < np.exp((new_score - current_score) / temp)):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        # Cool down
        temp *= cooling_rate
        
        # Break if temperature is too low
        if temp < end_temp:
            break
            
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 10  # Safety margin
    
    best_points = None
    best_score = -float('inf')
    rng = np.random.default_rng(12345)
    
    def try_update(points):
        nonlocal best_points, best_score
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s

    # --- Phase 1: Two-Ring (7+7) Global Search ---
    
    bounds_2ring = [
        (0.5, 2.0),  # r1 (outer)
        (0.0, 1.5),  # r2 (inner)
        (0, 2*np.pi), # th1
        (0, 2*np.pi), # th2
    ]
    
    try:
        res_de_2ring = differential_evolution(
            neg_two_ring_score,
            bounds_2ring,
            seed=42,
            maxiter=200,
            tol=1e-9,
            popsize=30,
            mutation=(0.5, 1.5),
            recombination=0.9,
            polish=True,
            init='sobol'
        )
        
        if res_de_2ring.fun < 0:
            best_params_2ring = res_de_2ring.x
            points_2ring = two_ring_configuration(best_params_2ring)
            try_update(points_2ring)
    except Exception:
        pass

    # --- Phase 2: Three-Ring (5+4+5) Global Search ---
    
    bounds_3ring = [
        (0.5, 2.0),  # r1
        (0, 2*np.pi), # th1
        (0.2, 1.5),  # r2
        (0, 2*np.pi), # th2
        (0.0, 1.0),  # r3
        (0, 2*np.pi), # th3
    ]
    
    if time.time() < deadline - 30:
        try:
            res_de_3ring = differential_evolution(
                neg_three_ring_score,
                bounds_3ring,
                seed=43,
                maxiter=150,
                tol=1e-8,
                popsize=25,
                mutation=(0.5, 1.5),
                recombination=0.9,
                polish=True,
                init='sobol'
            )
            
            if res_de_3ring.fun < 0:
                best_params_3ring = res_de_3ring.x
                points_3ring = three_ring_configuration(best_params_3ring)
                try_update(points_3ring)
        except Exception:
            pass

    # --- Phase 3: Simulated Annealing Refinement ---
    
    if best_points is not None:
        # Run SA for the remaining time
        # We will run a loop of SA restarts from the current best point
        current_best = best_points.copy()
        
        while time.time() < deadline - 5:
            # Run SA for a fixed number of steps
            sa_points, sa_score = simulated_annealing(
                current_best, 
                rng, 
                max_steps=5000, 
                start_temp=0.5, 
                end_temp=0.01, 
                cooling_rate=0.995
            )
            
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_points.copy()
                current_best = sa_points.copy()
            else:
                # If no improvement, perturb the starting point for next SA
                current_best = best_points + rng.normal(0, 0.1, best_points.shape)

    if best_points is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
