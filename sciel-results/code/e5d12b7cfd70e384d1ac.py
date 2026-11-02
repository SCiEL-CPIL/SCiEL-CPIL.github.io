"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

DEADLINE = 150.0
START_TIME = time.time()
SAFETY_MARGIN = 5.0

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    # C(13, 3) = 286 triangles
    # Vectorized calculation
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i = i[mask]
    j = j[mask]
    k = k[mask]
    
    p_i = points[i]
    p_j = points[j]
    p_k = points[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    return np.min(areas)

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-10:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _generate_initial_points(rng):
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Regular polygon
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    configs.append(np.column_stack([np.cos(theta), np.sin(theta)]))
    
    # 2. Concentric rings (high symmetry)
    # e.g., 1 center, 6 inner, 6 outer
    center = np.array([[0.0, 0.0]])
    r1, r2 = 0.5, 1.0
    th1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    th2 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    inner = r1 * np.column_stack([np.cos(th1), np.sin(th1)])
    outer = r2 * np.column_stack([np.cos(th2), np.sin(th2)])
    configs.append(np.vstack([center, inner, outer]))
    
    # 3. Random points in unit square
    configs.append(rng.random((N, 2)))
    
    # 4. Random points on circle with noise
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.normal(0, 0.1, N)
    r = 1.0 + rng.normal(0, 0.1, N)
    configs.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
    
    # 5. Jittered regular polygon
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    base = np.column_stack([np.cos(theta), np.sin(theta)])
    configs.append(base + rng.normal(0, 0.1, (N, 2)))
    
    return configs

def construct():
    entry_time = time.time()
    deadline = entry_time + (DEADLINE - SAFETY_MARGIN)
    
    rng = np.random.default_rng(42)
    best_score = 0.0
    best_points = None
    
    # Initial evaluation of diverse starts
    inits = _generate_initial_points(rng)
    for pts in inits:
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
            
    # Simulated Annealing Strategy
    # We use SA because the objective function (min triangle area) is non-smooth and multimodal.
    # SA allows escaping local minima by accepting worse solutions probabilistically.
    
    T = 0.1  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.995
    step_size = 0.05
    
    current_points = best_points.copy()
    current_score = best_score
    
    # Adaptive step size: start small, increase if stuck, decrease if improving
    # But we also want to explore. Let's use a fixed schedule with occasional resets.
    
    n_iter = 0
    max_iter = 100000
    stuck_count = 0
    
    while time.time() < deadline - 10 and n_iter < max_iter:
        n_iter += 1
        
        # Generate neighbor
        # Perturb a random subset of points to allow larger moves
        num_perturb = np.random.randint(1, 5)  # Perturb 1 to 4 points
        idx = np.random.choice(N, num_perturb, replace=False)
        
        neighbor = current_points.copy()
        # Gaussian perturbation
        noise = rng.normal(0, step_size, (num_perturb, 2))
        neighbor[idx] += noise
        
        # Optional: Normalize to prevent drift? 
        # The problem is scale invariant (ratio), so normalization isn't strictly needed for score, 
        # but keeping points bounded helps numerical stability.
        # However, SA works on the space of configurations. 
        # If points fly off to infinity, hull area grows, min tri area grows, ratio stays similar.
        # But numerical errors might occur. 
        # Let's center and scale occasionally or just let it run. 
        # Actually, since score is scale invariant, we can just let the optimizer wander.
        # But to keep numbers reasonable, let's re-center every 1000 steps.
        if n_iter % 1000 == 0:
            neighbor = neighbor - neighbor.mean(axis=0)
            norm = np.max(np.linalg.norm(neighbor, axis=1))
            if norm > 0:
                neighbor = neighbor / norm
        
        neighbor_score = _score(neighbor)
        
        delta = neighbor_score - current_score
        
        if delta > 0:
            # Accept improvement
            current_points = neighbor
            current_score = neighbor_score
            stuck_count = 0
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                # Cool down slightly faster when finding new best? Or keep same.
        else:
            # Accept worse with probability exp(delta / T)
            if np.random.rand() < np.exp(delta / T):
                current_points = neighbor
                current_score = neighbor_score
                stuck_count += 1
            else:
                stuck_count += 1
        
        # Cooling
        T *= cooling_rate
        
        # Adaptive step size
        if stuck_count > 200:
            # Stuck, try a bigger jump
            step_size *= 1.5
            stuck_count = 0
        elif stuck_count < 50:
            # Improving, refine
            step_size *= 0.95
        
        # Ensure step size is within bounds
        step_size = np.clip(step_size, 0.001, 0.5)
        
        # Restart logic: if T is too low or stuck for too long, restart from best with new noise
        if T < T_min or stuck_count > 500:
            # Restart from best_points with small perturbation
            current_points = best_points + rng.normal(0, 0.01, (N, 2))
            current_score = _score(current_points)
            T = 0.1
            step_size = 0.05
            stuck_count = 0
            # Limit restarts to avoid infinite loop if best is very stable
            # But we have time budget.
            
    # Final local polish with Nelder-Mead on the best point found
    if time.time() < deadline - 5:
        def neg_score(pts_flat):
            pts = pts_flat.reshape(N, 2)
            return -_score(pts)
        
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.success:
                pts_opt = res.x.reshape(N, 2)
                s_opt = -res.fun
                if s_opt > best_score:
                    best_score = s_opt
                    best_points = pts_opt.copy()
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
