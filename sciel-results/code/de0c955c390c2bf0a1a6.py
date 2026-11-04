import numpy as np
import time
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14

# Precompute triple indices
_TRIPLE_INDICES = np.array(list(combinations(range(N), 3)), dtype=int)
_N_TRIPLES = _TRIPLE_INDICES.shape[0]

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _triangle_areas(points):
    """Compute all triangle areas (unnormalized)."""
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
        
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    
    return min_area / hull_area

def _simulated_annealing(points, start_time, deadline, initial_temp=0.1, cooling_rate=0.995, min_temp=0.001):
    """
    Simulated annealing to maximize the minimum triangle area / hull area ratio.
    Uses targeted perturbations: move one point at a time with adaptive step size.
    """
    best_pts = points.copy()
    best_score = _score(best_pts)
    current_pts = points.copy()
    current_score = best_score
    temp = initial_temp
    
    rng = np.random.RandomState(42)
    
    while time.time() < deadline and temp > min_temp:
        # Choose a random point to perturb
        idx = rng.randint(N)
        
        # Adaptive step size based on temperature
        step_scale = temp * 0.5 + 0.01
        delta = rng.randn(2) * step_scale
        
        trial_pts = current_pts.copy()
        trial_pts[idx] += delta
        
        # Keep points within a reasonable bound (unit circle)
        norm = np.linalg.norm(trial_pts[idx])
        if norm > 1.5:
            trial_pts[idx] *= 1.5 / norm
        
        trial_score = _score(trial_pts)
        
        # Accept if better or with probability exp(-delta_score/temp)
        if trial_score > current_score:
            current_pts = trial_pts
            current_score = trial_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            delta_score = current_score - trial_score
            if delta_score < 0:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta_score / temp)
            if rng.random() < accept_prob:
                current_pts = trial_pts
                current_score = trial_score
        
        # Cool down
        temp *= cooling_rate
    
    return best_pts, best_score

def _local_search(points, max_iters=200, step_size=0.01):
    """
    Coordinate-level local search to escape local minima.
    Tries small perturbations in each coordinate direction.
    """
    best_pts = points.copy()
    best_s = _score(best_pts)
    
    for it in range(max_iters):
        improved = False
        for i in range(N):
            for coord in range(2):
                for delta in [step_size, -step_size, step_size*0.5, -step_size*0.5]:
                    trial = best_pts.copy()
                    trial[i, coord] += delta
                    s = _score(trial)
                    if s > best_s + 1e-10:
                        best_pts = trial
                        best_s = s
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            # Try larger steps
            step_size *= 0.5
            if step_size < 1e-4:
                break
    
    return best_pts, best_s

def _generate_symmetric_configs():
    """
    Generate various symmetric configurations as starting points.
    """
    configs = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Two concentric rings: 7 outer, 7 inner
    for r_inner in [0.3, 0.5, 0.7]:
        for theta_offset in [0, np.pi/7, np.pi/3.5]:
            outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
            inner_angles = theta_offset + np.linspace(0, 2*np.pi, 7, endpoint=False)
            outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
            inner_pts = r_inner * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
            configs.append(np.vstack([outer_pts, inner_pts]))
    
    # 3. Three rings: 6 outer, 5 middle, 3 inner
    for r_mid in [0.6, 0.8]:
        for r_in in [0.2, 0.4]:
            outer_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
            mid_angles = np.pi/6 + np.linspace(0, 2*np.pi, 5, endpoint=False)
            in_angles = np.linspace(0, 2*np.pi, 3, endpoint=False)
            outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
            mid_pts = r_mid * np.column_stack([np.cos(mid_angles), np.sin(mid_angles)])
            in_pts = r_in * np.column_stack([np.cos(in_angles), np.sin(in_angles)])
            configs.append(np.vstack([outer_pts, mid_pts, in_pts]))
    
    # 4. Grid-like pattern
    # 4 rows x 3.5 columns, centered
    rows = 4
    cols = 4
    x_vals = np.linspace(-0.8, 0.8, cols)
    y_vals = np.linspace(-0.8, 0.8, rows)
    grid_pts = []
    for y in y_vals:
        for x in x_vals:
            grid_pts.append([x, y])
    # Take first 14 points
    grid_pts = np.array(grid_pts[:N])
    configs.append(grid_pts)
    
    # 5. Distorted hexagonal pattern
    # Center + 6 around + 7 more
    center = np.array([0.0, 0.0])
    ring1_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    ring1 = 0.5 * np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles)])
    ring2_angles = np.pi/6 + np.linspace(0, 2*np.pi, 7, endpoint=False)
    ring2 = 0.9 * np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
    configs.append(np.vstack([center.reshape(1,2), ring1, ring2]))
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 125.0  # Safety margin before 150s limit

    # Generate diverse starting configurations
    candidates = _generate_symmetric_configs()
    
    # Add some random configurations
    np.random.seed(42)
    for _ in range(5):
        pts = np.random.rand(N, 2) * 2 - 1
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / np.maximum(norms, 1e-6)[:, np.newaxis]
        candidates.append(pts)
    
    # Score all candidates
    scored_candidates = []
    for pts in candidates:
        s = _score(pts)
        scored_candidates.append((s, pts))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_score = scored_candidates[0][0] if scored_candidates else 0
    best_points = scored_candidates[0][1].copy() if scored_candidates else np.zeros((N, 2))
    
    # Run simulated annealing on top candidates
    top_k = min(5, len(scored_candidates))
    for i in range(top_k):
        if time.time() > deadline - 10:
            break
        
        s_init, pts_init = scored_candidates[i]
        
        # Run simulated annealing with different parameters
        for temp0 in [0.05, 0.1, 0.2]:
            if time.time() > deadline - 10:
                break
            
            sa_pts, sa_score = _simulated_annealing(
                pts_init, start_time, deadline,
                initial_temp=temp0, cooling_rate=0.995, min_temp=0.001
            )
            
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_pts.copy()
        
        # Local search polish on the best from SA
        if time.time() < deadline - 5:
            polished_pts, polished_s = _local_search(best_points, max_iters=100, step_size=0.02)
            if polished_s > best_score:
                best_score = polished_s
                best_points = polished_pts
    
    return best_points
