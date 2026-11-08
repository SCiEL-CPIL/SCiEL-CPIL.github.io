"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull

DEADLINE = 145.0  # Safety margin before 150s kill


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            local_min = 0.5 * np.min(np.abs(cross))
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-14:
                    return min_area
    return min_area


def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area_vec(pts)
    return min_area / hull_a


def _normalize_unit_area(pts):
    """Scale points so convex hull area is 1."""
    hull_a = _hull_area(pts)
    if hull_a > 0:
        return pts / np.sqrt(hull_a)
    return pts


def _initial_regular_polygon(n):
    """Points on a regular n-gon inscribed in unit circle, scaled to unit area."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return _normalize_unit_area(pts)


def _initial_mixed(n):
    """Mix of boundary and interior points."""
    outer_n = 10
    inner_n = n - outer_n
    angles = 2 * np.pi * np.arange(outer_n) / outer_n
    outer = np.column_stack([np.cos(angles), np.sin(angles)])
    inner_angles = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    inner = 0.4 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    pts = np.vstack([outer, inner])
    return _normalize_unit_area(pts)

def _generate_initial_configs():
    """Generate a diverse set of initial configurations to ensure global coverage."""
    configs = []
    # 1. Regular polygon
    configs.append(_initial_regular_polygon(N))
    
    # 2. Mixed boundary/interior (various inner radii)
    for r in [0.2, 0.4, 0.6, 0.8]:
        outer_n = 10
        inner_n = N - outer_n
        angles = 2 * np.pi * np.arange(outer_n) / outer_n
        outer = np.column_stack([np.cos(angles), np.sin(angles)])
        inner_angles = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
        inner = r * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
        pts = np.vstack([outer, inner])
        configs.append(_normalize_unit_area(pts))
        
    # 3. Random configurations (seeded for reproducibility)
    rng = np.random.RandomState(42)
    for _ in range(5):
        # Random points in unit circle, then normalized
        theta = rng.uniform(0, 2*np.pi, N)
        r = rng.uniform(0, 1, N) ** 0.5
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        # Ensure non-degenerate hull
        if len(np.unique(np.round(pts, 4), axis=0)) < 3:
            continue
        configs.append(_normalize_unit_area(pts))
        
    # 4. "Hexagonal" style: 6 outer, 6 inner, 2 center-ish
    # This is a heuristic guess for better packing
    outer_n = 6
    inner_n = 6
    center_n = 2
    angles_out = 2 * np.pi * np.arange(outer_n) / outer_n
    outer = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    angles_in = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi/inner_n
    inner = 0.5 * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
    # Two points near center, slightly offset
    p1 = np.array([0.1, 0.1])
    p2 = np.array([-0.1, -0.1])
    pts = np.vstack([outer, inner, [p1, p2]])
    configs.append(_normalize_unit_area(pts))

    return configs


def _generalized_pattern_search(points, max_time=140.0):
    """
    Perform Generalized Pattern Search on the point configuration.
    Uses a set of directions (coordinate axes and their negatives) and adaptive step sizes.
    """
    start_time = time.time()
    best_pts = points.copy()
    best_score = _score(best_pts)
    
    # Directions: +/- e_i for each coordinate
    n_params = N * 2
    directions = np.zeros((n_params, n_params))
    for i in range(n_params):
        directions[i, i] = 1.0
        directions[i, i] = 1.0 # Wait, we need both + and -
        # Actually, GPS typically uses a set of directions. 
        # Let's use coordinate directions: +e_i and -e_i.
        
    # Construct direction matrix
    dirs = []
    for i in range(n_params):
        d_pos = np.zeros(n_params)
        d_pos[i] = 1.0
        d_neg = np.zeros(n_params)
        d_neg[i] = -1.0
        dirs.append(d_pos)
        dirs.append(d_neg)
    dirs = np.array(dirs)
    
    # Step sizes for each direction
    step_sizes = np.ones(len(dirs)) * 0.05
    
    # Main loop
    while time.time() - start_time < max_time:
        improved = False
        
        # Try each direction
        for k in range(len(dirs)):
            if time.time() - start_time > max_time:
                break
                
            # Propose new point
            new_pts_flat = best_pts.flatten() + step_sizes[k] * dirs[k]
            new_pts = new_pts_flat.reshape(N, 2)
            
            new_score = _score(new_pts)
            
            if new_score > best_score:
                # Accept
                best_pts = new_pts
                best_score = new_score
                step_sizes[k] *= 2.0  # Expand step size
                improved = True
            else:
                # Reject
                step_sizes[k] *= 0.5  # Contract step size
                
                # If step size is too small, skip this direction for a while?
                # Or just keep it small.
                if step_sizes[k] < 1e-8:
                    continue

        if not improved:
            # No direction improved. 
            # This is a local minimum for the current step sizes.
            # We can try to "restart" or "kick" by randomizing step sizes or directions?
            # Or we can conclude this local search is done and move to the next initial config.
            # Since we are doing this per config, we can just break and let the outer loop handle it.
            # However, GPS often has a "random direction" component or restarts.
            # Let's add a small random perturbation to escape if stuck for a while?
            # For simplicity, if no improvement, we break out of this config's GPS.
            break
            
    return best_pts, best_score


def construct():
    start_time = time.time()
    
    # Generate initial candidates
    candidates = _generate_initial_configs()
    
    best_pts_global = None
    best_score_global = -1.0
    
    # Evaluate each candidate and run GPS on it
    for cand in candidates:
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # Quick evaluation first
        cand_score = _score(cand)
        
        if cand_score > best_score_global:
            best_score_global = cand_score
            best_pts_global = cand.copy()
            
        # Run GPS to refine this candidate
        # Allocate remaining time minus safety margin, split among remaining candidates?
        # Simpler: give each candidate a fixed budget if time allows, or just run until deadline.
        # Since GPS is fast per iteration, let's run it for a short fixed time per candidate
        # to ensure we explore all, then polish the best.
        
        # Estimate time per candidate
        elapsed = time.time() - start_time
        remaining = DEADLINE - 5 - elapsed
        candidates_remaining = len(candidates) - candidates.index(cand)
        
        if candidates_remaining > 0:
            time_per_candidate = remaining / candidates_remaining
            # Cap time per candidate to avoid spending too long on a bad one
            time_per_candidate = min(time_per_candidate, 10.0) 
        else:
            time_per_candidate = remaining
        
        if time_per_candidate < 1.0:
            continue
            
        refined_pts, refined_score = _generalized_pattern_search(cand, max_time=time_per_candidate)
        
        if refined_score > best_score_global:
            best_score_global = refined_score
            best_pts_global = refined_pts.copy()
            
    # Final polish on the best found so far
    if time.time() - start_time < DEADLINE - 2 and best_pts_global is not None:
        # Run a tighter GPS on the best point
        final_pts, final_score = _generalized_pattern_search(best_pts_global, max_time=DEADLINE - time.time() - 2)
        if final_score > best_score_global:
            best_score_global = final_score
            best_pts_global = final_pts.copy()

    if best_pts_global is None:
        return _initial_regular_polygon(N)
        
    return best_pts_global
# EVOLVE-BLOCK-END
