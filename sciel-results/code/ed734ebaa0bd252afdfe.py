# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools
import math

N = 13
# Precompute triangle indices for efficiency
TRI_IDX = np.array(list(itertools.combinations(range(N), 3)))

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull
    Returns 0.0 if invalid
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
        
    # Vectorized area calculation
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    A_min = np.min(areas)
    
    return A_min / A_hull

def soft_min_score(points, temperature=0.01):
    """
    Smooth surrogate for the objective function.
    We want to maximize A_min / A_hull.
    So we minimize -A_min / A_hull.
    A_min = min(areas).
    Soft min: -T * log(sum(exp(-area/T)))
    """
    if not np.all(np.isfinite(points)):
        return 1e10
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 1e10
    
    if A_hull <= 0:
        return 1e10
        
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    # Soft min of areas
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_area = -temperature * log_sum
    
    # Objective to minimize: -soft_min_area / A_hull
    return -soft_min_area / A_hull

def generate_initial_candidates(rng):
    """
    Generate diverse initial candidates based on geometric templates.
    """
    candidates = []
    
    # Template 1: Regular 13-gon
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(pts)
    
    # Template 2: 1 center + 12 outer (regular 12-gon)
    center = np.array([0.0, 0.0])
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    outer = np.column_stack([np.cos(angles), np.sin(angles)])
    pts = np.vstack([center, outer])
    candidates.append(pts)
    
    # Template 3: 1 center + 6 inner + 6 outer
    center = np.array([0.0, 0.0])
    angles_in = np.linspace(0, 2*np.pi, 6, endpoint=False)
    inner = 0.5 * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
    angles_out = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    outer = 1.0 * np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    pts = np.vstack([center, inner, outer])
    candidates.append(pts)
    
    # Template 4: 2 center + 5 inner + 6 outer
    c1 = np.array([-0.1, 0.0])
    c2 = np.array([0.1, 0.0])
    angles_in = np.linspace(0, 2*np.pi, 5, endpoint=False)
    inner = 0.5 * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
    angles_out = np.linspace(0, 2*np.pi, 6, endpoint=False)
    outer = 1.0 * np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    pts = np.vstack([c1, c2, inner, outer])
    candidates.append(pts)
    
    # Template 5: Grid-like (4x4 grid, remove 3 points to get 13, centered)
    xs = np.array([-1, -0.33, 0.33, 1.0])
    ys = np.array([-1, -0.33, 0.33, 1.0])
    grid = np.array([[x, y] for x in xs for y in ys])
    pts = grid[:13]
    candidates.append(pts)
    
    # Template 6: Random
    pts = rng.uniform(-1, 1, size=(13, 2))
    candidates.append(pts)
    
    # Template 7: Fibonacci spiral
    golden_angle = 2.39996
    radii = np.sqrt(np.arange(13) / 13.0)
    angles = np.arange(13) * golden_angle
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    candidates.append(pts)
    
    # Template 8: Perturbed Grid
    xs = np.array([-1, -0.33, 0.33, 1.0])
    ys = np.array([-1, -0.33, 0.33, 1.0])
    grid = np.array([[x, y] for x in xs for y in ys])
    # Remove 3 random points
    indices = rng.choice(16, 13, replace=False)
    pts = grid[indices]
    candidates.append(pts)
    
    # Template 9: Two rings
    r1 = 0.5
    r2 = 1.0
    angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    angles2 = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    ring1 = r1 * np.column_stack([np.cos(angles1), np.sin(angles1)])
    ring2 = r2 * np.column_stack([np.cos(angles2), np.sin(angles2)])
    pts = np.vstack([ring1, ring2])
    candidates.append(pts)
    
    return candidates

def local_polish(points, time_limit):
    """
    Perform local coordinate descent to polish the points.
    """
    start_time = time.time()
    current_score = compute_score(points)
    current_points = points.copy()
    step_size = 0.05
    
    while time.time() - start_time < time_limit:
        improved = False
        for idx in range(N):
            # Try moving each point in small steps
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_points = current_points.copy()
                    new_points[idx, 0] += dx
                    new_points[idx, 1] += dy
                    
                    # Quick validity check
                    if not np.all(np.isfinite(new_points)):
                        continue
                    try:
                        hull = ConvexHull(new_points)
                        if hull.volume <= 0:
                            continue
                    except Exception:
                        continue
                    
                    new_score = compute_score(new_points)
                    if new_score > current_score:
                        current_points = new_points
                        current_score = new_score
                        improved = True
        if not improved:
            step_size *= 0.5
            if step_size < 1e-5:
                break
    return current_points, current_score

def heilbronn_convex13() -> np.ndarray:
    """
    Optimize 13 points to maximize the minimum triangle area ratio.
    Uses multi-start optimization with Powell's method and local polishing.
    """
    start_time = time.time()
    deadline = start_time + 600.0 - 10.0 # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    # Generate initial candidates
    candidates = generate_initial_candidates(rng)
    
    # Add some perturbed versions of the best structured candidates
    for base in candidates[:5]:
        for scale in [0.1, 0.3, 0.5]:
            perturbed = base + rng.normal(0, scale, size=base.shape)
            candidates.append(perturbed)
    
    # Shuffle candidates to ensure variety
    rng.shuffle(candidates)
    
    n_candidates = len(candidates)
    
    # Determine time per candidate
    # We want to run multiple full optimizations.
    # Allocate 80% of time to optimization, 20% to polish.
    search_deadline = start_time + (deadline - start_time) * 0.8
    
    # If we have very few candidates, give them more time.
    # If many, give them less.
    time_per_candidate = max(5.0, (search_deadline - start_time) / n_candidates)
    
    for i, init_pts in enumerate(candidates):
        current_time = time.time()
        if current_time > search_deadline:
            break
            
        # Normalize initial points to unit circle-ish scale for better optimization behavior
        init_pts_centered = init_pts - np.mean(init_pts, axis=0)
        max_norm = np.max(np.linalg.norm(init_pts_centered, axis=1))
        if max_norm > 1e-6:
            init_pts_scaled = init_pts_centered / max_norm
        else:
            init_pts_scaled = init_pts_centered
        
        # Flatten for optimization
        x0 = init_pts_scaled.flatten()
        
        # Define objective function for optimization (minimization)
        def obj_func(x_flat, temp=0.01):
            pts = x_flat.reshape(13, 2)
            return soft_min_score(pts, temperature=temp)
            
        # Run Powell optimization
        # Powell is derivative-free and robust for this dimension
        try:
            res = minimize(
                obj_func, 
                x0, 
                method='Powell', 
                options={'maxiter': 1000, 'xtol': 1e-8, 'ftol': 1e-8, 'disp': False},
                timeout=time_per_candidate
            )
            x_opt = res.x
        except Exception:
            x_opt = x0
            
        # Convert back to points
        pts_opt = x_opt.reshape(13, 2)
        
        # Calculate hard score
        score = compute_score(pts_opt)
        
        if score > best_score:
            best_score = score
            best_points = pts_opt.copy()
            
    # Polish the best point found using local coordinate descent
    if best_points is not None:
        polish_deadline = deadline - 5.0
        current_time = time.time()
        if current_time < polish_deadline:
            polish_time = polish_deadline - current_time
            # Allocate half the polish time to the best candidate, and maybe try a second best if time permits
            # For simplicity, just polish the best one thoroughly
            polished_points, polished_score = local_polish(best_points, polish_time)
            
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_points.copy()
                
            # If we have time left, try a second polish with a slightly different start or step
            current_time = time.time()
            if current_time < polish_deadline - 2:
                 # Try a random restart from the best point with small noise
                 noise = rng.normal(0, 0.01, size=best_points.shape)
                 new_start = best_points + noise
                 polished_points2, polished_score2 = local_polish(new_start, polish_deadline - current_time - 1)
                 if polished_score2 > best_score:
                     best_score = polished_score2
                     best_points = polished_points2.copy()

    # Fallback
    if best_points is None:
        best_points = candidates[0]
        
    # Ensure valid output
    if not np.all(np.isfinite(best_points)):
        best_points = np.column_stack([np.cos(np.linspace(0, 2*np.pi, 13, endpoint=False)), np.sin(np.linspace(0, 2*np.pi, 13, endpoint=False))])
        
    return best_points

# EVOLVE-BLOCK-END
