import time
import numpy as np
from scipy.spatial import ConvexHull
import random

N = 13

# EVOLVE-BLOCK-START

def compute_score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # Calculate all triangle areas
    min_area = float('inf')
    
    # Vectorized approach is possible but 286 triples is small.
    # Loop is sufficient and easy to optimize with numpy ops if needed, 
    # but pure python loop over 286 items is very fast.
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            p2 = points[j]
            v1 = p2 - p1
            for k in range(j + 1, n):
                p3 = points[k]
                v2 = p3 - p1
                # Area is 0.5 * |cross product|
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
                    if min_area < 1e-10:
                        return 0.0
    
    if min_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def generate_initialization(method='polygon'):
    """
    Generate various initial configurations.
    """
    if method == 'polygon':
        theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        radius = 0.5
        center = [0.5, 0.5]
        points = np.array([[center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)] for t in theta])
        return points
    elif method == 'halton':
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        points = []
        for i in range(1, N + 1):
            x = halton(i, 2)
            y = halton(i, 3)
            points.append([x, y])
        return np.array(points)
    elif method == 'grid':
        # 3x3 grid in [0.1, 0.9] plus 4 points on edges to make 13?
        # Or a 4x4 grid subset.
        # Let's try a specific structured grid:
        # 3x3 grid centered at 0.5, step 0.3
        coords = [0.2, 0.5, 0.8]
        grid_pts = [[x, y] for x in coords for y in coords]
        # Add 4 points: midpoints of the square [0.2, 0.8]x[0.2, 0.8] edges?
        # Or just push corners out?
        # Let's use the 4 corners of a larger square [0.1, 0.9]
        extra_pts = [[0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]]
        # This might create small triangles at corners. 
        # Better: Regular 13-gon is usually a strong start.
        # Let's stick to a perturbed polygon for 'grid' to ensure convexity initially.
        # Actually, let's just return a slightly perturbed regular polygon for variety in init.
        theta = np.linspace(0, 2*np.pi, N, endpoint=False) + np.random.uniform(-0.1, 0.1, N)
        radius = 0.5 + np.random.uniform(-0.05, 0.05, N)
        center = [0.5, 0.5]
        points = np.array([[center[0] + r * np.cos(t), center[1] + r * np.sin(t)] for t, r in zip(theta, radius)])
        return points
    elif method == 'random':
        return np.random.rand(N, 2)
    else:
        return np.random.rand(N, 2)

def local_search_adaptive(points, time_limit=10.0):
    """
    Perform Hill Climbing with adaptive step size.
    """
    current_points = points.copy()
    current_score = compute_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    # Adaptive step size
    step_size = 0.05
    min_step_size = 1e-6
    max_step_size = 0.5
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 50
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        # Generate perturbation
        # Move 1 or 2 points
        n_moves = random.randint(1, 2)
        indices = random.sample(range(N), n_moves)
        
        new_points = current_points.copy()
        for idx in indices:
            noise = np.random.normal(0, step_size, 2)
            new_points[idx] = current_points[idx] + noise
            
        new_score = compute_score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
            no_improve_count = 0
            # Optionally increase step size slightly if we are making good progress?
            # Usually we keep it or decrease slightly to refine.
            # Let's keep step size constant on success to explore local neighborhood.
        else:
            no_improve_count += 1
            # Reduce step size
            step_size *= 0.95
            if step_size < min_step_size:
                step_size = min_step_size
                
        # If stuck for a long time, kick
        if no_improve_count > max_no_improve:
            kick_idx = random.randint(0, N - 1)
            # Large random jump
            jump_size = random.uniform(0.1, 0.3)
            new_points = current_points.copy()
            new_points[kick_idx] = current_points[kick_idx] + np.random.normal(0, jump_size, 2)
            new_score = compute_score(new_points)
            
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                no_improve_count = 0
                step_size = 0.05 # Reset step size
            else:
                # If kick failed, just reset step size to a medium value and continue
                step_size = 0.02
                no_improve_count = 0

    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_global_points = None
    best_global_score = 0.0
    
    # List of strategies
    # We prioritize structured inits which tend to converge to better basins
    strategies = [
        'polygon',
        'halton',
        'grid',
        'polygon',
        'random',
        'random',
        'halton',
        'grid'
    ]
    
    # Shuffle to ensure variety in order, but we want to try 'polygon' first usually?
    # Actually, let's just iterate through them.
    
    num_strategies = len(strategies)
    
    for i, strat in enumerate(strategies):
        if time.time() > deadline - 5.0:
            break
            
        # Allocate time based on remaining strategies
        remaining = num_strategies - i
        time_per_strat = max(2.0, (deadline - time.time() - 5.0) / remaining)
        
        # Generate initial points
        init_pts = generate_initialization(strat)
        
        # Run local search
        pts, score = local_search_adaptive(init_pts, time_limit=time_per_strat)
        
        if score > best_global_score:
            best_global_score = score
            best_global_points = pts.copy()
            
    # If we still have time, do a final refinement on the best point
    if best_global_points is not None and time.time() < deadline - 10.0:
        remaining_time = deadline - time.time() - 5.0
        if remaining_time > 1.0:
             # Use a smaller initial step size for final polish
             # We can call local_search_adaptive again with a tighter time limit
             # and let the adaptive mechanism handle the fine-tuning
             final_pts, final_score = local_search_adaptive(best_global_points, time_limit=remaining_time)
             if final_score > best_global_score:
                 best_global_score = final_score
                 best_global_points = final_pts

    if best_global_points is None:
        # Fallback
        best_global_points = generate_initialization('polygon')
        
    return best_global_points
# EVOLVE-BLOCK-END
