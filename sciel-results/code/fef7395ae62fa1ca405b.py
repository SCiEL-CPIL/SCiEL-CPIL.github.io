import numpy as np
import time
from scipy.spatial.distance import cdist

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # 1. Initialization: Hexagonal grid
    # We want to fit 200 circles.
    # A hexagonal grid has rows with alternating offsets.
    # Let's estimate the number of rows.
    # Approximate area per circle is 1/sqrt(N).
    # Let's try to generate a grid that covers the square.
    
    # A better initialization:
    # Place points in a hexagonal lattice, then clip to the square.
    # Then, if we have too many, remove the ones closest to the boundary?
    # Or just take the first N.
    
    # Let's use a standard hex grid generation.
    # Spacing s.
    # We want the grid to cover [-0.5, 0.5] x [-0.5, 0.5].
    # Let's pick a spacing that gives roughly N points.
    # Area = 1. Number of points ~ 1 / (s * s * sin(60)) * 2?
    # Hex area per point is (sqrt(3)/2) * s^2.
    # N ~ 1 / ((sqrt(3)/2) * s^2) => s ~ sqrt(2 / (N * sqrt(3)))
    
    s = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    
    # Generate hex grid
    # Rows are horizontal.
    # Row i has y = i * s * sqrt(3)/2
    # x offset for row i is (i % 2) * s / 2
    
    points = []
    # Determine range of i and j
    # y range [-0.5, 0.5]
    # x range [-0.5, 0.5]
    
    # Let's generate a large enough grid and filter
    max_i = int(1.0 / (s * np.sqrt(3.0)/2.0)) + 2
    max_j = int(1.0 / s) + 2
    
    for i in range(-max_i, max_i + 1):
        y = i * s * np.sqrt(3.0) / 2.0
        if y > 0.5 or y < -0.5:
            continue
        x_offset = (i % 2) * s / 2.0
        for j in range(-max_j, max_j + 1):
            x = j * s + x_offset
            if x > 0.5 or x < -0.5:
                continue
            points.append([x, y])
            
    points = np.array(points)
    
    # If we have more than N points, we need to select N.
    # If we have fewer, we need to add more or adjust spacing.
    # For N=200, s ~ sqrt(2/(200*1.732)) ~ sqrt(0.00577) ~ 0.076.
    # Grid size approx 1/0.076 ~ 13.
    # Rows approx 13.
    # Total points approx 13*13 = 169. We might be short.
    # Let's adjust s to get closer to N points.
    
    # Actually, let's just use a square grid initialization if hex is tricky, 
    # but hex is better. Let's refine the hex generation to ensure we get >= N points.
    
    # Try to find a spacing that yields close to N points in the square.
    # We can just use the points we have. If < N, we can add random points or shrink s.
    # If > N, we can remove points that are "least useful" (e.g., closest to boundary? No, we want to keep the densest part).
    # Actually, for packing, we want to maximize min distance. 
    # A subset of a hex grid is a valid packing.
    # If we have > N, we should pick the N that maximize the min distance.
    # This is hard. 
    # Easier: Adjust s such that we get roughly N points.
    
    # Let's try a few spacings and pick the one that gives N points or closest.
    best_points = None
    best_count_diff = 1e9
    
    for scale in np.linspace(0.9, 1.1, 21):
        s_curr = s * scale
        pts = []
        max_i = int(1.0 / (s_curr * np.sqrt(3.0)/2.0)) + 2
        max_j = int(1.0 / s_curr) + 2
        for i in range(-max_i, max_i + 1):
            y = i * s_curr * np.sqrt(3.0) / 2.0
            if y > 0.5 or y < -0.5:
                continue
            x_offset = (i % 2) * s_curr / 2.0
            for j in range(-max_j, max_j + 1):
                x = j * s_curr + x_offset
                if x > 0.5 or x < -0.5:
                    continue
                pts.append([x, y])
        pts = np.array(pts)
        if len(pts) == 0:
            continue
        diff = abs(len(pts) - N)
        if diff < best_count_diff:
            best_count_diff = diff
            best_points = pts
            
    if best_points is None:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    # If we have more than N, we need to reduce.
    # If we have less, we need to increase.
    # For now, let's assume we can get close.
    # If len > N, we can just take the first N? No, that's biased.
    # Let's use a simple heuristic: keep the points that are furthest from the boundary?
    # Or just run local search on the set we have.
    
    # Let's ensure we have exactly N points.
    if len(best_points) > N:
        # Randomly select N? Or select the ones with largest local distance?
        # Let's just take a random subset to avoid bias, then optimize.
        indices = np.random.choice(len(best_points), N, replace=False)
        best_points = best_points[indices]
    elif len(best_points) < N:
        # Add random points in the square, avoiding overlaps if possible (hard).
        # Just add random points, local search will fix it.
        extra = np.random.uniform(-0.5, 0.5, size=(N - len(best_points), 2))
        best_points = np.vstack([best_points, extra])
        
    # Local Search / Optimization
    # Objective: Maximize r = min(min_dist, min_boundary_dist)
    # min_dist = min_{i<j} ||p_i - p_j||
    # min_boundary_dist = min_i min_k (0.5 - |p_ik|)
    
    def score(points):
        # Check boundaries
        b = np.min(0.5 - np.abs(points))
        if b <= 0:
            return -1.0 # Invalid
        # Check distances
        # cdist is O(N^2), 200x200 = 40000. Fast.
        d = cdist(points, points)
        np.fill_diagonal(d, np.inf)
        m_d = np.min(d)
        return min(b, m_d)
        
    current_score = score(best_points)
    if current_score < 0:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    best_points = best_points.copy()
    best_score = current_score
    
    # Hill Climbing
    # Step size starts large, decreases.
    step_size = 0.05
    no_improve_count = 0
    
    while time.time() < deadline:
        if no_improve_count > 1000:
            step_size *= 0.9
            if step_size < 1e-5:
                break
            no_improve_count = 0
            
        # Pick a random point
        idx = np.random.randint(N)
        old_point = best_points[idx].copy()
        
        # Random direction
        angle = np.random.uniform(0, 2*np.pi)
        move = np.array([np.cos(angle), np.sin(angle)]) * step_size
        new_point = old_point + move
        
        # Check if new point is inside square
        if np.any(np.abs(new_point) >= 0.5):
            continue
            
        # Update point
        best_points[idx] = new_point
        
        # Calculate new score
        # Optimization: Only need to check distances to neighbors?
        # For simplicity and correctness, calculate full score.
        # 40000 ops per step. 110s / (40000 ops) ~ huge number of steps.
        # NumPy cdist is fast.
        new_score = score(best_points)
        
        if new_score > best_score:
            best_score = new_score
            no_improve_count = 0
        else:
            # Revert
            best_points[idx] = old_point
            no_improve_count += 1
            # If we revert, we might want to try a smaller step or different direction?
            # For now, just count it.
            
        # Occasionally, try a larger jump to escape local minima?
        if no_improve_count % 100 == 0 and step_size > 0.01:
             # Try a random restart of a few points?
             pass # Keep it simple for now.

    return best_points
