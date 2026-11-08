"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle.
Strategy: Combinatorial Local Search (Hill Climbing).
Instead of smooth approximations, we directly manipulate points.
We identify the triangle with the minimum area (the 'critical' triangle).
We then attempt to move one of its vertices to increase this area, 
while ensuring no other triangle becomes smaller than the current target.
This is a form of active set method / coordinate descent on the combinatorial structure.
"""
import numpy as np
import random
import time
from itertools import combinations

N = 11

# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
AREA = np.sqrt(3) / 4.0  # area of the equilateral triangle


def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 of the points, normalized by AREA.
    Returns (min_area, min_indices) where min_indices is the tuple (i, j, k) of the critical triangle.
    """
    n = len(points)
    min_area = float('inf')
    min_idx = (0, 1, 2)
    
    # Precompute cross products for speed? 
    # For N=11, 165 triangles. Direct calculation is fine.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * | (Pj - Pi) x (Pk - Pi) |
                # Cross product in 2D: (x1*y2 - y1*x2)
                x1 = points[j, 0] - points[i, 0]
                y1 = points[j, 1] - points[i, 1]
                x2 = points[k, 0] - points[i, 0]
                y2 = points[k, 1] - points[i, 1]
                
                cross = x1 * y2 - y1 * x2
                area = 0.5 * abs(cross)
                
                if area < min_area:
                    min_area = area
                    min_idx = (i, j, k)
                    
    return (min_area / AREA, min_idx)


def _is_inside_triangle(p):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -1e-9:
        return False
    if x < -1e-9 or x > 1.0 + 1e-9:
        return False
    max_y = np.sqrt(3) * x if x <= 0.5 else np.sqrt(3) * (1.0 - x)
    return y <= max_y + 1e-9


def _generate_structured_initial():
    """
    Generate a highly symmetric initial configuration.
    For N=11, we can use:
    - 3 vertices (slightly inset to allow movement)
    - 3 midpoints of sides
    - 1 centroid
    - 4 points forming a smaller rotated square or similar around centroid?
    
    Let's try a specific known good pattern for 11 points:
    A regular hexagon of 6 points + 3 vertices + 2 interior?
    Or: 3 vertices, 3 midpoints, 1 centroid, 4 points at 1/3 and 2/3 of medians?
    
    Let's use a grid-like structure projected onto the triangle.
    Barycentric coordinates (u, v, w) with u+v+w=1.
    Points with coordinates from {0, 1/3, 2/3, 1} normalized?
    
    Let's try:
    Vertices: (1,0,0), (0,1,0), (0,0,1) -> inset slightly
    Midpoints: (1/2, 1/2, 0), etc.
    Centroid: (1/3, 1/3, 1/3)
    Remaining 4: Points at (2/3, 1/3, 0), (1/3, 2/3, 0), etc?
    Actually, let's just use a set of points that are well-separated.
    
    A good starting point is often the "best known" configuration or a symmetric one.
    Let's construct:
    1. V0_inset
    2. V1_inset
    3. V2_inset
    4. Mid(V0,V1)
    5. Mid(V1,V2)
    6. Mid(V2,V0)
    7. Centroid
    8-11. Points forming a small square around centroid?
    """
    inset = 0.05
    pts = [
        V0 + np.array([inset, inset]),
        V1 + np.array([-inset, inset]),
        V2 + np.array([0, -inset]),
        (V0 + V1) / 2.0,
        (V1 + V2) / 2.0,
        (V2 + V0) / 2.0,
        (V0 + V1 + V2) / 3.0,
        # Add 4 more points. Let's place them at 1/4 and 3/4 along the medians?
        # Or just random but seeded to be somewhat spread out.
        # Let's try points at barycentric (1/4, 1/4, 1/2) etc?
        # (1/4, 1/4, 1/2) -> 0.25*V0 + 0.25*V1 + 0.5*V2
        0.25 * V0 + 0.25 * V1 + 0.5 * V2,
        0.5 * V0 + 0.25 * V1 + 0.25 * V2,
        0.25 * V0 + 0.5 * V1 + 0.25 * V2,
        # One more? Maybe a point near the center but offset?
        # Let's use a random point near center for variety, or another symmetric one.
        # Symmetry suggests 3 points around center. We have 7 (3v, 3m, 1c). 
        # 4 more. 
        # Let's try (1/6, 1/6, 2/3) etc? No, that's close to vertices.
        # Let's just use a random initialization for the last 4 to break symmetry if needed, 
        # but for a structured start, let's place them at 1/3 height lines.
        # Line y = sqrt(3)/6. x from 1/6 to 5/6.
        # Points: (1/6, sqrt(3)/6), (5/6, sqrt(3)/6), (3/6, sqrt(3)/6) is centroid? No.
        # Centroid is y = sqrt(3)/6. x=0.5.
        # So we have centroid.
        # Let's add: (0.25, sqrt(3)*0.25), (0.75, sqrt(3)*0.25), (0.5, sqrt(3)*0.5), (0.5, sqrt(3)*0.25)??
        # Let's just pick 4 distinct points.
        np.array([0.25, np.sqrt(3) * 0.25]),
        np.array([0.75, np.sqrt(3) * 0.25]),
        np.array([0.5, np.sqrt(3) * 0.5]),
        np.array([0.5, np.sqrt(3) * 0.1])
    ]
    
    # Ensure they are inside
    valid_pts = []
    for p in pts:
        if _is_inside_triangle(p):
            valid_pts.append(p)
        else:
            # Project to nearest point inside? Or just skip and random fill?
            # For robustness, if any is outside, we'll just randomize the whole set later.
            pass
            
    if len(valid_pts) == N:
        return np.array(valid_pts)
    
    # Fallback to random if structured fails
    rng = random.Random(123)
    return _generate_random_points(rng)


def _generate_random_points(rng, n=N):
    """Generate n random points inside the triangle using rejection sampling."""
    points = []
    while len(points) < n:
        u = rng.random()
        v = rng.random()
        if u + v <= 1.0:
            p = V0 * (1 - u - v) + V1 * u + V2 * v
            points.append(p)
    return np.array(points)


def _local_search(points, time_limit=10.0):
    """
    Perform local search to improve the minimum triangle area.
    Uses a hill-climbing approach:
    1. Find the critical triangle (min area).
    2. Try to move one of its vertices to increase the area of that triangle,
       while checking that no other triangle area decreases below the current min.
    """
    start_time = time.time()
    current_min, min_idx = _min_triangle_area(points)
    points = points.copy()
    
    # Step size for perturbation
    # Start with a relatively large step, decrease if stuck
    base_step = 0.05
    min_step = 1e-5
    
    step = base_step
    stall_count = 0
    
    while time.time() - start_time < time_limit:
        # If step is too small, we've converged locally
        if step < min_step:
            break
            
        improved = False
        i, j, k = min_idx
        
        # Try moving i, j, k
        for idx in [i, j, k]:
            # Try moving in 8 directions + 4 diagonal?
            # Directions: N, S, E, W, NE, NW, SE, SW
            directions = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ]
            
            for dx, dy in directions:
                new_point = points[idx].copy()
                new_point[0] += dx * step
                new_point[1] += dy * step
                
                # Check if inside
                if not _is_inside_triangle(new_point):
                    continue
                
                # Try the move
                old_point = points[idx].copy()
                points[idx] = new_point
                
                new_min, _ = _min_triangle_area(points)
                
                if new_min > current_min + 1e-12:
                    current_min = new_min
                    min_idx = _min_triangle_area(points)[1] # Re-find critical
                    improved = True
                    stall_count = 0
                    break # Restart direction search for this point? Or just continue?
                    # Let's restart the loop to re-evaluate critical triangle
                else:
                    # Revert
                    points[idx] = old_point
            
            if improved:
                break
        
        if not improved:
            # Try moving other points? 
            # The strategy is to move critical triangle vertices. 
            # If that fails, maybe move a non-critical point to "make room"?
            # For now, reduce step size.
            stall_count += 1
            if stall_count > 5:
                step *= 0.8
                stall_count = 0
            else:
                # Try a random perturbation of a non-critical point?
                # This helps escape local minima where moving critical vertices is blocked.
                non_critical = [x for x in range(N) if x not in min_idx]
                if non_critical:
                    rand_idx = random.choice(non_critical)
                    # Small random kick
                    dx = (random.random() - 0.5) * step
                    dy = (random.random() - 0.5) * step
                    new_point = points[rand_idx].copy()
                    new_point[0] += dx
                    new_point[1] += dy
                    if _is_inside_triangle(new_point):
                        old_point = points[rand_idx].copy()
                        points[rand_idx] = new_point
                        new_min, _ = _min_triangle_area(points)
                        if new_min > current_min + 1e-12:
                            current_min = new_min
                            min_idx = _min_triangle_area(points)[1]
                            improved = True
                        else:
                            points[rand_idx] = old_point
        
        if not improved and step < min_step * 2:
            break
            
    return points, current_min


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_points = None
    best_min = 0.0
    
    rng = random.Random(42)
    
    # Initial candidates
    candidates = []
    candidates.append(_generate_structured_initial())
    
    # Add random starts
    for _ in range(5):
        candidates.append(_generate_random_points(rng))
    
    # For each candidate, run local search
    # We have ~145s. 
    # Local search is fast (165 area calcs per step). 
    # We can afford many iterations.
    
    for idx, init_pts in enumerate(candidates):
        if time.time() > deadline:
            break
            
        # Run local search with a time budget
        # Give more time to the first (structured) and subsequent ones
        # If we have lots of time left, give more.
        time_left = deadline - time.time()
        # Distribute time: give at least 5s to each, but cap total.
        # Since we have ~6 candidates, 145/6 is ~24s each.
        # Let's use a dynamic budget: if early, give 20s. If late, give remaining.
        search_time = min(20.0, time_left / 2.0) # Heuristic
        
        try:
            optimized_pts, min_val = _local_search(init_pts, time_limit=search_time)
            
            if min_val > best_min:
                best_min = min_val
                best_points = optimized_pts.copy()
        except Exception:
            continue
    
    # If we have time left, polish the best solution with a very fine step
    if best_points is not None and time.time() < deadline - 5:
        polish_time = deadline - time.time() - 2
        if polish_time > 1:
            best_points, best_min = _local_search(best_points, time_limit=polish_time)
            # Ensure best_min is updated
            best_min, _ = _min_triangle_area(best_points)
            
    if best_points is None:
        best_points = _generate_random_points(rng)
        
    return best_points
