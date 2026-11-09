import numpy as np
import time
import itertools

N = 11
h = np.sqrt(3) / 2

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, h])

def project_to_triangle(p):
    """
    Project points (N, 2) into the triangle defined by V0, V1, V2.
    Uses barycentric coordinates.
    """
    # Barycentric coordinates
    # p = u0*V0 + u1*V1 + u2*V2
    # V0=[0,0], V1=[1,0], V2=[0.5, h]
    # p_x = u1 + 0.5 u2
    # p_y = h u2
    # u2 = p_y / h
    # u1 = p_x - 0.5 u2
    # u0 = 1 - u1 - u2
    
    u2 = p[:, 1] / h
    u1 = p[:, 0] - 0.5 * u2
    u0 = 1.0 - u1 - u2
    
    # Clip to [0, 1] and normalize to ensure they sum to 1 (simplex constraint)
    # This is a standard projection onto the simplex
    u = np.column_stack((u0, u1, u2))
    
    # Simple projection: if any < 0, clip to 0 and renormalize
    # This is not the exact Euclidean projection but sufficient for SA
    # to keep points inside.
    
    # A better projection:
    # 1. Clip negatives to 0
    # 2. Normalize rows to sum to 1
    
    u_clipped = np.clip(u, 0, None)
    sums = np.sum(u_clipped, axis=1, keepdims=True)
    # Handle case where sum is 0 (shouldn't happen if we start inside)
    sums = np.where(sums == 0, 1, sums)
    u_norm = u_clipped / sums
    
    # Convert back to cartesian
    # P = u1*V1 + u2*V2 (since V0=0)
    p_x = u_norm[:, 1] * 1.0 + u_norm[:, 2] * 0.5
    p_y = u_norm[:, 1] * 0.0 + u_norm[:, 2] * h
    
    return np.column_stack((p_x, p_y))

def min_triangle_area(points):
    """
    Vectorized calculation of the minimum area of any triangle formed by 3 points.
    points: (N, 2)
    """
    n = len(points)
    # We need to check all combinations of 3 points
    # N=11, C(11,3) = 165 combinations.
    # We can compute all areas at once using broadcasting or loops.
    # Loop is likely faster for N=11 due to low overhead compared to huge intermediate arrays?
    # Actually, let's just loop over i, j, k but vectorize the area calc?
    # No, 165 is small. A simple triple loop in Python might be slow.
    # Let's use itertools to generate indices and then vectorize the area calculation.
    
    # Generate all triples
    indices = list(itertools.combinations(range(n), 3))
    if not indices:
        return 0.0
        
    # Create an array of shape (M, 3, 2) where M is number of triples
    # This might be memory heavy for large N, but for N=11, M=165, it's tiny.
    idx_arr = np.array(indices) # (M, 3)
    
    # Get points for each triple
    # p1: (M, 2), p2: (M, 2), p3: (M, 2)
    p1 = points[idx_arr[:, 0]]
    p2 = points[idx_arr[:, 1]]
    p3 = points[idx_arr[:, 2]]
    
    # Vector area calculation: 0.5 * | (p2-p1) x (p3-p1) |
    v12 = p2 - p1
    v13 = p3 - p1
    
    # Cross product in 2D: x1*y2 - y1*x2
    cross = v12[:, 0] * v13[:, 1] - v12[:, 1] * v13[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial guess: Random points inside triangle
    # To ensure they are inside, we can sample barycentric coords uniformly
    rng = np.random.default_rng(42)
    
    # Generate random barycentric coordinates
    # u ~ Dirichlet(1,1,1)
    # Or just random uniform in [0,1] and normalize
    u = rng.random((N, 3))
    u = u / np.sum(u, axis=1, keepdims=True)
    
    # Convert to cartesian
    p_x = u[:, 1] * 1.0 + u[:, 2] * 0.5
    p_y = u[:, 1] * 0.0 + u[:, 2] * h
    current_points = np.column_stack((p_x, p_y))
    
    # Ensure strictly inside (avoid boundary issues if any)
    current_points = project_to_triangle(current_points)
    
    current_area = min_triangle_area(current_points)
    best_points = current_points.copy()
    best_area = current_area
    
    # Simulated Annealing Parameters
    T = 0.01  # Initial temperature (area scale)
    T_min = 1e-6
    alpha = 0.999  # Cooling rate
    step_size = 0.05 # Initial step size for Gaussian perturbation
    
    while time.time() < deadline:
        # Generate new points
        # Perturb each point with Gaussian noise
        noise = rng.normal(0, step_size, current_points.shape)
        new_points = current_points + noise
        
        # Project back to triangle
        new_points = project_to_triangle(new_points)
        
        # Calculate new area
        new_area = min_triangle_area(new_points)
        
        # Acceptance criterion
        if new_area > current_area:
            # Always accept improvement
            current_points = new_points
            current_area = new_area
            
            if current_area > best_area:
                best_area = current_area
                best_points = current_points.copy()
        else:
            # Accept worse solution with probability exp((new - current) / T)
            # Note: We are maximizing area, so delta = new - current is negative
            delta = new_area - current_area
            prob = np.exp(delta / T)
            if rng.random() < prob:
                current_points = new_points
                current_area = new_area
        
        # Cool down
        T *= alpha
        # Occasionally reheat if stuck?
        # For now, just cool. If T gets too small, we might get stuck.
        # Let's add a simple restart logic if no improvement for a long time?
        # Or just keep going. SA with many iterations is robust.
        
        # Adjust step size?
        # If we accept many worse solutions, maybe increase step size?
        # If we accept few, decrease?
        # Simple heuristic: keep constant or slowly decrease.
        step_size *= 0.9995
        
        if T < T_min:
            # Reset temperature and restart from best
            T = 0.01
            current_points = best_points + rng.normal(0, 0.01, best_points.shape)
            current_points = project_to_triangle(current_points)
            current_area = min_triangle_area(current_points)
            step_size = 0.05

    return best_points

# EVOLVE-BLOCK-END
