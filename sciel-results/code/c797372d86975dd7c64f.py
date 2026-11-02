import numpy as np
import time
import random

N = 11
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0  # Area of unit equilateral triangle

def bary_to_cartesian(bary):
    """Convert barycentric coordinates to Cartesian.
    bary: shape (N, 3) with columns summing to 1
    Returns: shape (N, 2)
    """
    # Vertices of equilateral triangle
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, SQRT3 / 2.0])
    
    x = bary[:, 1] * v1[0] + bary[:, 2] * v2[0]
    y = bary[:, 1] * v1[1] + bary[:, 2] * v2[1]
    return np.column_stack((x, y))

def min_triangle_area(points):
    """Compute the minimum area among all triangles formed by 3 points.
    points: shape (N, 2)
    Returns: minimum area (float)
    """
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross product|
                dx1 = points[j, 0] - points[i, 0]
                dy1 = points[j, 1] - points[i, 1]
                dx2 = points[k, 0] - points[i, 0]
                dy2 = points[k, 1] - points[i, 1]
                area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                if area < min_area:
                    min_area = area
    return min_area

def normalize_bary(bary):
    """Ensure barycentric coordinates are valid (non-negative, sum to 1)."""
    bary = np.maximum(bary, 0.0)
    sums = bary.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    bary = bary / sums
    return bary

def hexagonal_initialization():
    """Create initial configuration using hexagonal lattice pattern."""
    np.random.seed(42)
    bary = np.zeros((N, 3))
    
    # Place points in a roughly hexagonal pattern
    # Use a grid and add small random perturbation
    positions = []
    
    # Center point
    positions.append([1/3, 1/3, 1/3])
    
    # Ring 1: 6 points around center
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    for angle in angles:
        r = 0.25
        # Convert to barycentric-like coordinates
        x = 0.5 + r * np.cos(angle)
        y = SQRT3/4 + r * np.sin(angle)
        # Convert back to barycentric
        # x = b1 + 0.5*b2, y = SQRT3/2 * b2
        b2 = y / (SQRT3/2)
        b1 = x - 0.5 * b2
        b0 = 1 - b1 - b2
        if b0 >= 0 and b1 >= 0 and b2 >= 0:
            positions.append([b0, b1, b2])
    
    # Ring 2: more points if needed
    while len(positions) < N:
        angle = random.uniform(0, 2*np.pi)
        r = random.uniform(0.3, 0.45)
        x = 0.5 + r * np.cos(angle)
        y = SQRT3/4 + r * np.sin(angle)
        b2 = y / (SQRT3/2)
        b1 = x - 0.5 * b2
        b0 = 1 - b1 - b2
        if b0 >= 0 and b1 >= 0 and b2 >= 0:
            positions.append([b0, b1, b2])
    
    bary = np.array(positions[:N])
    bary = normalize_bary(bary)
    
    # Add small random perturbation
    for i in range(N):
        bary[i] += np.random.normal(0, 0.01, 3)
    bary = normalize_bary(bary)
    
    return bary

def simulated_annealing(bary0, start_time, deadline):
    """Run simulated annealing to maximize minimum triangle area."""
    current_bary = bary0.copy()
    current_points = bary_to_cartesian(current_bary)
    current_area = min_triangle_area(current_points)
    
    best_bary = current_bary.copy()
    best_area = current_area
    
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.995
    max_steps = 50000
    
    step = 0
    while step < max_steps:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # Generate perturbation
        new_bary = current_bary.copy()
        # Randomly perturb 1-3 points
        n_perturb = random.randint(1, 3)
        idx = random.sample(range(N), n_perturb)
        for i in idx:
            new_bary[i] += np.random.normal(0, 0.02, 3)
        
        new_bary = normalize_bary(new_bary)
        new_points = bary_to_cartesian(new_bary)
        new_area = min_triangle_area(new_points)
        
        # Accept or reject
        if new_area >= current_area:
            current_bary = new_bary
            current_points = new_points
            current_area = new_area
            
            if current_area > best_area:
                best_bary = current_bary.copy()
                best_area = current_area
        else:
            # Accept with probability exp((new_area - current_area) / T)
            delta = new_area - current_area
            if random.random() < np.exp(delta / T):
                current_bary = new_bary
                current_points = new_points
                current_area = new_area
        
        # Cool down
        T = max(T * cooling_rate, T_min)
        step += 1
    
    return best_bary, best_area

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin from 120s limit
    
    # Initial configuration
    bary0 = hexagonal_initialization()
    
    # Run simulated annealing
    best_bary, best_area = simulated_annealing(bary0, start_time, deadline)
    
    # Try a few more restarts if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        for restart in range(3):
            if time.time() - start_time > deadline - 2.0:
                break
            
            # New random initialization
            bary_restart = np.random.dirichlet(np.ones(N * 3)).reshape(N, 3)
            bary_restart = normalize_bary(bary_restart)
            
            bary_res, area_res = simulated_annealing(bary_restart, start_time, deadline)
            
            if area_res > best_area:
                best_bary = bary_res
                best_area = area_res
    
    # Convert to Cartesian
    points = bary_to_cartesian(best_bary)
    
    # Verify points are inside the triangle
    # Check barycentric coordinates are non-negative
    bary_final = normalize_bary(best_bary)
    points = bary_to_cartesian(bary_final)
    
    return points
