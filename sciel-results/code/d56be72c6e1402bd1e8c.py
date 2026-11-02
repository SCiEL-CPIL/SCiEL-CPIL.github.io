import numpy as np
import time
import random

N = 11
SQRT3 = np.sqrt(3)

def bary_to_cart(bary):
    """
    Convert barycentric coordinates (u, v, w) to Cartesian (x, y).
    bary shape: (N, 3)
    Vertices: A=(0,0), B=(1,0), C=(0.5, SQRT3/2)
    Point = u*A + v*B + w*C
    """
    u, v, w = bary[:, 0], bary[:, 1], bary[:, 2]
    x = v * 1.0 + w * 0.5
    y = w * (SQRT3 / 2)
    return np.column_stack((x, y))

def min_area_cartesian(points):
    """
    Calculate the minimum area of any triangle formed by 3 points.
    points shape: (N, 2)
    """
    n = len(points)
    min_a = float('inf')
    # Vectorized calculation for all pairs is possible but N=11 is small.
    # We can precompute cross products or just loop.
    # Loop is fast enough for SA steps (millions of calls).
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx_ij = xj - xi
            dy_ij = yj - yi
            for k in range(j + 1, n):
                xk, yk = points[k]
                # Area = 0.5 * |cross(Pj-Pi, Pk-Pi)|
                cross = dx_ij * (yk - yi) - dy_ij * (xk - xi)
                area = 0.5 * abs(cross)
                if area < min_a:
                    min_a = area
    return min_a

def normalize_bary(bary):
    """Ensure barycentric coordinates are valid (sum to 1, non-negative)."""
    # Clip negative values
    bary = np.maximum(bary, 0)
    sums = bary.sum(axis=1, keepdims=True)
    # Avoid division by zero
    sums[sums == 0] = 1
    return bary / sums

def construct():
    start_time = time.time()
    deadline = 115.0 # Safety margin
    
    # Initial configuration: Random points inside the triangle
    # Use Dirichlet distribution for valid barycentric coordinates
    np.random.seed(42)
    current_bary = np.random.dirichlet(np.ones(3), size=N)
    current_points = bary_to_cart(current_bary)
    current_min_area = min_area_cartesian(current_points)
    
    best_bary = current_bary.copy()
    best_min_area = current_min_area
    
    # Simulated Annealing parameters
    T_start = 0.01
    T_end = 1e-5
    cooling_rate = 0.999
    
    T = T_start
    steps = 0
    no_improve_count = 0
    max_no_improve = 5000 # Reset temperature if stuck? Or just keep going.
    
    while time.time() - start_time < deadline:
        # Perturb a random point
        idx = np.random.randint(N)
        # Random step size
        step_size = 0.05
        delta = np.random.uniform(-step_size, step_size, size=3)
        
        new_bary = current_bary.copy()
        new_bary[idx] += delta
        
        # Project back to valid barycentric space
        # If any coord is negative, clip and renormalize
        # This projection might move the point more than intended, 
        # but it's a standard way to handle constraints in SA.
        if np.any(new_bary[idx] < 0):
             # Simple rejection or projection? 
             # Projection: clip to 0, then normalize.
             new_bary[idx] = np.maximum(new_bary[idx], 0)
             s = new_bary[idx].sum()
             if s == 0:
                 new_bary[idx] = current_bary[idx] # Revert if collapsed
             else:
                 new_bary[idx] /= s
        else:
             # Even if positive, ensure it sums to 1? 
             # Dirichlet space is a simplex. Adding a vector might break sum=1.
             # Actually, barycentric coords must sum to 1.
             # My perturbation added a vector that doesn't necessarily sum to 0.
             # Better perturbation: subtract a random vector that sums to 0?
             # Or just normalize after perturbation.
             # Let's stick to: perturb, then normalize.
             pass
        
        # Ensure sum is 1
        s = new_bary[idx].sum()
        if s > 0:
            new_bary[idx] /= s
        else:
            continue # Skip invalid move
        
        new_points = bary_to_cart(new_bary)
        new_min_area = min_area_cartesian(new_points)
        
        delta_area = new_min_area - current_min_area
        
        if delta_area > 0 or np.random.rand() < np.exp(delta_area / T):
            # Accept move
            current_bary = new_bary
            current_points = new_points
            current_min_area = new_min_area
            
            if current_min_area > best_min_area:
                best_min_area = current_min_area
                best_bary = current_bary.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Cool down
            T *= cooling_rate
            if T < T_end:
                T = T_end
                
        else:
            no_improve_count += 1
            
        steps += 1
        # Optional: Reheat if stuck for too long?
        # For now, just let it cool.
        
    return bary_to_cart(best_bary)
