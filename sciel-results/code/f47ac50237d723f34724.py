import numpy as np
import time
from itertools import combinations

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600  # 600s limit, keep margin

    # Lattice basis
    # u = (sqrt(3), 0)
    # v = (sqrt(3)/2, 3/2)
    sqrt3 = np.sqrt(3)
    u = np.array([sqrt3, 0])
    v = np.array([sqrt3/2, 1.5])
    
    # Generate a set of candidate centers on the lattice
    # Range -5 to 5 in i, j
    centers_pool = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            c = i * u + j * v
            centers_pool.append(c)
    centers_pool = np.array(centers_pool)
    
    # Function to calculate minimal s for a given set of centers and outer params
    def calc_min_s(inner_centers, cx, cy, theta):
        # Inner hexagons have angle 0
        # Generate all vertices of inner hexagons
        # Vertices of unit hex at (0,0) angle 0:
        # (1, 0), (0.5, sqrt3/2), (-0.5, sqrt3/2), (-1, 0), (-0.5, -sqrt3/2), (0.5, -sqrt3/2)
        base_vertices = np.array([
            [1, 0],
            [0.5, sqrt3/2],
            [-0.5, sqrt3/2],
            [-1, 0],
            [-0.5, -sqrt3/2],
            [0.5, -sqrt3/2]
        ])
        
        all_vertices = []
        for c in inner_centers:
            all_vertices.append(base_vertices + c)
        all_vertices = np.vstack(all_vertices)
        
        # Outer hexagon normals
        # Normals at angles theta, theta+60, theta+120
        # The distance from center to side is h = s * sqrt(3)/2
        # Condition: dot(n, v - c_outer) <= h for all v
        # So s >= 2/sqrt(3) * max_k max_v dot(n_k, v - c_outer)
        
        max_proj = 0
        for k in range(3):
            angle = theta + k * np.pi / 3
            n = np.array([np.cos(angle), np.sin(angle)])
            projs = np.dot(all_vertices, n)
            # Subtract outer center projection
            c_outer_proj = np.dot(np.array([cx, cy]), n)
            max_proj_k = np.max(projs - c_outer_proj)
            if max_proj_k > max_proj:
                max_proj = max_proj_k
        
        s = (2 / sqrt3) * max_proj
        return s

    # Heuristic to find a good set of 12 centers
    # Start with 0 centers, add one by one to minimize bounding hexagon size
    # This is a greedy approach.
    
    best_centers = None
    best_s = float('inf')
    
    # Initial guess: Center at origin, then add neighbors
    current_centers = [np.array([0.0, 0.0])]
    
    # Pre-compute distances or just use brute force for small pool
    # To speed up, we only consider centers within a certain radius
    # But let's just iterate through the pool for the greedy step
    
    # Optimization: We will run a local search on the set of centers.
    # Represent a set by indices into centers_pool.
    
    # Initialize with a symmetric 7-hex cluster
    init_indices = []
    # Find index of (0,0)
    idx_00 = np.argmin(np.sum((centers_pool - np.array([0,0]))**2, axis=1))
    init_indices.append(idx_00)
    
    # Add 6 neighbors
    neighbors = [
        (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    ]
    for ni, nj in neighbors:
        target = ni * u + nj * v
        idx = np.argmin(np.sum((centers_pool - target)**2, axis=1))
        if idx not in init_indices:
            init_indices.append(idx)
            
    # Add 5 more greedily
    # We want to minimize the bounding hexagon size.
    # We will try adding each available center and pick the one that minimizes s (approximated by max distance from origin for now, or actual s)
    
    # To make it faster, let's just try a few random restarts or a simple local search.
    
    # Define a function to evaluate a set of indices
    def evaluate_indices(indices):
        if len(indices) != 12:
            return float('inf')
        cs = centers_pool[indices]
        # Optimize outer center and angle for this fixed set
        # Use scipy minimize for (cx, cy, theta)
        from scipy.optimize import minimize
        
        # Initial guess for outer center: centroid
        c0 = np.mean(cs, axis=0)
        x0 = [c0[0], c0[1], 0.0]
        
        def objective(x):
            cx, cy, theta = x
            return calc_min_s(cs, cx, cy, theta)
            
        res = minimize(objective, x0, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 1000})
        return res.fun

    # Current best
    current_indices = list(init_indices)
    # Add 5 more from the pool
    available = [i for i in range(len(centers_pool)) if i not in current_indices]
    
    # Sort available by distance from origin to prefer compactness
    dists = np.sum(centers_pool**2, axis=1)
    available_sorted = sorted(available, key=lambda i: dists[i])
    
    for i in available_sorted:
        if len(current_indices) >= 12:
            break
        current_indices.append(i)
        
    # Now we have 12.
    # Local search: swap one center with another to improve s
    current_s = evaluate_indices(current_indices)
    best_indices = list(current_indices)
    
    iterations = 0
    while time.time() < deadline - 60: # 60s margin
        improved = False
        # Try swapping each center with a random other center
        for i in range(12):
            if time.time() > deadline - 60:
                break
            # Pick a random replacement
            j = np.random.randint(len(centers_pool))
            if j in current_indices:
                continue
            
            new_indices = list(current_indices)
            new_indices[i] = j
            
            new_s = evaluate_indices(new_indices)
            if new_s < current_s - 1e-6:
                current_s = new_s
                current_indices = new_indices
                improved = True
                if current_s < best_s:
                    best_s = current_s
                    best_indices = list(current_indices)
        
        if not improved:
            # Random restart if stuck
            # Generate a new random set of 12
            random_indices = np.random.choice(len(centers_pool), 12, replace=False)
            current_indices = list(random_indices)
            current_s = evaluate_indices(current_indices)
            if current_s < best_s:
                best_s = current_s
                best_indices = list(current_indices)
            iterations += 1
            if iterations > 50: # Limit restarts
                break

    # Final optimization on the best set
    best_cs = centers_pool[best_indices]
    from scipy.optimize import minimize
    c0 = np.mean(best_cs, axis=0)
    x0 = [c0[0], c0[1], 0.0]
    
    def objective_final(x):
        cx, cy, theta = x
        return calc_min_s(best_cs, cx, cy, theta)
        
    res = minimize(objective_final, x0, method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 2000})
    final_s = res.fun
    cx, cy, theta = res.x
    
    # Construct return arrays
    inner_hex_data = np.zeros((12, 3))
    for i, c in enumerate(best_cs):
        inner_hex_data[i] = [c[0], c[1], 0.0]
        
    outer_hex_data = np.array([cx, cy, np.degrees(theta)])
    
    return inner_hex_data, outer_hex_data, final_s

if __name__ == "__main__":
    inner, outer, s = hexagon_packing_12()
    print(f"s = {s}")
