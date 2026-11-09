import numpy as np
import time
import itertools

N = 11
h = np.sqrt(3) / 2
# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, h])

# Precompute all triplets for vectorized area computation
_TRIPLETS = np.array(list(itertools.combinations(range(N), 3)), dtype=np.int32)

def barycentric_to_cartesian(bary):
    """
    bary: array of shape (N, 3) where each row sums to 1 and is non-negative.
    Returns Cartesian coordinates.
    """
    # Cartesian = b0*V0 + b1*V1 + b2*V2
    # Since V0 is (0,0), we only need b1*V1 + b2*V2
    # V1 = (1, 0), V2 = (0.5, h)
    # x = b1*1 + b2*0.5
    # y = b1*0 + b2*h
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = h * bary[:, 2]
    return np.column_stack((x, y))

def min_triangle_area(points):
    """
    points: (N, 2)
    Returns the minimum area of any triangle formed by 3 points.
    """
    p1 = points[_TRIPLETS[:, 0]]
    p2 = points[_TRIPLETS[:, 1]]
    p3 = points[_TRIPLETS[:, 2]]
    
    v12 = p2 - p1
    v13 = p3 - p1
    
    cross = v12[:, 0] * v13[:, 1] - v12[:, 1] * v13[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer for 150s limit
    
    # We will work in barycentric coordinates to ensure points stay inside the triangle.
    # To reduce dimensionality and exploit symmetry, we can fix the "center" of mass 
    # or specific points. However, a robust approach is to optimize all N points in barycentric space 
    # but use a symmetry-reduced representation if we assume the optimal solution is symmetric.
    # 
    # Actually, for N=11, the optimal configuration is likely not perfectly symmetric in a simple way 
    # (e.g., 11 is not a triangular number). But we can still use barycentric coords for the whole set.
    # 
    # Strategy: Simulated Annealing in Barycentric Space.
    # State: B (N, 3)
    # Move: Perturb B, renormalize each row to sum to 1, clip negatives to 0 and renormalize again.
    # Objective: Maximize min_area.
    
    rng = np.random.default_rng(42)
    
    # Initialize Barycentric Coordinates
    # A good starting point is a grid or random points mapped to barycentric.
    # Let's start with a symmetric grid-like pattern in Cartesian, then convert to Barycentric.
    # Or just random valid barycentric coordinates.
    
    def random_barycentric(n):
        # Generate random points in the simplex
        # Method: Generate n+1 random numbers, normalize
        # Or: Generate uniform in [0,1]^n, subtract cumulative sum, normalize
        # Simple: Generate random in [0,1]^3, normalize
        b = rng.random((n, 3))
        b = b / b.sum(axis=1, keepdims=True)
        return b

    # Initial State
    # Try a few random initializations and pick the best
    best_bary = None
    best_score = -1.0
    
    # Initialize with a structured pattern to give SA a head start
    # Pattern: Rows of points
    # Let's create a structured initial guess in Cartesian, then convert to Barycentric
    # This is often better than pure random.
    
    # Convert Cartesian to Barycentric
    # x = b1 + 0.5 b2
    # y = h b2  => b2 = y/h
    # b1 = x - 0.5 b2
    # b0 = 1 - b1 - b2
    
    def cartesian_to_barycentric(pts):
        b2 = pts[:, 1] / h
        b1 = pts[:, 0] - 0.5 * b2
        b0 = 1.0 - b1 - b2
        # Clip small negatives due to floating point
        b = np.column_stack((b0, b1, b2))
        b[b < 0] = 0
        # Renormalize just in case
        b = b / b.sum(axis=1, keepdims=True)
        return b

    # Initial Guess 1: Grid
    # 4 rows: 5, 3, 2, 1
    y_levels = [0.1*h, 0.3*h, 0.5*h, 0.7*h]
    counts = [5, 3, 2, 1]
    pts_init = []
    for y, c in zip(y_levels, counts):
        x_min = y / np.sqrt(3)
        x_max = 1.0 - y / np.sqrt(3)
        if c > 1:
            margin = 0.05 * (x_max - x_min)
            xs = np.linspace(x_min + margin, x_max - margin, c)
        else:
            xs = np.array([(x_min + x_max) / 2])
        for x in xs:
            pts_init.append([x, y])
    pts_init = np.array(pts_init)
    bary_init = cartesian_to_barycentric(pts_init)
    
    current_bary = bary_init.copy()
    current_pts = barycentric_to_cartesian(current_bary)
    current_score = min_triangle_area(current_pts)
    best_score = current_score
    best_bary = current_bary.copy()

    # Simulated Annealing Parameters
    T = 0.01 # Initial Temperature
    T_min = 1e-6
    alpha = 0.999 # Cooling rate
    max_iter = 100000 # Max iterations, but we check time
    
    # We need to be careful with the step size. 
    # In barycentric coords, a perturbation of size sigma in the simplex.
    sigma = 0.05
    
    for i in range(max_iter):
        if time.time() > deadline - 2.0:
            break
            
        # Generate new candidate
        new_bary = current_bary + rng.normal(0, sigma, (N, 3))
        
        # Enforce simplex constraints: non-negative and sum to 1
        # 1. Clip negatives
        new_bary[new_bary < 0] = 0
        # 2. Normalize
        sums = new_bary.sum(axis=1, keepdims=True)
        # Avoid division by zero (if all zeros, which is unlikely with noise)
        sums[sums == 0] = 1.0
        new_bary = new_bary / sums
        
        # Evaluate
        new_pts = barycentric_to_cartesian(new_bary)
        new_score = min_triangle_area(new_pts)
        
        # Acceptance Criterion
        if new_score > current_score:
            # Always accept improvement
            current_bary = new_bary
            current_score = new_score
            current_pts = new_pts
            
            if current_score > best_score:
                best_score = current_score
                best_bary = current_bary.copy()
        else:
            # Accept with probability exp((new_score - current_score) / T)
            # Note: We are maximizing, so delta = new - current is negative.
            # Prob = exp(delta / T)
            delta = new_score - current_score
            if delta > 0: # Safety, shouldn't happen
                current_bary = new_bary
                current_score = new_score
                current_pts = new_pts
                if current_score > best_score:
                    best_score = current_score
                    best_bary = current_bary.copy()
            else:
                # Avoid log(0) or negative T
                if T > 0:
                    prob = np.exp(delta / T)
                    if rng.random() < prob:
                        current_bary = new_bary
                        current_score = new_score
                        current_pts = new_pts
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            # If stuck at T_min, maybe restart or increase sigma? 
            # Let's just keep going, the noise is small.
            
        # Dynamic Sigma: If we've accepted many moves, maybe increase sigma to explore?
        # If we've rejected many, decrease?
        # Simple heuristic: Every 1000 steps, slightly reduce sigma to converge
        if i % 1000 == 0:
            sigma *= 0.95

    # Final Polish: Local search with smaller steps
    if time.time() < deadline - 1.0:
        # Run a short, fine-grained SA on the best point found
        T = 1e-4
        sigma = 0.01
        current_bary = best_bary.copy()
        current_pts = barycentric_to_cartesian(current_bary)
        current_score = min_triangle_area(current_pts)
        
        for i in range(20000):
            if time.time() > deadline - 0.5:
                break
            new_bary = current_bary + rng.normal(0, sigma, (N, 3))
            new_bary[new_bary < 0] = 0
            sums = new_bary.sum(axis=1, keepdims=True)
            sums[sums == 0] = 1.0
            new_bary = new_bary / sums
            
            new_pts = barycentric_to_cartesian(new_bary)
            new_score = min_triangle_area(new_pts)
            
            if new_score > current_score:
                current_bary = new_bary
                current_score = new_score
                current_pts = new_pts
                if current_score > best_score:
                    best_score = current_score
                    best_bary = current_bary.copy()
            else:
                delta = new_score - current_score
                if T > 0:
                    prob = np.exp(delta / T)
                    if rng.random() < prob:
                        current_bary = new_bary
                        current_score = new_score
                        current_pts = new_pts
            T *= 0.999
            if T < 1e-6:
                T = 1e-6

    return barycentric_to_cartesian(best_bary)

# EVOLVE-BLOCK-END
