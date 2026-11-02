import numpy as np
import time

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triplets
triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            triplets.append((i, j, k))
triplets = np.array(triplets)

def evaluate(points):
    """
    Calculate the minimum area of all triangles formed by the points.
    points: np.array of shape (N, 2)
    """
    A = points[triplets[:, 0]]
    B = points[triplets[:, 1]]
    C = points[triplets[:, 2]]
    AB = B - A
    AC = C - A
    # Cross product in 2D
    areas = 0.5 * np.abs(AB[:, 0] * AC[:, 1] - AB[:, 1] * AC[:, 0])
    return np.min(areas)

def random_point_in_triangle(rng):
    """
    Generate a uniform random point inside the triangle defined by V0, V1, V2.
    Uses barycentric coordinates (u, v) where P = u*V1 + v*V2, u>=0, v>=0, u+v<=1.
    """
    u = rng.random()
    v = rng.random()
    if u + v > 1.0:
        u = 1.0 - u
        v = 1.0 - v
    return u * V1 + v * V2

def construct():
    # Seed for reproducibility
    rng = np.random.RandomState(42)
    
    # Time management
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for evaluator overhead
    
    # 1. Initialization: Find a good starting point
    best_points = None
    best_score = 0.0
    
    # Try a few random initial configurations
    for _ in range(50):
        pts = np.array([random_point_in_triangle(rng) for _ in range(N)])
        score = evaluate(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            
    # Also try a grid-like initialization
    # Generate points on a triangular grid
    # For N=11, we can try to place them roughly evenly
    # This is a heuristic start
    grid_pts = []
    # 3x3 grid in barycentric (0, 1/3, 2/3) -> 10 points?
    # Let's just stick to random, SA is powerful.
    
    # 2. Simulated Annealing
    current_points = best_points.copy()
    current_score = best_score
    
    # Temperature tuning
    # Estimate typical improvement step size
    # If best_score is ~0.01, T should be comparable.
    T = best_score * 0.1
    if T < 1e-6: T = 1e-6
    
    alpha = 0.9995  # Cooling rate
    
    # To avoid getting stuck in local minima, we can occasionally restart T
    # or just rely on the long run.
    
    while time.time() < deadline:
        # Pick a random point to perturb
        idx = rng.randint(N)
        
        # Generate new position
        new_pt = random_point_in_triangle(rng)
        
        # Save old state
        old_pt = current_points[idx].copy()
        current_points[idx] = new_pt
        
        # Evaluate new state
        new_score = evaluate(current_points)
        
        # Acceptance criterion
        if new_score >= current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = current_points.copy()
        else:
            # Calculate probability of accepting worse solution
            delta = new_score - current_score
            # Avoid division by zero or overflow
            if T > 1e-12:
                # exp(delta/T) where delta < 0
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_score = new_score
                else:
                    # Reject
                    current_points[idx] = old_pt
            else:
                # T is effectively 0, reject
                current_points[idx] = old_pt
                
        # Cool down
        T *= alpha
        
    return best_points
