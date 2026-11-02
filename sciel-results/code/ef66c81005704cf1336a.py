"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _min_triangle_area_vectorized(points):
    """
    Compute the minimum area over all triples of points using vectorized NumPy operations.
    points: shape (N, 2)
    """
    n = len(points)
    # Generate all unique triples (i, j, k) with i < j < k
    # Number of triples is N*(N-1)*(N-2)/6. For N=14, this is 364.
    # We can create arrays of indices.
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_vals = i_idx[mask]
    j_vals = j_idx[mask]
    k_vals = k_idx[mask]
    
    # Get coordinates
    p1 = points[i_vals]
    p2 = points[j_vals]
    p3 = points[k_vals]
    
    # Calculate 2*Area using cross product determinant
    # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    # Let v1 = p2 - p1, v2 = p3 - p1
    # Cross = v1_x * v2_y - v1_y * v2_x
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    try:
        min_area = _min_triangle_area_vectorized(points)
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
        return min_area / hull_area
    except Exception:
        return 0.0

def _differential_evolution(points, n_iter=200, pop_size=50, F=0.8, CR=0.9):
    """
    Run Differential Evolution on the initial population.
    """
    n = len(points)
    
    # Initialize population
    # Use a mix of random and structured initialization to avoid local minima
    pop = []
    for i in range(pop_size):
        if i == 0:
            p = points.copy()
        elif i == 1:
            # Regular polygon on unit circle, scaled
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            p = np.column_stack([0.5 + 0.4*np.cos(angles), 0.5 + 0.4*np.sin(angles)])
        elif i == 2:
            # Grid-like
            xs = np.linspace(0.1, 0.9, 4)
            ys = np.linspace(0.1, 0.9, 4)
            grid = np.array([[x, y] for x in xs for y in ys])
            np.random.shuffle(grid)
            p = grid[:n].copy()
            p += np.random.normal(0, 0.1, (n, 2))
            p = np.clip(p, 0.05, 0.95)
        else:
            p = np.random.uniform(0.1, 0.9, (n, 2))
        pop.append(p)
        
    # Evaluate initial population
    scores = [_score(p) for p in pop]
    
    # Main loop
    for g in range(n_iter):
        for i in range(pop_size):
            # Choose 3 distinct random indices different from i
            while True:
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                if len(set([r1, r2, r3, i])) == 4:
                    break
            
            # Mutation: v = x_r1 + F * (x_r2 - x_r3)
            v = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Crossover: Trial vector u
            u = np.copy(pop[i])
            j_rand = np.random.randint(n)
            for j in range(n):
                if np.random.random() < CR or j == j_rand:
                    u[j] = v[j]
            
            # Clip to bounds [0, 1]
            u = np.clip(u, 0.0, 1.0)
            
            # Selection
            u_score = _score(u)
            if u_score > scores[i]:
                pop[i] = u
                scores[i] = u_score
                
    # Return best individual
    best_idx = np.argmax(scores)
    return pop[best_idx]

def construct():
    """
    Use Differential Evolution to find 14 points that maximize the minimum triangle area
    relative to the convex hull area.
    """
    start_time = time.time()
    time_limit = 115.0 # Leave 5s margin
    
    best_points = None
    best_score = -1.0
    
    # Initial guess: regular polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    init_points = np.column_stack([0.5 + 0.45*np.cos(angles), 0.5 + 0.45*np.sin(angles)])
    
    # Run DE with multiple restarts if time permits
    restarts = 0
    while time.time() - start_time < time_limit:
        restarts += 1
        
        # Vary initial points slightly for diversity in restarts
        if restarts == 1:
            current_init = init_points.copy()
        else:
            # Add noise to best found or random
            if best_points is not None:
                current_init = best_points + np.random.normal(0, 0.1, (N, 2))
            else:
                current_init = np.random.uniform(0.1, 0.9, (N, 2))
            current_init = np.clip(current_init, 0.05, 0.95)
            
        # Adjust n_iter based on time remaining
        time_elapsed = time.time() - start_time
        time_remaining = time_limit - time_elapsed
        # Estimate: DE iteration takes ~pop_size * (eval_time)
        # eval_time for N=14 vectorized is very fast, maybe 1-2ms?
        # Let's estimate 50 evaluations * 2ms = 100ms per gen.
        # 1000 gens = 100s.
        # We have time_remaining seconds.
        # Let's be conservative: 100 gens per 5 seconds?
        # Let's just run a fixed number and check.
        n_iter = 300
        pop_size = 40
        
        res_points = _differential_evolution(current_init, n_iter=n_iter, pop_size=pop_size)
        res_score = _score(res_points)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_points.copy()
            
        # Check time
        if time.time() - start_time > time_limit:
            break
            
    if best_points is None:
        best_points = init_points
        
    return best_points

# EVOLVE-BLOCK-END
