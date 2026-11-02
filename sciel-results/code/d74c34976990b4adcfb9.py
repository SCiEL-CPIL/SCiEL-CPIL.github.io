"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _min_triangle_area_over_hull(points):
    """
    Compute min triangle area / convex hull area for given points.
    Returns the ratio (higher is better).
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0
    
    if hull_area <= 1e-10:
        return 0.0
    
    # Vectorized triangle area calculation
    # Triangles are defined by indices i, j, k with i < j < k
    # We can generate all combinations or just loop since 364 is small.
    # For N=14, loop is fine.
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
                    if min_area < 1e-10:
                        return 0.0 # Early exit for degenerate cases
    
    return min_area / hull_area


def construct():
    """
    Uses Simulated Annealing to maximize the minimum triangle area ratio.
    """
    rng = np.random.RandomState(42)
    
    # Initial configuration: Regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize initial points to have hull area 1.0 to start in a "good" scale
    # Although the score is scale invariant, starting at a reasonable scale helps SA.
    try:
        hull = ConvexHull(points)
        area = hull.volume
        if area > 0:
            points /= np.sqrt(area)
    except:
        pass
        
    current_score = _min_triangle_area_over_hull(points)
    best_points = points.copy()
    best_score = current_score
    
    # SA Parameters
    T_start = 0.05  # Initial temperature
    T_end = 1e-5    # Final temperature
    num_iterations = 50000
    cooling_rate = (T_end / T_start) ** (1.0 / num_iterations)
    
    current_T = T_start
    
    for it in range(num_iterations):
        # Generate a new candidate
        # Perturb a random subset of points (1 or 2)
        num_perturb = rng.randint(1, 3) # 1 or 2 points
        indices = rng.choice(N, size=num_perturb, replace=False)
        
        new_points = points.copy()
        for idx in indices:
            # Gaussian noise
            noise = 0.05 * rng.randn(2)
            new_points[idx] += noise
            
        # Calculate score for new points
        new_score = _min_triangle_area_over_hull(new_points)
        
        # Metropolis acceptance criterion
        delta = new_score - current_score
        if delta > 0:
            # Always accept improvement
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            # Accept worse solution with probability exp(delta / T)
            if delta < -100 * current_T:
                # Avoid underflow in exp
                accept_prob = 0.0
            else:
                accept_prob = np.exp(delta / current_T)
            if rng.rand() < accept_prob:
                points = new_points
                current_score = new_score
                
        # Cool down
        current_T *= cooling_rate
        
    return best_points
# EVOLVE-BLOCK-END
