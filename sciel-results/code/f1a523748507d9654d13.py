"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull

def _compute_score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    # Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area < 1e-10:
        return 0.0
        
    # Min Triangle Area
    # Vectorized calculation for speed
    n = len(points)
    min_area = float('inf')
    # Pre-compute vectors? No, just loop. N=14 is small.
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Cross product magnitude
                # (p2-p1) x (p3-p1)
                dx1 = points[j, 0] - points[i, 0]
                dy1 = points[j, 1] - points[i, 1]
                dx2 = points[k, 0] - points[i, 0]
                dy2 = points[k, 1] - points[i, 1]
                cross = dx1 * dy2 - dy1 * dx2
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
                    if min_area < 1e-10:
                        return 0.0
    return min_area / hull_area

def construct():
    """
    Uses Simulated Annealing to maximize the minimum triangle area 
    normalized by the convex hull area.
    """
    rng = np.random.RandomState(42)
    
    # Initial configuration: Jittered grid to break symmetry and provide good spread
    # 14 points. A 4x4 grid has 16 points, so we can take 14 from a jittered 4x4 grid
    # or use a random initialization. Let's try a mix: start with regular polygon 
    # but jitter it, then SA.
    
    # Regular polygon is a good starting point for convex hull area maximization
    angles = 2 * np.pi * np.arange(N) / N
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    current_score = _compute_score(points)
    best_points = points.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 0.1  # Initial temperature
    T_end = 1e-5
    cooling = 0.999
    max_iter = 20000
    move_scale = 0.1
    
    for i_iter in range(max_iter):
        # Pick a random point to move
        idx = rng.randint(N)
        
        # Generate a random move
        # Use a Gaussian move for smoothness
        move = rng.normal(0, move_scale, size=2)
        
        new_points = points.copy()
        new_points[idx] = points[idx] + move
        
        # Optional: Keep points within a bounding box to prevent drift
        # The score normalizes by hull area, so scaling doesn't matter, 
        # but translation does not matter either. 
        # However, if points fly apart, hull area grows, min area might grow slower?
        # Actually, if we scale the whole set, ratio is invariant. 
        # Translation is invariant.
        # So we can center the points to avoid numerical issues or drift.
        new_points -= np.mean(new_points, axis=0)
        
        new_score = _compute_score(new_points)
        
        delta = new_score - current_score
        
        # Acceptance criterion
        if delta > 0 or rng.random() < np.exp(delta / T):
            points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        
        # Cool down
        T *= cooling
        
        # Adaptive move scale: decrease as we cool
        move_scale = 0.1 * (T / 0.1)
        
        if T < T_end:
            break
            
    return best_points
# EVOLVE-BLOCK-END
