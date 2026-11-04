"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import math

def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    # Optimization: Precompute cross products or just loop efficiently
    # N=14 is small, so O(N^3) is fine.
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            v1 = p2 - p1
            for k in range(j + 1, n):
                p3 = points[k]
                v2 = p3 - p1
                # Cross product magnitude
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _halton_sequence(n, base):
    """Generate n points in [0, 1) using the Halton sequence."""
    result = []
    for i in range(1, n + 1):
        f = 1.0
        h = 0.0
        i_val = i
        while i_val > 0:
            f /= base
            i_val, rem = divmod(i_val, base)
            h += f * rem
        result.append(h)
    return result

def construct():
    """
    Place 14 points in a planar convex region to maximize the minimum triangle area
    divided by the convex hull area.
    
    Strategy: Simulated Annealing with Halton sequence initialization.
    """
    rng = np.random.RandomState(42)
    best_points_global = None
    best_score_global = -1.0
    
    # Number of restarts to escape local minima
    n_restarts = 10
    
    for restart in range(n_restarts):
        # Initialize using Halton sequence for good spread
        # Base 2 and 3 for x and y
        x_coords = _halton_sequence(N, 2)
        y_coords = _halton_sequence(N, 3)
        
        # Add some randomness to avoid getting stuck in specific patterns
        # Scale to roughly unit square, then we can let SA expand/contract if needed?
        # Actually, the score is scale-invariant (ratio). 
        # But the convex hull area changes.
        # We just need a good relative configuration.
        
        points = np.column_stack([x_coords, y_coords])
        
        # Optional: Random rotation and scaling for variety
        angle = rng.uniform(0, 2 * np.pi)
        scale = rng.uniform(0.5, 1.5)
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]])
        points = (points - 0.5) * scale
        points = points @ rot_matrix.T
        points += 0.5 # Center in [0,1] roughly
        
        current_score = _score(points)
        best_points = points.copy()
        best_score = current_score
        
        # Simulated Annealing Parameters
        T = 1.0  # Initial temperature
        cooling_rate = 0.9995
        min_T = 1e-4
        step_size = 0.1
        max_iterations = 5000 # Limit iterations per restart to stay within time
        
        iteration = 0
        while T > min_T and iteration < max_iterations:
            # Pick a random point to move
            i = rng.randint(N)
            
            # Determine move magnitude (adaptive step size)
            # If we've been stuck, maybe reduce step size? 
            # For now, fixed step size with SA acceptance is standard.
            move = rng.uniform(-step_size, step_size, 2)
            
            new_points = points.copy()
            new_points[i] = points[i] + move
            
            # Keep points within a reasonable bound to avoid numerical issues?
            # The score is invariant to translation and scaling, 
            # but large coordinates can cause precision issues.
            # Let's re-center and scale if they drift too far.
            # Actually, let's just let them move. The ratio handles scale.
            
            new_score = _score(new_points)
            
            # Acceptance criterion
            if new_score > best_score:
                points = new_points
                current_score = new_score
                best_score = new_score
                best_points = points.copy()
            else:
                # Accept worse solution with probability exp((new_score - current_score) / T)
                # Note: We are maximizing score, so delta = new - current.
                # If new < current, delta is negative.
                delta = new_score - current_score
                if delta < 0:
                    prob = math.exp(delta / T)
                    if rng.random() < prob:
                        points = new_points
                        current_score = new_score
            
            # Cool down
            T *= cooling_rate
            iteration += 1
            
        # After SA, do a local refinement (Hill Climbing)
        # Small steps to polish the solution
        for _ in range(200):
            improved = False
            for i in range(N):
                for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                    for scale in [0.01, 0.005, 0.001]:
                        dx = scale * np.cos(angle)
                        dy = scale * np.sin(angle)
                        new_points = points.copy()
                        new_points[i] = points[i] + np.array([dx, dy])
                        new_score = _score(new_points)
                        if new_score > current_score:
                            points = new_points
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_points = points.copy()
                            break
                    if improved:
                        break
            if not improved:
                break
        
        if best_score > best_score_global:
            best_score_global = best_score
            best_points_global = best_points.copy()
            
    return best_points_global
# EVOLVE-BLOCK-END
