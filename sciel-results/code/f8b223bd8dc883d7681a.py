import numpy as np
from scipy.spatial import ConvexHull
import random
import time

N = 13

# EVOLVE-BLOCK-START
def construct():
    # 1. Initialization: Fermat Spiral (Vogel's model)
    # This provides a very uniform distribution in a circular domain.
    n = N
    i = np.arange(n)
    r = np.sqrt(i / n)
    theta = i * 2.399963229728653  # Golden angle
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    points = np.column_stack((x, y))
    
    # Center the points
    points -= np.mean(points, axis=0)
    
    # 2. Precompute triangle indices
    tri_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    
    def get_score(pts):
        # Calculate convex hull area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume # In 2D, volume is area
            if hull_area <= 1e-10:
                return 0.0
        except:
            return 0.0
            
        # Calculate all triangle areas
        # p1, p2, p3
        p1 = pts[tri_indices[:, 0]]
        p2 = pts[tri_indices[:, 1]]
        p3 = pts[tri_indices[:, 2]]
        
        # Area = 0.5 * | (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) |
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        
        return min_area / hull_area

    current_score = get_score(points)
    best_points = points.copy()
    best_score = current_score
    
    # 3. Simulated Annealing
    # Time limit check
    start_time = time.time()
    max_time = 100  # Leave 20s buffer for Python overhead
    
    T = 0.01  # Initial temperature
    cooling_rate = 0.999
    min_T = 1e-5
    
    while T > min_T and (time.time() - start_time) < max_time:
        # Generate a new candidate
        new_points = points.copy()
        
        # Strategy: Move points involved in the worst triangle, or random points
        # To find the worst triangle, we need the areas again. 
        # For efficiency in the loop, we might just pick a random point or 
        # recompute the worst triangle only every few steps.
        # Given N=13, recomputing is fast enough.
        
        # Find the worst triangle index
        p1 = points[tri_indices[:, 0]]
        p2 = points[tri_indices[:, 1]]
        p3 = points[tri_indices[:, 2]]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        # Try to improve the worst triangle
        worst_idx = np.argmin(areas)
        worst_tri = tri_indices[worst_idx]
        
        # Pick a point from the worst triangle to move
        # Or pick a random point if T is high
        if random.random() < 0.7:
            point_to_move = worst_tri[random.randint(0, 2)]
        else:
            point_to_move = random.randint(0, n - 1)
            
        # Random perturbation
        # Step size proportional to T
        step_size = 0.1 * T + 0.001
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        
        new_points[point_to_move, 0] += dx
        new_points[point_to_move, 1] += dy
        
        new_score = get_score(new_points)
        
        # Acceptance criterion
        if new_score >= current_score:
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            # Metropolis criterion
            delta = current_score - new_score
            if delta < 0:
                points = new_points
                current_score = new_score
            else:
                # Probability of accepting worse solution
                p = np.exp(-delta / T)
                if random.random() < p:
                    points = new_points
                    current_score = new_score
        T *= cooling_rate
        
    # Final polish: Simple local search to ensure we are at a local maximum
    # Just a few iterations of moving the worst triangle points towards better positions
    for _ in range(1000):
        p1 = points[tri_indices[:, 0]]
        p2 = points[tri_indices[:, 1]]
        p3 = points[tri_indices[:, 2]]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        worst_idx = np.argmin(areas)
        worst_tri = tri_indices[worst_idx]
        
        improved = False
        for pt_idx in worst_tri:
            for _ in range(10):
                new_points = points.copy()
                step = 0.01
                dx = random.gauss(0, step)
                dy = random.gauss(0, step)
                new_points[pt_idx, 0] += dx
                new_points[pt_idx, 1] += dy
                new_score = get_score(new_points)
                if new_score > current_score:
                    points = new_points
                    current_score = new_score
                    improved = True
                    break
        if not improved:
            break
            
    if current_score > best_score:
        best_points = points.copy()
        best_score = current_score

    return best_points
# EVOLVE-BLOCK-END
