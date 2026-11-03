"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

# Triangle vertices
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, np.sqrt(3)/2])

# Precompute combinations for speed
COMBOS = np.array(list(combinations(range(N), 3)))

def point_in_triangle(p, a=V1, b=V2, c=V3):
    """Check if point p is inside or on the triangle abc."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def min_triangle_area(points):
    """Vectorized minimum area of any triangle formed by three points."""
    p1 = points[COMBOS[:, 0]]
    p2 = points[COMBOS[:, 1]]
    p3 = points[COMBOS[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def generate_symmetric_seed(jitter=0.0):
    """
    Generate a high-quality symmetric initial configuration for 11 points.
    """
    # 1. Vertices
    pts = [V1, V2, V3]
    
    # 2. Midpoints
    m12 = (V1 + V2) / 2.0
    m23 = (V2 + V3) / 2.0
    m31 = (V3 + V1) / 2.0
    pts.extend([m12, m23, m31])
    
    # 3. Centroid
    centroid = (V1 + V2 + V3) / 3.0
    pts.append(centroid)
    
    # 4. Four sub-triangle centroids
    c1 = (V1 + m12 + m31) / 3.0
    c2 = (V2 + m12 + m23) / 3.0
    c3 = (V3 + m23 + m31) / 3.0
    c4 = (m12 + m23 + m31) / 3.0
    
    pts.extend([c1, c2, c3, c4])
    
    seed = np.array(pts)
    
    if jitter > 0:
        # Deterministic jitter based on index to break symmetry slightly
        # We use a fixed random state for reproducibility but different offsets
        rng = np.random.RandomState(42)
        seed = seed + rng.uniform(-jitter, jitter, seed.shape)
        
    # Ensure points stay inside
    for i in range(len(seed)):
        if not point_in_triangle(seed[i]):
            # Project to centroid if outside (shouldn't happen with small jitter)
            seed[i] = (V1 + V2 + V3) / 3.0
            
    return seed

def local_optimize(points, max_iters=500, step_init=0.02, step_decay=0.95):
    """
    Deterministic Hill Climbing.
    """
    current = points.copy()
    current_score = min_triangle_area(current)
    step = step_init
    
    for _ in range(max_iters):
        improved = False
        # Try moving each point
        for i in range(N):
            original = current[i].copy()
            # 8 directions
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    new_p = original + np.array([dx, dy])
                    if point_in_triangle(new_p):
                        current[i] = new_p
                        new_score = min_triangle_area(current)
                        if new_score > current_score:
                            current_score = new_score
                            improved = True
                        else:
                            current[i] = original
            # Early break if we found a better position for this point? 
            # No, we want to try all directions to find the best step? 
            # Actually, standard HC tries one move at a time. 
            # This loop tries all 8 and keeps the best one found in this pass for point i.
            # Wait, the logic above accepts the FIRST improvement found? 
            # No, it updates current_score and current[i] immediately. 
            # This means it might accept a small improvement, then try another direction from the NEW position.
            # This is a "steepest ascent" variant if we checked all and picked best, but here it's "first best".
            # Let's stick to "first best" for speed, it works well enough.
            
        if not improved:
            step *= step_decay
            if step < 1e-5:
                break
        else:
            step *= 0.99 # Slow decay if improving
    
    return current, current_score

def construct():
    # Multi-start with deterministic seeds
    best_points = None
    best_score = 0.0
    
    # Seed 1: Pure Symmetric
    seed1 = generate_symmetric_seed(0.0)
    p1, s1 = local_optimize(seed1)
    if s1 > best_score:
        best_score = s1
        best_points = p1
        
    # Seed 2: Small Jitter
    seed2 = generate_symmetric_seed(0.01)
    p2, s2 = local_optimize(seed2)
    if s2 > best_score:
        best_score = s2
        best_points = p2
        
    # Seed 3: Larger Jitter
    seed3 = generate_symmetric_seed(0.05)
    p3, s3 = local_optimize(seed3)
    if s3 > best_score:
        best_score = s3
        best_points = p3
        
    return best_points
# EVOLVE-BLOCK-END
