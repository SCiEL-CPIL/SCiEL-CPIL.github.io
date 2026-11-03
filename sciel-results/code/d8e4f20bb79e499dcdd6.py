"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum triangle area divided by the convex hull area."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0
    
    if hull_area <= 1e-12:
        return 0.0
    
    # Compute minimum triangle area
    min_tri_area = float('inf')
    # Vectorized calculation for speed
    # points shape: (n, 2)
    # We need all combinations of 3 points.
    # For n=13, 286 triangles. A loop is fast enough, but let's try to be efficient.
    
    # Using a loop is often faster for small n due to overhead of creating index arrays
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            # Vectorized for k > j
            p3s = points[j+1:]
            if len(p3s) == 0: continue
            # Area = 0.5 * | (p2-p1) x (p3-p1) |
            # (p2-p1) is (2,), (p3-p1) is (m, 2)
            # Cross product in 2D: x1*y2 - y1*x2
            v1 = p2 - p1
            v3s = p3s - p1
            areas = 0.5 * np.abs(v1[0]*v3s[:,1] - v1[1]*v3s[:,0])
            if len(areas) > 0:
                min_val = np.min(areas)
                if min_val < min_tri_area:
                    min_tri_area = min_val
    
    if min_tri_area == float('inf'):
        return 0.0
        
    return min_tri_area / hull_area


def _regular_polygon(n, radius=1.0, rotation=0.0):
    """Generate a regular polygon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack((x, y))


def _local_search_worst_case(points, max_iter=2000, step_size=0.01, rng_seed=42):
    """
    Local search that specifically targets the minimum area triangles.
    """
    rng = np.random.RandomState(rng_seed)
    current = points.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    n = len(current)
    
    for i in range(max_iter):
        # Find the worst triangle indices
        # We need to identify which triangle has the min area
        min_area = float('inf')
        worst_idx = (0, 1, 2)
        
        # Re-compute to find indices (slightly expensive but necessary for targeted moves)
        # Optimization: We can just pick random points and move them, 
        # but targeting the worst is better.
        
        # To save time, we don't find the exact worst index every time if it's too slow.
        # But for N=13, it's fast.
        for i_idx in range(n):
            p1 = current[i_idx]
            for j_idx in range(i_idx + 1, n):
                p2 = current[j_idx]
                v1 = p2 - p1
                for k_idx in range(j_idx + 1, n):
                    p3 = current[k_idx]
                    v2 = p3 - p1
                    area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                    if area < min_area:
                        min_area = area
                        worst_idx = (i_idx, j_idx, k_idx)
        
        # Try to improve the worst triangle by moving one of its vertices
        # Pick a random vertex from the worst triangle
        vertex_to_move = worst_idx[rng.randint(0, 3)]
        
        # Generate a move
        # Direction: random, or specifically away from the other two points
        other1, other2 = [v for v in worst_idx if v != vertex_to_move]
        
        # Try a few random small steps
        improved = False
        for _ in range(10):
            new_points = current.copy()
            # Random perturbation
            noise = rng.normal(0, step_size, 2)
            new_points[vertex_to_move] += noise
            
            new_score = _compute_score(new_points)
            
            if new_score > current_score:
                current = new_points
                current_score = new_score
                if current_score > best_score:
                    best = current.copy()
                    best_score = current_score
                improved = True
                break
        
        if not improved:
            # If we can't improve by moving the worst vertex, try moving any random point
            # with a smaller step, or reduce step size
            step_size *= 0.95
            if step_size < 1e-5:
                break
                
    return best, best_score


def _simulated_annealing(points, max_iter=10000, t_start=0.05, t_end=0.001, 
                         cooling_rate=0.9999, rng_seed=42):
    """
    Simulated Annealing with moves that try to spread points.
    """
    rng = np.random.RandomState(rng_seed)
    current = points.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    n = len(current)
    
    for i in range(max_iter):
        t = t_start * (cooling_rate ** i)
        if t < t_end:
            break
        
        # Pick a random point to move
        idx = rng.randint(0, n)
        
        # Move
        new_points = current.copy()
        # Random step
        step = rng.normal(0, 0.05, 2)
        new_points[idx] += step
        
        new_score = _compute_score(new_points)
        
        # Accept if better or with probability exp((new - current) / t)
        if new_score > current_score:
            current = new_points
            current_score = new_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        else:
            delta = new_score - current_score
            # delta is negative
            if delta > -0.1: # Avoid overflow in exp
                prob = np.exp(delta / t)
                if rng.random() < prob:
                    current = new_points
                    current_score = new_score
    
    return best, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Initializations: Regular polygon is the strongest prior
    initializations = []
    # Regular 13-gon
    initializations.append(_regular_polygon(N, radius=1.0, rotation=0.0))
    # Slightly rotated
    initializations.append(_regular_polygon(N, radius=1.0, rotation=np.pi/13))
    # Slightly perturbed regular polygon
    rng_init = np.random.RandomState(0)
    base = _regular_polygon(N, radius=1.0, rotation=0.0)
    perturbed = base + rng_init.normal(0, 0.05, (N, 2))
    initializations.append(perturbed)
    
    best_points = None
    best_score = -float('inf')
    
    # Evaluate initializations
    for init in initializations:
        score = _compute_score(init)
        if score > best_score:
            best_score = score
            best_points = init.copy()
    
    # Optimization Loop
    # We will alternate between SA (to explore) and Worst-Case LS (to exploit)
    
    # 1. Run SA on the best initialization to escape potential symmetries if needed
    # But regular polygon is likely optimal. Let's do LS first.
    
    if time.time() < deadline - 20:
        ls_points, ls_score = _local_search_worst_case(
            best_points, max_iter=3000, step_size=0.02, rng_seed=100
        )
        if ls_score > best_score:
            best_score = ls_score
            best_points = ls_points.copy()
            
    # 2. Run SA to see if we can find a better configuration
    if time.time() < deadline - 30:
        sa_points, sa_score = _simulated_annealing(
            best_points, max_iter=10000, t_start=0.02, t_end=0.0005,
            cooling_rate=0.9999, rng_seed=200
        )
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()
            
        # Polish SA result
        if time.time() < deadline - 10:
            ls_points, ls_score = _local_search_worst_case(
                best_points, max_iter=2000, step_size=0.01, rng_seed=300
            )
            if ls_score > best_score:
                best_score = ls_score
                best_points = ls_points.copy()
                
    # 3. Try a different starting point if time remains
    # Random points in a circle
    if time.time() < deadline - 40:
        rng = np.random.RandomState(42)
        angles = rng.uniform(0, 2*np.pi, N)
        radii = rng.uniform(0.8, 1.2, N)
        random_init = np.column_stack((radii*np.cos(angles), radii*np.sin(angles)))
        
        sa_points, sa_score = _simulated_annealing(
            random_init, max_iter=5000, t_start=0.05, t_end=0.001,
            cooling_rate=0.9999, rng_seed=500
        )
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()
            
        if time.time() < deadline - 10:
            ls_points, ls_score = _local_search_worst_case(
                best_points, max_iter=2000, step_size=0.01, rng_seed=600
            )
            if ls_score > best_score:
                best_score = ls_score
                best_points = ls_points.copy()

    # Final polish
    if time.time() < deadline - 5 and best_points is not None:
        ls_points, ls_score = _local_search_worst_case(
            best_points, max_iter=1000, step_size=0.005, rng_seed=700
        )
        if ls_score > best_score:
            best_score = ls_score
            best_points = ls_points.copy()

    if best_points is None:
        return initializations[0]
        
    return best_points

# EVOLVE-BLOCK-END
