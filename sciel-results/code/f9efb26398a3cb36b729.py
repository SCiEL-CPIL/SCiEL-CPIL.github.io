# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

def compute_score(points):
    """Compute the Heilbronn score: A_min / A_hull (vectorized)"""
    n = len(points)
    if n < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
    
    # Vectorized triangle area computation
    idx = np.array(list(itertools.combinations(range(n), 3)))
    if len(idx) == 0:
        return 0.0
    
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / A_hull

def compute_min_triangles(points, k=5):
    """Return indices of the k smallest triangles"""
    n = len(points)
    idx = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Get indices of k smallest
    k = min(k, len(areas))
    smallest_idx = np.argpartition(areas, k)[:k]
    return idx[smallest_idx]

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area ratio.
    Uses vectorized computation and focused optimization on critical triangles.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 630s limit
    
    def time_left():
        return deadline - time.time()
    
    # Initial configurations based on known Heilbronn structures
    
    # Config 1: Regular 12-gon + center
    def make_config1():
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        outer = np.column_stack([np.cos(angles), np.sin(angles)])
        center = np.array([[0.0, 0.0]])
        return np.vstack([outer, center])
    
    # Config 2: Two concentric rings (7 outer, 6 inner)
    def make_config2():
        angles1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        angles2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
        outer = np.column_stack([np.cos(angles1), np.sin(angles1)])
        inner = np.column_stack([0.55 * np.cos(angles2), 0.55 * np.sin(angles2)])
        return np.vstack([outer, inner])
    
    # Config 3: Three rings (5, 4, 4)
    def make_config3():
        a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        a3 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        p1 = np.column_stack([np.cos(a1), np.sin(a1)])
        p2 = np.column_stack([0.6 * np.cos(a2), 0.6 * np.sin(a2)])
        p3 = np.column_stack([0.3 * np.cos(a3), 0.3 * np.sin(a3)])
        return np.vstack([p1, p2, p3])
    
    # Config 4: Based on known optimal for small n - mix of structured
    def make_config4():
        # Try a pattern: 1 center, 6 at r=0.5, 6 at r=1.0
        a1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
        p1 = np.array([[0.0, 0.0]])
        p2 = np.column_stack([0.5 * np.cos(a1), 0.5 * np.sin(a1)])
        p3 = np.column_stack([np.cos(a2), np.sin(a2)])
        return np.vstack([p1, p2, p3])
    
    # Config 5: Perturbed regular 13-gon
    def make_config5():
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        rng = np.random.default_rng(42)
        points += rng.normal(0, 0.03, size=points.shape)
        return points
    
    # Config 6: Another structured pattern
    def make_config6():
        # 1 center, 3 at r=0.3, 3 at r=0.6, 6 at r=1.0
        a1 = np.linspace(0, 2 * np.pi, 3, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 3, endpoint=False) + np.pi / 3
        a3 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        p1 = np.array([[0.0, 0.0]])
        p2 = np.column_stack([0.3 * np.cos(a1), 0.3 * np.sin(a1)])
        p3 = np.column_stack([0.6 * np.cos(a2), 0.6 * np.sin(a2)])
        p4 = np.column_stack([np.cos(a3), np.sin(a3)])
        return np.vstack([p1, p2, p3, p4])
    
    initial_configs = [
        make_config1(),
        make_config2(),
        make_config3(),
        make_config4(),
        make_config5(),
        make_config6(),
    ]
    
    # Also add some random starts
    rng = np.random.default_rng(7)
    for _ in range(3):
        pts = rng.uniform(-1, 1, size=(13, 2))
        initial_configs.append(pts)
    
    best_points = None
    best_score = 0.0
    
    for config in initial_configs:
        if time_left() < 10:
            break
        
        points = config.copy()
        score = compute_score(points)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # Local optimization: focus on critical triangles
        step_size = 0.05
        rng_local = np.random.default_rng(hash(tuple(points.flatten().round(6))) % 2**32)
        
        for iteration in range(20000):
            if time_left() < 10:
                break
            
            # Find the critical triangles (smallest area)
            min_tris = compute_min_triangles(points, k=3)
            
            # Get all point indices involved in critical triangles
            critical_indices = set()
            for tri in min_tris:
                for idx in tri:
                    critical_indices.add(idx)
            
            # Perturb points involved in critical triangles
            new_points = points.copy()
            n_perturb = min(len(critical_indices), 3)
            indices_to_perturb = list(critical_indices)[:n_perturb]
            
            for idx in indices_to_perturb:
                # Try to move the point away from forming small triangles
                # Compute gradient-like direction: move away from other critical points
                perturbation = rng_local.normal(0, step_size, size=2)
                new_points[idx] += perturbation
            
            # Also try perturbing a random point
            if rng_local.random() < 0.3:
                rand_idx = rng_local.integers(0, n)
                new_points[rand_idx] += rng_local.normal(0, step_size, size=2)
            
            # Keep in reasonable bounds
            new_points = np.clip(new_points, -1.5, 1.5)
            
            new_score = compute_score(new_points)
            
            if new_score > score:
                points = new_points
                score = new_score
                step_size = min(step_size * 1.05, 0.3)
                
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
            else:
                step_size = max(step_size * 0.97, 0.005)
        
        # After local search, try a final refinement with smaller steps
        if time_left() > 30:
            step_size = 0.01
            for iteration in range(5000):
                if time_left() < 10:
                    break
                
                min_tris = compute_min_triangles(points, k=2)
                critical_indices = set()
                for tri in min_tris:
                    for idx in tri:
                        critical_indices.add(idx)
                
                new_points = points.copy()
                for idx in list(critical_indices)[:2]:
                    new_points[idx] += rng_local.normal(0, step_size, size=2)
                
                new_points = np.clip(new_points, -1.5, 1.5)
                new_score = compute_score(new_points)
                
                if new_score > score:
                    points = new_points
                    score = new_score
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
                else:
                    step_size = max(step_size * 0.95, 0.001)
    
    # Final: try a global random search with the best structure
    if time_left() > 60 and best_points is not None:
        # Use the best structure as a base and try more perturbations
        base = best_points.copy()
        rng_final = np.random.default_rng(99)
        for _ in range(100):
            if time_left() < 10:
                break
            new_pts = base + rng_final.normal(0, 0.1, size=(13, 2))
            new_pts = np.clip(new_pts, -1.5, 1.5)
            s = compute_score(new_pts)
            if s > best_score:
                best_score = s
                best_points = new_pts.copy()
    
    # Ensure validity
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = make_config1()
    
    # Final validation
    try:
        hull = ConvexHull(best_points)
    except Exception:
        best_points = make_config1()
    
    return best_points

# EVOLVE-BLOCK-END
