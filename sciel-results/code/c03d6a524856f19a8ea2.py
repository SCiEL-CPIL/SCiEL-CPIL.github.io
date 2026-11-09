# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 590  # Leave 40s margin from 630s

    # Precompute indices for all triples of points
    idx = np.array(list((i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)))
    
    def min_triangle_area_vectorized(points):
        """
        Calculate the minimum triangle area among all triples of points using vectorization.
        points: (n, 2)
        """
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        
        areas = 0.5 * np.abs(p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
                             p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
                             p3[:, 0] * (p1[:, 1] - p2[:, 1]))
        
        return np.min(areas)

    def hull_area(points):
        """Calculate the convex hull area of the points."""
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0

    def objective_softmin(points, beta=50.0):
        """
        Smooth surrogate objective using soft-min via log-sum-exp.
        Minimizes -soft_min_triangle_area / bbox_area.
        
        The soft-min approximates the minimum but is differentiable,
        helping gradient-free optimizers navigate the nonsmooth landscape.
        As beta increases, soft-min -> true min.
        """
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        
        areas = 0.5 * np.abs(p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
                             p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
                             p3[:, 0] * (p1[:, 1] - p2[:, 1]))
        
        # Soft-min via log-sum-exp: -log(sum(exp(-beta * a))) / beta
        # Numerically stable version
        min_a = np.min(areas)
        if min_a < 1e-15:
            return 0.0
        shifted = -beta * (areas - min_a)
        log_sum = np.log(np.sum(np.exp(shifted)))
        soft_min = min_a - log_sum / beta
        
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        bbox_area = (x_max - x_min) * (y_max - y_min)
        
        if bbox_area < 1e-15:
            return 0.0
            
        return -soft_min / bbox_area

    def objective_exact(points):
        """
        Exact objective: minimize -A_min / A_hull.
        Uses actual ConvexHull area.
        """
        A_hull = hull_area(points)
        if A_hull <= 1e-10:
            return 0.0
        
        A_min = min_triangle_area_vectorized(points)
        
        if A_min < 1e-12:
            return 0.0
            
        return -A_min / A_hull

    def score(points):
        """Calculate the actual score: A_min / A_hull"""
        A_hull = hull_area(points)
        if A_hull <= 1e-10:
            return 0.0
        A_min = min_triangle_area_vectorized(points)
        if A_min < 1e-12:
            return 0.0
        return A_min / A_hull

    # Generate diverse initial configurations
    rng = np.random.default_rng(seed=42)
    
    initial_configs = []
    
    # 1. Regular polygon (baseline)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    initial_configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Regular polygon with center point and perturbed outer ring
    center_pt = np.array([0.0, 0.0])
    outer_angles = np.linspace(0, 2*np.pi, n-1, endpoint=False)
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    initial_configs.append(np.vstack([outer_pts, [center_pt]]))
    
    # 3. Hexagonal lattice pattern
    # Create a hexagonal grid pattern
    hex_configs = []
    for offset in [0.0, 0.1, 0.2]:
        pts = []
        # Center
        pts.append([0.0, 0.0])
        # First ring (6 points)
        for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
            r = 1.0 + offset * rng.random()
            pts.append([r * np.cos(angle), r * np.sin(angle)])
        # Second ring (6 points)
        for angle in np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6:
            r = 1.8 + offset * rng.random()
            pts.append([r * np.cos(angle), r * np.sin(angle)])
        # We have 13 points now
        hex_configs.append(np.array(pts[:13]))
    initial_configs.extend(hex_configs)
    
    # 4. Perturbed regular polygon
    for _ in range(3):
        perturbation = rng.normal(0, 0.15, (n, 2))
        initial_configs.append(np.column_stack([np.cos(angles), np.sin(angles)]) + perturbation)
        
    # 5. Random points in unit circle
    for _ in range(3):
        r = np.sqrt(rng.random(n))
        theta = rng.random(n) * 2 * np.pi
        initial_configs.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
        
    # 6. Grid-like with jitter (4x4 grid minus one, centered)
    grid_x = [-1.5, -0.5, 0.5, 1.5]
    grid_y = [-1.5, -0.5, 0.5, 1.5]
    for _ in range(3):
        pts = []
        used = set()
        while len(pts) < n:
            i = rng.integers(0, 4)
            j = rng.integers(0, 4)
            if (i, j) not in used:
                used.add((i, j))
                x = grid_x[i] + rng.normal(0, 0.1)
                y = grid_y[j] + rng.normal(0, 0.1)
                pts.append([x, y])
        initial_configs.append(np.array(pts))

    best_points = None
    best_score = -float('inf')
    
    # First pass: quick optimization with soft-min surrogate objective
    for i, init_points in enumerate(initial_configs):
        if time.time() - start_time > time_limit - 60:
            break
            
        # Center the points
        center = np.mean(init_points, axis=0)
        centered = init_points - center
        norm = np.linalg.norm(centered)
        if norm > 1e-10:
            centered = centered / norm * 1.0
            
        try:
            # Use Nelder-Mead with soft-min surrogate (smooth objective)
            # Start with moderate beta, then increase for sharper min
            for beta in [30.0, 80.0, 200.0]:
                if time.time() - start_time > time_limit - 60:
                    break
                result = minimize(
                    lambda x, b=beta: objective_softmin(x, b),
                    centered,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1500,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True,
                        'initial_simplex': 0.15
                    }
                )
                
                if result.fun < 0:
                    # Evaluate with exact objective
                    exact_score = -objective_exact(result.x)
                    if exact_score > best_score:
                        best_score = exact_score
                        best_points = result.x.copy()
                    # Use result as new starting point for next beta
                    centered = result.x.copy()
                    
        except Exception:
            pass

    # Second pass: refine best points with exact objective
    if best_points is not None and time.time() - start_time < time_limit - 30:
        try:
            result = minimize(
                objective_exact,
                best_points,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True,
                    'initial_simplex': 0.05
                }
            )
            
            if result.fun < 0:
                exact_score = -result.fun
                if exact_score > best_score:
                    best_score = exact_score
                    best_points = result.x.copy()
        except Exception:
            pass

    # Third pass: high-precision polish with progressively tighter settings
    if best_points is not None and time.time() - start_time < time_limit - 20:
        polish_configs = [
            {'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13, 'initial_simplex': 0.02},
            {'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'initial_simplex': 0.01},
            {'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-15, 'initial_simplex': 0.005},
        ]
        for cfg in polish_configs:
            if time.time() - start_time > time_limit - 15:
                break
            try:
                result = minimize(
                    objective_exact,
                    best_points,
                    method='Nelder-Mead',
                    options={
                        'maxiter': cfg['maxiter'],
                        'xatol': cfg['xatol'],
                        'fatol': cfg['fatol'],
                        'adaptive': True,
                        'initial_simplex': cfg['initial_simplex']
                    }
                )
                
                if result.fun < 0:
                    exact_score = -result.fun
                    if exact_score > best_score:
                        best_score = exact_score
                        best_points = result.x.copy()
            except Exception:
                pass

    # Fourth pass: coordinate descent refinement on individual points
    # Each point is moved to maximize the minimum triangle area
    if best_points is not None and time.time() - start_time < time_limit - 10:
        for iteration in range(5):
            if time.time() - start_time > time_limit - 10:
                break
            improved = False
            for i in range(n):
                if time.time() - start_time > time_limit - 10:
                    break
                # Try small perturbations of point i
                base = best_points.copy()
                base_score = score(base)
                for dx in [-0.02, -0.01, 0.0, 0.01, 0.02]:
                    for dy in [-0.02, -0.01, 0.0, 0.01, 0.02]:
                        if dx == 0.0 and dy == 0.0:
                            continue
                        candidate = base.copy()
                        candidate[i, 0] += dx
                        candidate[i, 1] += dy
                        cand_score = score(candidate)
                        if cand_score > base_score + 1e-12:
                            base = candidate
                            base_score = cand_score
                            improved = True
            if improved:
                best_points = base
                best_score = base_score

    # Final fallback
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points

# EVOLVE-BLOCK-END
