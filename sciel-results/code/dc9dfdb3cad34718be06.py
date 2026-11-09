# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 585  # Leave 45s margin from 630s

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

    def all_triangle_areas(points):
        """
        Calculate all triangle areas among all triples of points using vectorization.
        points: (n, 2)
        Returns: array of shape (286,)
        """
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        
        areas = 0.5 * np.abs(p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
                             p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
                             p3[:, 0] * (p1[:, 1] - p2[:, 1]))
        
        return areas

    def hull_area(points):
        """Calculate the convex hull area of the points."""
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0

    def soft_min(areas, beta):
        """
        Smooth approximation of min using log-sum-exp trick.
        soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
        As beta -> inf, soft_min -> min
        """
        shifted = -beta * areas
        max_shifted = np.max(shifted)
        log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        return -log_sum / beta

    def objective_smooth(points, beta=50.0):
        """
        Smooth objective using soft-min surrogate for the minimum triangle area.
        This provides a smoother landscape for optimization.
        """
        areas = all_triangle_areas(points)
        A_min_smooth = soft_min(areas, beta)
        
        if A_min_smooth < 1e-15:
            return 0.0
        
        # Use bounding box area as a fast proxy for hull area
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        bbox_area = (x_max - x_min) * (y_max - y_min)
        
        if bbox_area < 1e-15:
            return 0.0
            
        return -A_min_smooth / bbox_area

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
    hex_configs = []
    for offset in [0.0, 0.1, 0.2]:
        pts = []
        pts.append([0.0, 0.0])
        for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
            r = 1.0 + offset * rng.random()
            pts.append([r * np.cos(angle), r * np.sin(angle)])
        for angle in np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6:
            r = 1.8 + offset * rng.random()
            pts.append([r * np.cos(angle), r * np.sin(angle)])
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
        
    # 6. Grid-like with jitter
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
    
    # 7. Two concentric rings with alternating angles
    for ring_ratio in [0.5, 0.6, 0.7]:
        inner_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
        outer_angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
        inner_pts = np.column_stack([ring_ratio * np.cos(inner_angles), ring_ratio * np.sin(inner_angles)])
        outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        initial_configs.append(np.vstack([inner_pts, outer_pts]))
    
    # 8. Fibonacci spiral pattern
    phi = (1 + np.sqrt(5)) / 2
    fib_angles = 2 * np.pi * np.arange(n) / phi
    fib_radii = np.sqrt(np.arange(n) / n) * 0.9
    fib_pts = np.column_stack([fib_radii * np.cos(fib_angles), fib_radii * np.sin(fib_angles)])
    initial_configs.append(fib_pts)

    # 9. Additional: Perturbed hexagonal lattice (more structured)
    for _ in range(2):
        pts = []
        # Center
        pts.append([0.0, 0.0] + rng.normal(0, 0.05, 2))
        # Inner ring of 6
        for angle in np.linspace(0, 2*np.pi, 6, endpoint=False) + rng.uniform(0, 2*np.pi):
            r = 1.0 + rng.normal(0, 0.1)
            pts.append([r * np.cos(angle), r * np.sin(angle)])
        # Outer ring of 6
        for angle in np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6 + rng.uniform(0, 2*np.pi):
            r = 1.8 + rng.normal(0, 0.1)
            pts.append([r * np.cos(angle), r * np.sin(angle)])
        initial_configs.append(np.array(pts[:13]))

    # 10. Additional: Square lattice with center and corners
    for scale in [0.8, 1.0, 1.2]:
        pts = []
        # 4 corners
        for x in [-scale, scale]:
            for y in [-scale, scale]:
                pts.append([x, y])
        # 4 edge midpoints
        for x in [0]:
            for y in [-scale, scale]:
                pts.append([x, y])
        for y in [0]:
            for x in [-scale, scale]:
                pts.append([x, y])
        # 4 inner points
        for x in [-scale/2, scale/2]:
            for y in [-scale/2, scale/2]:
                pts.append([x, y])
        initial_configs.append(np.array(pts[:13]))

    best_points = None
    best_score = -float('inf')
    
    # Phase 1: Quick optimization with smooth objective (multiple beta values)
    for beta in [20.0, 50.0, 100.0, 200.0]:
        if time.time() - start_time > time_limit - 150:
            break
        
        for i, init_points in enumerate(initial_configs):
            if time.time() - start_time > time_limit - 120:
                break
                
            center = np.mean(init_points, axis=0)
            centered = init_points - center
            norm = np.linalg.norm(centered)
            if norm > 1e-10:
                centered = centered / norm * 1.0
            
            try:
                result = minimize(
                    lambda pts: objective_smooth(pts, beta),
                    centered,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                        'initial_simplex': 0.1
                    }
                )
                
                if result.fun < 0:
                    exact_score = score(result.x)
                    if exact_score > best_score:
                        best_score = exact_score
                        best_points = result.x.copy()
                        
            except Exception:
                pass

    # Phase 2: Refine with exact objective using L-BFGS-B for better convergence
    if best_points is not None and time.time() - start_time < time_limit - 80:
        # Try multiple refinement strategies
        for strategy in ['nm_large', 'nm_small', 'nm_tiny']:
            if time.time() - start_time > time_limit - 60:
                break
                
            if strategy == 'nm_large':
                opts = {'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True, 'initial_simplex': 0.05}
            elif strategy == 'nm_small':
                opts = {'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True, 'initial_simplex': 0.01}
            else:
                opts = {'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True, 'initial_simplex': 0.003}
            
            try:
                result = minimize(
                    objective_exact,
                    best_points,
                    method='Nelder-Mead',
                    options=opts
                )
                
                if result.fun < 0:
                    exact_score = -result.fun
                    if exact_score > best_score:
                        best_score = exact_score
                        best_points = result.x.copy()
            except Exception:
                pass

    # Phase 3: High-precision polish with multiple simplex sizes
    if best_points is not None and time.time() - start_time < time_limit - 50:
        for simplex_size in [0.001, 0.0005, 0.0002, 0.0001]:
            if time.time() - start_time > time_limit - 30:
                break
            try:
                result = minimize(
                    objective_exact,
                    best_points,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 8000,
                        'xatol': 1e-14,
                        'fatol': 1e-16,
                        'adaptive': True,
                        'initial_simplex': simplex_size
                    }
                )
                
                if result.fun < 0:
                    exact_score = -result.fun
                    if exact_score > best_score:
                        best_score = exact_score
                        best_points = result.x.copy()
            except Exception:
                pass

    # Phase 4: Final ultra-precise polish
    if best_points is not None and time.time() - start_time < time_limit - 20:
        try:
            result = minimize(
                objective_exact,
                best_points,
                method='Nelder-Mead',
                options={
                    'maxiter': 15000,
                    'xatol': 1e-15,
                    'fatol': 1e-17,
                    'adaptive': True,
                    'initial_simplex': 0.00005
                }
            )
            
            if result.fun < 0:
                exact_score = -result.fun
                if exact_score > best_score:
                    best_score = exact_score
                    best_points = result.x.copy()
        except Exception:
            pass

    # Phase 5: Additional restarts from slightly perturbed best solution
    if best_points is not None and time.time() - start_time < time_limit - 15:
        for _ in range(3):
            if time.time() - start_time > time_limit - 10:
                break
            perturbed = best_points + rng.normal(0, 0.001, (n, 2))
            try:
                result = minimize(
                    objective_exact,
                    perturbed,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 3000,
                        'xatol': 1e-14,
                        'fatol': 1e-16,
                        'adaptive': True,
                        'initial_simplex': 0.001
                    }
                )
                
                if result.fun < 0:
                    exact_score = -result.fun
                    if exact_score > best_score:
                        best_score = exact_score
                        best_points = result.x.copy()
            except Exception:
                pass

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
