"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _min_triangle_area(points):
    """Compute minimum area among all C(11,3)=165 triangles using vectorized computation."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _min_triangle_area_fast(points):
    """Faster min triangle area using numpy broadcasting for small n."""
    n = len(points)
    # Compute all pairwise differences
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
                if min_area < 1e-10:
                    return min_area
    return min_area

def _inside_triangle(p, tol=1e-6):
    """Check if point is inside or on boundary of equilateral triangle."""
    h = np.sqrt(3) / 2
    x, y = p[0], p[1]
    if x < -tol or x > 1+tol or y < -tol or y > h+tol:
        return False
    if y > 2*h*x + tol:
        return False
    if y > 2*h*(1-x) + tol:
        return False
    return True

def _all_inside(points, tol=1e-6):
    """Check if all points are inside the triangle."""
    h = np.sqrt(3) / 2
    for p in points:
        x, y = p[0], p[1]
        if x < -tol or x > 1+tol or y < -tol or y > h+tol:
            return False
        if y > 2*h*x + tol:
            return False
        if y > 2*h*(1-x) + tol:
            return False
    return True

def _penalty(points):
    """Compute penalty for points outside the triangle."""
    h = np.sqrt(3) / 2
    penalty = 0.0
    for p in points:
        x, y = p[0], p[1]
        if x < 0:
            penalty += (0 - x) ** 2
        elif x > 1:
            penalty += (x - 1) ** 2
        if y < 0:
            penalty += (0 - y) ** 2
        elif y > h:
            penalty += (y - h) ** 2
        if y > 2*h*x:
            penalty += (y - 2*h*x) ** 2
        if y > 2*h*(1-x):
            penalty += (y - 2*h*(1-x)) ** 2
    return penalty

def _negative_min_area(params):
    """Objective for differential evolution: negative of min triangle area."""
    points = params.reshape(-1, 2)
    pen = _penalty(points)
    if pen > 1e-8:
        return -0.0001 + pen * 10
    return -_min_triangle_area_fast(points)

def _simulated_annealing(points, deadline, rng):
    """Simulated annealing to maximize minimum triangle area."""
    h = np.sqrt(3) / 2
    current_points = points.copy()
    current_area = _min_triangle_area_fast(current_points)
    best_points = current_points.copy()
    best_area = current_area
    
    # Annealing schedule
    t_start = 0.05
    t_end = 0.0001
    n_iter = 50000
    cooling_rate = (t_end / t_start) ** (1.0 / n_iter)
    
    t = t_start
    for iteration in range(n_iter):
        if time.time() > deadline - 5:
            break
        
        # Propose a move: perturb one or two points
        n_move = 1 if rng.random() < 0.7 else 2
        indices = rng.choice(N, size=n_move, replace=False)
        
        new_points = current_points.copy()
        for idx in indices:
            # Random perturbation with adaptive step size
            step = t * 2.0
            delta = rng.normal(0, step, size=2)
            new_points[idx] = current_points[idx] + delta
        
        # Check bounds
        if _penalty(new_points) > 1e-6:
            continue
        
        new_area = _min_triangle_area_fast(new_points)
        delta_area = new_area - current_area
        
        if delta_area > 0 or rng.random() < np.exp(delta_area / t):
            current_points = new_points
            current_area = new_area
            if current_area > best_area:
                best_area = current_area
                best_points = current_points.copy()
        
        t *= cooling_rate
    
    return best_points, best_area

def _local_polish(points, deadline):
    """Local refinement using Nelder-Mead on flattened parameters."""
    points = points.copy()
    current_min_area = _min_triangle_area_fast(points)
    
    def objective(params):
        pts = params.reshape(-1, 2)
        pen = _penalty(pts)
        if pen > 1e-8:
            return 10.0 + pen * 10
        return -_min_triangle_area_fast(pts)
    
    # Multiple Nelder-Mead passes with different starting perturbations
    rng = np.random.RandomState(42)
    for pass_num in range(6):
        if time.time() > deadline - 3:
            break
        
        x0 = points.flatten()
        if pass_num > 0:
            x0 = x0 + rng.normal(0, 0.01, size=2*N)
            pts_check = x0.reshape(-1, 2)
            if _penalty(pts_check) > 1e-6:
                x0 = points.flatten()
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if result.fun < -current_min_area:
                new_points = result.x.reshape(-1, 2)
                if _penalty(new_points) < 1e-8:
                    new_min_area = _min_triangle_area_fast(new_points)
                    if new_min_area > current_min_area:
                        points = new_points
                        current_min_area = new_min_area
        except Exception:
            pass
    
    # Coordinate descent refinement
    if time.time() > deadline - 2:
        return points, current_min_area
    
    for _ in range(300):
        if time.time() > deadline - 2:
            break
        improved = False
        for i in range(N):
            for coord in range(2):
                for step in [0.003, 0.001, 0.0003, 0.0001]:
                    old_val = points[i, coord]
                    points[i, coord] = old_val + step
                    if _penalty(points) < 1e-8:
                        new_area = _min_triangle_area_fast(points)
                        if new_area > current_min_area:
                            current_min_area = new_area
                            improved = True
                            continue
                    points[i, coord] = old_val - step
                    if _penalty(points) < 1e-8:
                        new_area = _min_triangle_area_fast(points)
                        if new_area > current_min_area:
                            current_min_area = new_area
                            improved = True
                            continue
                    points[i, coord] = old_val
        if not improved:
            break
    
    return points, current_min_area

def _rigid_cluster_config(params):
    """
    Build 11 points from rigid cluster parameters.
    params: [scale, rotation, tx, ty, px, py]
    First 10 points: rigid transform of base config
    Last point: free point (px, py)
    """
    scale, rotation, tx, ty, px, py = params
    
    # Base 10-point configuration: 3 corners + 6 edge points + 1 center
    h = np.sqrt(3) / 2
    base = np.array([
        [0.0, 0.0],           # corner
        [1.0, 0.0],           # corner
        [0.5, h],             # corner
        [0.333, 0.0],         # bottom edge
        [0.667, 0.0],         # bottom edge
        [0.167, 0.289],       # left edge
        [0.833, 0.289],       # right edge
        [0.333, 0.577],       # upper left edge
        [0.667, 0.577],       # upper right edge
        [0.5, h/3],           # center
    ])
    
    # Center the base configuration
    center = base.mean(axis=0)
    centered = base - center
    
    # Rotate
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    
    # Scale, rotate, translate
    cluster = centered * scale @ R.T + np.array([tx, ty])
    
    # Add free point
    points = np.vstack([cluster, [[px, py]]])
    return points

def _cluster_objective(params):
    """Objective: negative min triangle area with penalty for out-of-bounds."""
    points = _rigid_cluster_config(params)
    pen = _penalty(points)
    if pen > 1e-8:
        return -0.001 + pen * 10
    return -_min_triangle_area_fast(points)

def _optimize_rigid_cluster(deadline):
    """Optimize the rigid cluster + free point configuration."""
    h = np.sqrt(3) / 2
    
    bounds = [
        (0.4, 1.3),   # scale
        (-0.8, 0.8),  # rotation
        (0.1, 0.9),   # tx
        (0.05, h-0.05),  # ty
        (0.02, 0.98),    # px
        (0.02, h-0.02)   # py
    ]
    
    best_params = None
    best_obj = float('inf')
    
    # Run differential evolution
    try:
        de_result = differential_evolution(
            _cluster_objective, bounds,
            seed=42, maxiter=150, popsize=15,
            tol=1e-10, mutation=(0.5, 1.5), recombination=0.8,
            polish=True, init='sobol'
        )
        if de_result.fun < best_obj:
            best_obj = de_result.fun
            best_params = de_result.x
    except Exception:
        pass
    
    # Try multiple random starts with local optimization
    rng = np.random.RandomState(123)
    for trial in range(10):
        if time.time() > deadline - 25:
            break
        
        x0 = np.array([
            rng.uniform(0.5, 1.2),
            rng.uniform(-0.5, 0.5),
            rng.uniform(0.2, 0.8),
            rng.uniform(0.1, h-0.1),
            rng.uniform(0.1, 0.9),
            rng.uniform(0.1, h-0.1)
        ])
        
        try:
            result = minimize(_cluster_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    if best_params is not None:
        points = _rigid_cluster_config(best_params)
        if _penalty(points) < 1e-8:
            return points, _min_triangle_area_fast(points)
    
    return None, 0.0

def _generate_initial_points(rng, method='random'):
    """Generate initial point configurations."""
    h = np.sqrt(3) / 2
    
    if method == 'grid':
        # Grid-like initial configuration
        points = []
        for i in range(4):
            for j in range(3):
                if len(points) >= N:
                    break
                x = 0.15 + i * 0.23
                y = 0.1 + j * 0.25
                if _inside_triangle([x, y]):
                    points.append([x, y])
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.95 and u > 0.02 and v > 0.02:
                px = u + 0.5 * v
                py = h * v
                points.append([px, py])
        return np.array(points[:N])
    elif method == 'hexagonal':
        # Hexagonal-like packing
        points = []
        h = np.sqrt(3) / 2
        # Corners
        points.append([0.05, 0.05])
        points.append([0.95, 0.05])
        points.append([0.5, h - 0.05])
        # Edge midpoints
        points.append([0.5, 0.08])
        points.append([0.1, 0.08 + 2*h*0.1])
        points.append([0.9, 0.08 + 2*h*0.9])
        # Interior
        cx, cy = 0.5, h/3
        r = 0.18
        points.append([cx - r*0.7, cy])
        points.append([cx + r*0.7, cy])
        points.append([cx, cy + r*0.7])
        points.append([cx, cy - r*0.7])
        return np.array(points[:N])
    else:
        # Random
        points = []
        for _ in range(N):
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 0.95 and u > 0.02 and v > 0.02:
                    px = u + 0.5 * v
                    py = h * v
                    points.append([px, py])
                    break
        return np.array(points)

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    h = np.sqrt(3) / 2
    
    best_points = None
    best_min_area = 0.0
    rng = np.random.RandomState(42)
    
    # Strategy 1: Rigid cluster optimization (primary, reduces dimensionality)
    if time.time() < deadline - 40:
        cluster_points, cluster_area = _optimize_rigid_cluster(deadline)
        if cluster_points is not None:
            if cluster_area > best_min_area:
                best_min_area = cluster_area
                best_points = cluster_points
    
    # Strategy 2: Full differential evolution
    if best_points is None or time.time() < deadline - 50:
        bounds = [(0, 1)] * 2 * N
        try:
            de_result = differential_evolution(
                _negative_min_area, bounds,
                seed=42, maxiter=80, popsize=12,
                tol=1e-8, mutation=(0.5, 1.5), recombination=0.7,
                polish=True, init='sobol'
            )
            de_points = de_result.x.reshape(-1, 2)
            if _penalty(de_points) < 1e-8:
                de_min_area = _min_triangle_area_fast(de_points)
                if de_min_area > best_min_area:
                    best_min_area = de_min_area
                    best_points = de_points.copy()
        except Exception:
            pass
    
    # Strategy 3: Multiple random initializations with SA + local polish
    if best_points is None or time.time() < deadline - 30:
        methods = ['hexagonal', 'grid', 'random', 'random', 'random', 'random']
        for method in methods:
            if time.time() > deadline - 20:
                break
            
            points = _generate_initial_points(rng, method)
            
            # Simulated annealing
            sa_points, sa_area = _simulated_annealing(points, deadline, rng)
            
            # Local polish
            polished, min_area = _local_polish(sa_points, deadline)
            
            if min_area > best_min_area:
                best_min_area = min_area
                best_points = polished
    
    # Final polish on the best result
    if best_points is not None and time.time() < deadline - 5:
        best_points, best_min_area = _local_polish(best_points, deadline)
    
    if best_points is None:
        # Fallback: simple configuration
        h = np.sqrt(3) / 2
        best_points = np.array([
            [0.05, 0.05], [0.95, 0.05], [0.5, h - 0.05],
            [0.5, 0.08], [0.1, 0.1 + 2*h*0.1], [0.9, 0.1 + 2*h*0.9],
            [0.3, h/3], [0.7, h/3], [0.5, h/3 + 0.15], [0.5, h/3 - 0.15],
            [0.5, h/3]
        ])
    
    # Final verification
    if _penalty(best_points) > 1e-6:
        # Project points inside triangle
        for i in range(N):
            x, y = best_points[i]
            # Clamp to triangle
            if y < 0:
                y = 0.01
            if y > h:
                y = h - 0.01
            if x < 0:
                x = 0.01
            if x > 1:
                x = 0.99
            # Check slanted edges
            max_y_left = 2*h*x
            max_y_right = 2*h*(1-x)
            if y > max_y_left:
                y = max_y_left - 0.01
            if y > max_y_right:
                y = max_y_right - 0.01
            best_points[i] = [x, y]
    
    return best_points
# EVOLVE-BLOCK-END
