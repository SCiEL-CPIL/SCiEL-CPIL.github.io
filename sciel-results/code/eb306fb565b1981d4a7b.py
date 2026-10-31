"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _min_triangle_area(points):
    """Compute minimum area among all C(11,3)=165 triangles."""
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

def _inside_triangle(p, tol=1e-6):
    """Check if point is inside or on boundary of equilateral triangle."""
    h = np.sqrt(3) / 2
    x, y = p[0], p[1]
    if x < -tol or x > 1+tol or y < -tol or y > h+tol:
        return False
    if y < -tol:
        return False
    if y > 2*h*x + tol:
        return False
    if y > 2*h*(1-x) + tol:
        return False
    return True

def _all_inside(points, tol=1e-6):
    """Check if all points are inside the triangle."""
    return all(_inside_triangle(p, tol) for p in points)

def _negative_min_area(params):
    """Objective for differential evolution: negative of min triangle area."""
    points = params.reshape(-1, 2)
    if not _all_inside(points):
        penalty = 0.0
        for p in points:
            h = np.sqrt(3) / 2
            x, y = p[0], p[1]
            if x < 0 or x > 1 or y < 0 or y > h:
                penalty += 10.0
            if y > 2*h*x + 1e-4:
                penalty += 10.0
            if y > 2*h*(1-x) + 1e-4:
                penalty += 10.0
        return -0.001 + penalty
    return -_min_triangle_area(points)

def _local_polish(points, deadline):
    """Local refinement using Nelder-Mead on flattened parameters."""
    points = points.copy()
    current_min_area = _min_triangle_area(points)
    
    for pass_num in range(3):
        if time.time() > deadline - 5:
            break
        
        def objective(params):
            pts = params.reshape(-1, 2)
            if not _all_inside(pts):
                return 10.0
            return -_min_triangle_area(pts)
        
        x0 = points.flatten()
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.success or result.fun < -current_min_area:
                new_points = result.x.reshape(-1, 2)
                if _all_inside(new_points):
                    new_min_area = _min_triangle_area(new_points)
                    if new_min_area > current_min_area:
                        points = new_points
                        current_min_area = new_min_area
        except Exception:
            pass
        
        if time.time() > deadline - 3:
            break
        
        for _ in range(100):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(N):
                for coord in range(2):
                    for step in [0.01, 0.005, 0.001, 0.0005]:
                        old_val = points[i, coord]
                        points[i, coord] = old_val + step
                        if _all_inside(points):
                            new_area = _min_triangle_area(points)
                            if new_area > current_min_area:
                                current_min_area = new_area
                                improved = True
                                continue
                        points[i, coord] = old_val - step
                        if _all_inside(points):
                            new_area = _min_triangle_area(points)
                            if new_area > current_min_area:
                                current_min_area = new_area
                                improved = True
                                continue
                        points[i, coord] = old_val
            if not improved:
                break
    
    return points, current_min_area

def _build_cluster_config():
    """
    Build a 10-point cluster configuration based on known Heilbronn-type arrangements.
    Returns a 10x2 array of relative coordinates centered at origin.
    """
    # Use a hexagonal-like packing: center + 2 rings
    # Ring 1: 6 points at radius r1
    # Ring 2: 3 points at radius r2 (to keep total at 10)
    # Actually, let's use a more structured approach:
    # 3 corners of inner triangle + 3 midpoints + 3 outer + 1 center = 10
    
    h = np.sqrt(3) / 2
    
    # Configuration: 3 vertices of a smaller equilateral triangle,
    # 3 midpoints of its edges, 3 points further out, and 1 center
    # All scaled to fit nicely
    
    # Let's use a known good 10-point pattern:
    # Arrange as 2-3-3-2 rows (like a hexagonal packing)
    # Row 0 (top): 2 points
    # Row 1: 3 points  
    # Row 2: 3 points
    # Row 3 (bottom): 2 points
    
    # Spacing
    dx = 0.25  # horizontal spacing
    dy = dx * np.sqrt(3) / 2  # vertical spacing for hexagonal
    
    # Center the configuration
    points = []
    # Row 0: y = 1.5*dy, x = -dx/2, dx/2
    points.append([-dx/2, 1.5*dy])
    points.append([dx/2, 1.5*dy])
    # Row 1: y = 0.5*dy, x = -dx, 0, dx
    points.append([-dx, 0.5*dy])
    points.append([0, 0.5*dy])
    points.append([dx, 0.5*dy])
    # Row 2: y = -0.5*dy, x = -dx, 0, dx
    points.append([-dx, -0.5*dy])
    points.append([0, -0.5*dy])
    points.append([dx, -0.5*dy])
    # Row 3: y = -1.5*dy, x = -dx/2, dx/2
    points.append([-dx/2, -1.5*dy])
    points.append([dx/2, -1.5*dy])
    
    return np.array(points)

def _rigid_cluster_objective(params):
    """
    Objective for optimizing rigid cluster placement.
    params: [tx, ty, angle, scale, fx, fy]
    tx, ty: translation of cluster center
    angle: rotation angle
    scale: uniform scale factor
    fx, fy: position of the 11th (free) point
    """
    tx, ty, angle, scale, fx, fy = params
    
    # Get base cluster configuration
    base_cluster = _build_cluster_config()
    
    # Scale and rotate
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    # Transform cluster points
    cluster_pts = (base_cluster * scale) @ rot_mat.T + np.array([tx, ty])
    
    # Add free point
    all_points = np.vstack([cluster_pts, [[fx, fy]]])
    
    # Check if all points are inside the triangle
    if not _all_inside(all_points, tol=1e-4):
        # Compute penalty
        penalty = 0.0
        h = np.sqrt(3) / 2
        for p in all_points:
            x, y = p[0], p[1]
            if x < -1e-3 or x > 1+1e-3 or y < -1e-3 or y > h+1e-3:
                penalty += 5.0
            if y > 2*h*x + 1e-3:
                penalty += 5.0
            if y > 2*h*(1-x) + 1e-3:
                penalty += 5.0
        return 1.0 + penalty
    
    min_area = _min_triangle_area(all_points)
    return -min_area

def _optimize_rigid_cluster(deadline):
    """
    Optimize the rigid cluster placement using differential evolution
    in 6D parameter space, then polish all 11 points.
    """
    h = np.sqrt(3) / 2
    
    # Bounds: [tx, ty, angle, scale, fx, fy]
    bounds = [
        (0.1, 0.9),      # tx
        (0.1, h-0.1),    # ty
        (-np.pi, np.pi), # angle
        (0.3, 1.0),      # scale
        (0.0, 1.0),      # fx
        (0.0, h),        # fy
    ]
    
    try:
        de_result = differential_evolution(
            _rigid_cluster_objective, bounds,
            seed=42, maxiter=50, popsize=10,
            tol=1e-6, mutation=(0.5, 1.0), recombination=0.5,
            polish=True
        )
        
        tx, ty, angle, scale, fx, fy = de_result.x
        
        base_cluster = _build_cluster_config()
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        cluster_pts = (base_cluster * scale) @ rot_mat.T + np.array([tx, ty])
        all_points = np.vstack([cluster_pts, [[fx, fy]]])
        
        if _all_inside(all_points):
            polished, min_area = _local_polish(all_points, deadline)
            return polished, min_area
    except Exception:
        pass
    
    return None, 0.0

def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    h = np.sqrt(3) / 2
    
    best_points = None
    best_min_area = 0.0
    
    # Strategy 1: Rigid cluster optimization (reduced dimensionality)
    if time.time() < deadline - 40:
        cluster_points, cluster_min_area = _optimize_rigid_cluster(deadline)
        if cluster_points is not None and cluster_min_area > best_min_area:
            best_min_area = cluster_min_area
            best_points = cluster_points.copy()
    
    # Strategy 2: Differential evolution for full 22D search
    if best_points is None or time.time() < deadline - 30:
        bounds = [(0, 1)] * 2 * N
        
        try:
            de_result = differential_evolution(
                _negative_min_area, bounds,
                seed=42, maxiter=100, popsize=15,
                tol=1e-8, mutation=(0.5, 1.5), recombination=0.7,
                polish=True, init='sobol'
            )
            de_points = de_result.x.reshape(-1, 2)
            if _all_inside(de_points):
                de_min_area = _min_triangle_area(de_points)
                if de_min_area > best_min_area:
                    best_min_area = de_min_area
                    best_points = de_points.copy()
        except Exception:
            pass
    
    # Strategy 3: Try multiple random initializations with local polish
    if best_points is None or time.time() < deadline - 20:
        rng = np.random.RandomState(123)
        
        for seed in range(8):
            if time.time() > deadline - 15:
                break
            
            points = []
            for _ in range(N):
                while True:
                    u, v = rng.random(), rng.random()
                    if u + v < 0.95 and u > 0.02 and v > 0.02:
                        px = u + 0.5 * v
                        py = h * v
                        points.append([px, py])
                        break
            
            points = np.array(points)
            
            polished, min_area = _local_polish(points, deadline)
            
            if min_area > best_min_area:
                best_min_area = min_area
                best_points = polished
    
    # Strategy 4: Structured initializations
    if best_points is not None and time.time() < deadline - 10:
        structured_configs = []
        
        cx, cy = 0.5, h/3
        for r in [0.0, 0.12, 0.22, 0.32]:
            if r == 0.0:
                structured_configs.append([cx, cy])
            else:
                n_pts = int(2 * np.pi * r / 0.15)
                angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
                for a in angles:
                    px = cx + r*np.cos(a)
                    py = cy + r*np.sin(a)*0.8
                    if _inside_triangle([px, py]):
                        structured_configs.append([px, py])
        
        if len(structured_configs) >= N:
            points = np.array(structured_configs[:N])
            polished, min_area = _local_polish(points, deadline)
            if min_area > best_min_area:
                best_min_area = min_area
                best_points = polished
    
    # Final polish on the best result
    if best_points is not None and time.time() < deadline - 5:
        best_points, best_min_area = _local_polish(best_points, deadline)
    
    return best_points
# EVOLVE-BLOCK-END
