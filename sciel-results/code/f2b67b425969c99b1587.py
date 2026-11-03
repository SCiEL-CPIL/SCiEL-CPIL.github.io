"""Program for sphere covering with 50 points using Gram-matrix optimization."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.optimize import linprog

N = 50
DEADLINE_MARGIN = 5.0  # seconds before the 150s limit to stop

def _to_sphere(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(pts):
    """
    Compute the covering radius in radians using convex hull facet circumradii.
    Returns np.pi if invalid.
    """
    pts = _to_sphere(pts)
    n = len(pts)
    
    # Check for coincident points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-8:
                return np.pi
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    
    simplices = hull.simplices
    if len(simplices) == 0:
        return np.pi
    
    max_radius = 0.0
    for simplex in simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Compute normal to the facet
        v1 = b - a
        v2 = c - a
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm
        
        # Check which side (normal or -normal) gives smaller max distance
        dot_a = np.clip(normal @ a, -1, 1)
        dot_b = np.clip(normal @ b, -1, 1)
        dot_c = np.clip(normal @ c, -1, 1)
        
        # Distances for +normal
        da = np.arccos(dot_a)
        db = np.arccos(dot_b)
        dc = np.arccos(dot_c)
        max_pos = max(da, db, dc)
        
        # Distances for -normal
        da_neg = np.arccos(-dot_a)
        db_neg = np.arccos(-dot_b)
        dc_neg = np.arccos(-dot_c)
        max_neg = max(da_neg, db_neg, dc_neg)
        
        radius = min(max_pos, max_neg)
        if radius > max_radius:
            max_radius = radius
    
    return max_radius

def _fibonacci_spiral(n, offset=0.0):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=None):
    """Generate n random points on the sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    pts = rng.randn(n, 3)
    return _to_sphere(pts)

def _lloyd_relax(pts, max_iter=200, deadline=None):
    """
    Lloyd's relaxation: repeatedly move each point to the centroid of its Voronoi cell.
    For sphere covering, we use the centroid of the spherical Voronoi cell,
    approximated by the centroid of the nearest points.
    """
    pts = _to_sphere(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        new_pts = np.zeros_like(pts)
        for i in range(n):
            # Find the centroid of the region where pts[i] is the nearest
            # Approximate by averaging pts[i] with nearby points weighted by inverse distance
            # Better: use the spherical centroid of the Voronoi cell
            # Simple approach: move towards the centroid of the farthest point in the cell
            # For now, use a simple repulsion-based relaxation
            
            # Compute pairwise distances
            dists = np.array([np.arccos(np.clip(pts[i] @ pts[j], -1, 1)) for j in range(n)])
            dists[i] = np.inf
            
            # Find the point that is farthest from pts[i] among its neighbors
            # and move towards it
            # Actually, let's use a proper Voronoi-based approach
            # For each point, find the centroid of its Voronoi cell on the sphere
            
            # Spherical Voronoi cell centroid approximation:
            # The cell consists of points y on the sphere where d(y, pts[i]) <= d(y, pts[j]) for all j
            # This is equivalent to y . pts[i] >= y . pts[j] for all j
            # The centroid can be approximated by solving a convex optimization,
            # but for speed, we use the following heuristic:
            # Move pts[i] towards the direction that maximizes the minimum distance to other points
            
            # Gradient ascent on min_j d(pts[i], pts[j])
            # d(pts[i], pts[j]) = arccos(pts[i] . pts[j])
            # grad_{pts[i]} min_j d(pts[i], pts[j]) = -pts[j_k] / sqrt(1 - (pts[i].pts[j_k])^2)
            # where j_k is the index of the nearest point
            
            # Find nearest neighbor
            nearest_j = np.argmin(dists)
            dot_val = np.clip(pts[i] @ pts[nearest_j], -1, 1)
            denom = np.sqrt(1.0 - dot_val * dot_val)
            if denom < 1e-8:
                denom = 1e-8
            
            # Move away from nearest neighbor
            step_dir = -pts[nearest_j] / denom
            # Project onto tangent space
            step_dir = step_dir - np.dot(step_dir, pts[i]) * pts[i]
            step_dir = _to_sphere(step_dir.reshape(1, 3))[0]
            
            # Adaptive step size
            step_size = 0.05
            new_pts[i] = pts[i] + step_size * step_dir
            new_pts[i] = _to_sphere(new_pts[i].reshape(1, 3))[0]
        
        pts = _to_sphere(new_pts)
        
        # Check convergence
        if it > 10:
            change = np.linalg.norm(new_pts - pts)
            if change < 1e-6:
                break
    
    return pts

def _gradient_optimize(pts, max_iter=50, deadline=None):
    """
    Gradient-based optimization using a smooth approximation of the covering radius.
    Uses L-BFGS-B on the flat vector representation.
    """
    current_pts = _to_sphere(pts.copy())
    
    def objective(flat_pts):
        pts_local = _to_sphere(flat_pts.reshape(N, 3))
        # Compute covering radius
        radius = _covering_radius(pts_local)
        if not np.isfinite(radius):
            return 1e10
        return radius
    
    def gradient(flat_pts):
        pts_local = _to_sphere(flat_pts.reshape(N, 3))
        n = len(pts_local)
        
        # Compute the covering radius and identify the worst facet
        try:
            hull = ConvexHull(pts_local)
        except Exception:
            return np.zeros_like(flat_pts)
        
        simplices = hull.simplices
        if len(simplices) == 0:
            return np.zeros_like(flat_pts)
        
        # Find the facet with the largest circumradius
        max_radius = 0.0
        worst_simplex = None
        worst_pole = None
        worst_active_vertex = None
        worst_active_idx = None
        
        for simplex in simplices:
            a = pts_local[simplex[0]]
            b = pts_local[simplex[1]]
            c = pts_local[simplex[2]]
            
            v1 = b - a
            v2 = c - a
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            
            dot_a = np.clip(normal @ a, -1, 1)
            dot_b = np.clip(normal @ b, -1, 1)
            dot_c = np.clip(normal @ c, -1, 1)
            
            da = np.arccos(dot_a)
            db = np.arccos(dot_b)
            dc = np.arccos(dot_c)
            max_pos = max(da, db, dc)
            
            da_neg = np.arccos(-dot_a)
            db_neg = np.arccos(-dot_b)
            dc_neg = np.arccos(-dot_c)
            max_neg = max(da_neg, db_neg, dc_neg)
            
            if max_pos <= max_neg:
                radius = max_pos
                pole = normal
                distances = [da, db, dc]
                vertices = [a, b, c]
                indices = [simplex[0], simplex[1], simplex[2]]
            else:
                radius = max_neg
                pole = -normal
                distances = [da_neg, db_neg, dc_neg]
                vertices = [a, b, c]
                indices = [simplex[0], simplex[1], simplex[2]]
            
            if radius > max_radius:
                max_radius = radius
                worst_simplex = simplex
                worst_pole = pole
                active_k = np.argmax(distances)
                worst_active_vertex = vertices[active_k]
                worst_active_idx = indices[active_k]
        
        if worst_simplex is None:
            return np.zeros_like(flat_pts)
        
        # Gradient of the covering radius w.r.t. the active vertex
        dot_val = np.clip(worst_pole @ worst_active_vertex, -1, 1)
        denom = np.sqrt(1.0 - dot_val * dot_val)
        if denom < 1e-8:
            denom = 1e-8
        
        # grad_{v} arccos(p . v) = -p / sqrt(1 - (p.v)^2)
        grad_v = -worst_pole / denom
        
        # Only the active vertex contributes to the gradient (first-order approximation)
        grad_flat = np.zeros_like(flat_pts)
        grad_flat[worst_active_idx * 3:(worst_active_idx + 1) * 3] = grad_v
        
        return grad_flat
    
    try:
        result = minimize(objective, current_pts.flatten(), method='L-BFGS-B',
                          jac=gradient, options={'maxiter': max_iter, 'gtol': 1e-6})
        new_pts = _to_sphere(result.x.reshape(N, 3))
        new_val = _covering_radius(new_pts)
        
        current_val = _covering_radius(current_pts)
        if new_val < current_val:
            return new_pts, new_val
    except Exception:
        pass
    
    return current_pts, _covering_radius(current_pts)

def _local_refine(pts, max_iter=100, step_size=0.02, deadline=None):
    """
    Local optimization with adaptive step size.
    """
    best_pts = _to_sphere(pts.copy())
    best_val = _covering_radius(best_pts)
    current_step = step_size
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb a random point
        idx = np.random.randint(len(best_pts))
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        new_pts = best_pts.copy()
        new_pts[idx] = best_pts[idx] + current_step * direction
        new_pts[idx] = _to_sphere(new_pts[idx].reshape(1, 3))[0]
        
        new_val = _covering_radius(new_pts)
        
        if new_val < best_val:
            best_pts = new_pts
            best_val = new_val
            current_step *= 1.02
        else:
            current_step *= 0.95
            
        if current_step < 1e-7:
            break
    
    return best_pts, best_val

def _simulated_annealing(pts, max_iter=500, deadline=None):
    """
    Simulated annealing for global optimization.
    """
    best_pts = _to_sphere(pts.copy())
    best_val = _covering_radius(best_pts)
    current_pts = best_pts.copy()
    current_val = best_val
    
    T = 0.1  # Initial temperature
    T_min = 1e-6
    cooling = 0.99
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb a random point
        idx = np.random.randint(len(current_pts))
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        new_pts = current_pts.copy()
        new_pts[idx] = current_pts[idx] + 0.05 * direction
        new_pts[idx] = _to_sphere(new_pts[idx].reshape(1, 3))[0]
        
        new_val = _covering_radius(new_pts)
        
        # Accept if better, or with probability exp(-(new_val - current_val) / T)
        if new_val < current_val:
            current_pts = new_pts
            current_val = new_val
            if current_val < best_val:
                best_pts = new_pts
                best_val = current_val
        else:
            delta = new_val - current_val
            if np.random.rand() < np.exp(-delta / T):
                current_pts = new_pts
                current_val = new_val
        
        T *= cooling
        if T < T_min:
            break
    
    return best_pts, best_val

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    best_pts = None
    best_val = np.pi
    
    # Strategy 1: Fibonacci spiral
    pts = _fibonacci_spiral(N)
    val = _covering_radius(pts)
    if val < best_val:
        best_val = val
        best_pts = pts.copy()
    
    # Lloyd relaxation from Fibonacci
    if time.time() < deadline - 30.0:
        lloyd_pts = _lloyd_relax(pts, max_iter=100, deadline=deadline)
        lloyd_val = _covering_radius(lloyd_pts)
        if lloyd_val < best_val:
            best_val = lloyd_val
            best_pts = lloyd_pts.copy()
    
    # Gradient refinement
    if time.time() < deadline - 20.0:
        grad_pts, grad_val = _gradient_optimize(best_pts, max_iter=30, deadline=deadline)
        if grad_val < best_val:
            best_val = grad_val
            best_pts = grad_pts.copy()
    
    # Local refinement
    if time.time() < deadline - 10.0:
        local_pts, local_val = _local_refine(best_pts, max_iter=100, step_size=0.02, deadline=deadline)
        if local_val < best_val:
            best_val = local_val
            best_pts = local_pts.copy()
    
    # Simulated annealing from multiple starting points
    if time.time() < deadline - 5.0:
        for seed in range(3):
            if time.time() > deadline:
                break
            random_pts = _random_points(N, seed=seed)
            sa_pts, sa_val = _simulated_annealing(random_pts, max_iter=200, deadline=deadline)
            if sa_val < best_val:
                best_val = sa_val
                best_pts = sa_pts.copy()
    
    return _to_sphere(best_pts)
