"""Heilbronn triangles: place 11 points to maximize the minimum triangle area."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11
S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0
DEADLINE = 145.0

def _elapsed(start_time):
    return time.time() - start_time

def _min_triangle_area(points):
    """Compute minimum area and identify the argmin triangle (i,j,k)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    min_area = float('inf')
    min_ijk = (0, 1, 2)
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                area = 0.5 * abs((x[j]-x[i])*(y[k]-y[i]) - (y[j]-y[i])*(x[k]-x[i]))
                if area < min_area:
                    min_area = area
                    min_ijk = (i, j, k)
    
    return min_area, min_ijk

def _all_areas(points):
    """Compute all triangle areas as a sorted array with indices."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                area = 0.5 * abs((x[j]-x[i])*(y[k]-y[i]) - (y[j]-y[i])*(x[k]-x[i]))
                areas.append((area, (i, j, k)))
    
    areas.sort(key=lambda t: t[0])
    return areas

def _min_area_normalized(points):
    """Compute minimum triangle area normalized by the equilateral triangle area."""
    min_area, _ = _min_triangle_area(points)
    return min_area / TRI_AREA

def _cartesian_to_barycentric(points):
    """Convert Cartesian to barycentric coordinates."""
    x = points[:, 0]
    y = points[:, 1]
    w = y / S3H
    u = 1.0 - x - 0.5 * w
    v = x - 0.5 * w
    return np.column_stack([u, v, w])

def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates to Cartesian."""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = S3H * bary[:, 2]
    return np.column_stack([x, y])

def _project_to_triangle(points):
    """Project points to be inside the triangle using barycentric coordinates."""
    bary = _cartesian_to_barycentric(points)
    bary = np.maximum(bary, 0)
    s = np.sum(bary, axis=1, keepdims=True)
    s[s == 0] = 1
    bary = bary / s
    return _barycentric_to_cartesian(bary)

def _push_apart(points, strength=0.01, max_iter=5000, start_time=0, deadline=145):
    """
    Push-apart local search: identify the minimum area triangle and push its
    vertices apart to increase that area.
    """
    current = points.copy()
    current_min, current_ijk = _min_triangle_area(current)
    best = current.copy()
    best_min = current_min
    
    for it in range(max_iter):
        if _elapsed(start_time) > deadline - 5:
            break
        
        # Find the minimum area triangle
        min_area, (i, j, k) = _min_triangle_area(current)
        
        # Compute the normal to the triangle (perpendicular in 2D)
        p_i = current[i]
        p_j = current[j]
        p_k = current[k]
        
        # Edge vectors
        e1 = p_j - p_i
        e2 = p_k - p_i
        
        # Signed area (positive = CCW)
        signed_area = 0.5 * (e1[0]*e2[1] - e1[1]*e2[0])
        
        # Direction to push: for each vertex, push away from the centroid of the triangle
        centroid = (p_i + p_j + p_k) / 3.0
        
        # Push each vertex away from centroid, but also consider the specific geometry
        # For vertex i: push in direction that increases area
        # If signed_area > 0 (CCW), pushing i in direction perpendicular to jk away from i increases area
        # Simpler: push each vertex away from the other two
        
        # Push i away from midpoint of jk
        mid_jk = (p_j + p_k) / 2.0
        dir_i = p_i - mid_jk
        norm_i = np.linalg.norm(dir_i)
        if norm_i > 1e-10:
            dir_i /= norm_i
        else:
            dir_i = np.array([1.0, 0.0])
        
        # Push j away from midpoint of ik
        mid_ik = (p_i + p_k) / 2.0
        dir_j = p_j - mid_ik
        norm_j = np.linalg.norm(dir_j)
        if norm_j > 1e-10:
            dir_j /= norm_j
        else:
            dir_j = np.array([0.0, 1.0])
        
        # Push k away from midpoint of ij
        mid_ij = (p_i + p_j) / 2.0
        dir_k = p_k - mid_ij
        norm_k = np.linalg.norm(dir_k)
        if norm_k > 1e-10:
            dir_k /= norm_k
        else:
            dir_k = np.array([-0.5, np.sqrt(3)/2])
        
        # Apply push
        step = strength * (1.0 + 0.5 * it / max_iter)
        new_pts = current.copy()
        new_pts[i] += step * dir_i
        new_pts[j] += step * dir_j
        new_pts[k] += step * dir_k
        
        # Project to triangle
        new_pts = _project_to_triangle(new_pts)
        
        new_min, _ = _min_triangle_area(new_pts)
        
        if new_min >= current_min - 1e-12:
            current = new_pts
            current_min = new_min
            if current_min > best_min:
                best_min = current_min
                best = current.copy()
        else:
            # Try smaller step
            step2 = step * 0.1
            new_pts2 = current.copy()
            new_pts2[i] += step2 * dir_i
            new_pts2[j] += step2 * dir_j
            new_pts2[k] += step2 * dir_k
            new_pts2 = _project_to_triangle(new_pts2)
            new_min2, _ = _min_triangle_area(new_pts2)
            
            if new_min2 >= current_min - 1e-12:
                current = new_pts2
                current_min = new_min2
                if current_min > best_min:
                    best_min = current_min
                    best = current.copy()
    
    return best, best_min / TRI_AREA

def _sobol_sequence(n_points, n_dims, seed=42):
    """Generate low-discrepancy points in [0,1]^n_dims using a simple quasi-random approach."""
    rng = np.random.RandomState(seed)
    
    # Use Halton-like sequence for 2D
    points = []
    for i in range(1, n_points + 1):
        # Van der Corput sequence for 2D
        def van_der_corput(n, base):
            f = 1.0
            r = 0.0
            while n > 0:
                f /= base
                r += f * (n % base)
                n //= base
            return r
        
        x1 = van_der_corput(i, 2)
        x2 = van_der_corput(i, 3)
        points.append([x1, x2])
    
    return np.array(points)

def _random_in_triangle(rng, n):
    """Generate n random points in the triangle."""
    pts = []
    for _ in range(n):
        while True:
            u, v = rng.random(), rng.random()
            if u + v < 1.0:
                break
        x = v + 0.5 * (1 - u - v)
        y = S3H * (1 - u - v)
        pts.append([x, y])
    return np.array(pts)

def _layered_initialization(rng):
    """
    Initialize points in a 3-4-4 layer structure relative to the triangle.
    Layer 1 (bottom): 3 points near the base
    Layer 2 (middle): 4 points in the middle band
    Layer 3 (top): 4 points near the apex
    """
    pts = []
    
    # Layer 1: 3 points near the base (y ~ 0 to 0.2*S3H)
    for i in range(3):
        x = 0.15 + 0.35 * i + rng.normal(0, 0.02)
        y = rng.uniform(0.02, 0.12)
        pts.append([x, y])
    
    # Layer 2: 4 points in the middle (y ~ 0.2*S3H to 0.5*S3H)
    for i in range(4):
        x = 0.2 + 0.25 * i + rng.normal(0, 0.02)
        y = rng.uniform(0.25, 0.45) * S3H
        # Adjust x to stay in triangle at this height
        max_x_at_y = 0.5 + 0.5 * (1 - y/S3H)
        min_x_at_y = 0.5 - 0.5 * (1 - y/S3H)
        x = np.clip(x, min_x_at_y + 0.02, max_x_at_y - 0.02)
        pts.append([x, y])
    
    # Layer 3: 4 points near the top (y ~ 0.5*S3H to 0.9*S3H)
    for i in range(4):
        x = 0.3 + 0.15 * i + rng.normal(0, 0.02)
        y = rng.uniform(0.55, 0.85) * S3H
        max_x_at_y = 0.5 + 0.5 * (1 - y/S3H)
        min_x_at_y = 0.5 - 0.5 * (1 - y/S3H)
        x = np.clip(x, min_x_at_y + 0.02, max_x_at_y - 0.02)
        pts.append([x, y])
    
    return np.array(pts)

def _nelder_mead_optimize(points, start_time, deadline=145):
    """Run Nelder-Mead optimization on the points."""
    def objective(flat):
        pts = flat.reshape(N, 2)
        bary = _cartesian_to_barycentric(pts)
        penalty = 1000.0 * (np.sum(np.maximum(0, -bary[:, 0])**2) +
                           np.sum(np.maximum(0, -bary[:, 1])**2) +
                           np.sum(np.maximum(0, -bary[:, 2])**2))
        min_a, _ = _min_triangle_area(pts)
        return -min_a / TRI_AREA + penalty
    
    try:
        res = minimize(
            objective,
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        polished = res.x.reshape(N, 2)
        polished = _project_to_triangle(polished)
        score = _min_area_normalized(polished)
        return polished, score
    except Exception:
        return points, _min_area_normalized(points)

def _slp_iteration(points, start_time, deadline=145, n_iters=100):
    """
    SLP (Sequential Linear Programming) style iteration:
    At each step, identify the minimum area triangle, add a linear constraint
    that its area >= current_min + epsilon, and solve the resulting problem.
    Approximated by iterative push-apart with constraint enforcement.
    """
    current = points.copy()
    current_min, _ = _min_triangle_area(current)
    best = current.copy()
    best_min = current_min
    
    for it in range(n_iters):
        if _elapsed(start_time) > deadline - 5:
            break
        
        # Find the k smallest area triangles
        areas_list = _all_areas(current)
        k_small = areas_list[:5]
        
        # Push apart each of the k smallest triangles
        improved = False
        for area_val, (i, j, k) in k_small:
            if _elapsed(start_time) > deadline - 5:
                break
            
            p_i = current[i]
            p_j = current[j]
            p_k = current[k]
            
            # Push each vertex away from the other two
            mid_jk = (p_j + p_k) / 2.0
            dir_i = p_i - mid_jk
            norm_i = np.linalg.norm(dir_i)
            if norm_i > 1e-10:
                dir_i /= norm_i
            
            mid_ik = (p_i + p_k) / 2.0
            dir_j = p_j - mid_ik
            norm_j = np.linalg.norm(dir_j)
            if norm_j > 1e-10:
                dir_j /= norm_j
            
            mid_ij = (p_i + p_j) / 2.0
            dir_k = p_k - mid_ij
            norm_k = np.linalg.norm(dir_k)
            if norm_k > 1e-10:
                dir_k /= norm_k
            
            step = 0.005
            new_pts = current.copy()
            new_pts[i] += step * dir_i
            new_pts[j] += step * dir_j
            new_pts[k] += step * dir_k
            new_pts = _project_to_triangle(new_pts)
            
            new_min, _ = _min_triangle_area(new_pts)
            
            if new_min > current_min + 1e-10:
                current = new_pts
                current_min = new_min
                improved = True
                if current_min > best_min:
                    best_min = current_min
                    best = current.copy()
        
        if not improved:
            break
    
    return best, best_min / TRI_AREA

def construct():
    start_time = time.time()
    
    best_points = None
    best_score = -float('inf')
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Layered initialization
    layered = _layered_initialization(rng)
    initial_configs.append(layered)
    
    # 2. Random points
    for _ in range(3):
        initial_configs.append(_random_in_triangle(rng, N))
    
    # 3. Low-discrepancy based
    sobol_pts = _sobol_sequence(N, 2, seed=42)
    cart_pts = np.column_stack([
        sobol_pts[:, 1] + 0.5 * (1 - sobol_pts[:, 0] - sobol_pts[:, 1]),
        S3H * (1 - sobol_pts[:, 0] - sobol_pts[:, 1])
    ])
    initial_configs.append(cart_pts)
    
    # 4. Symmetric configuration (known good starting point)
    # Place points in a roughly symmetric pattern
    sym_pts = np.array([
        [0.5, 0.05],          # bottom center
        [0.2, 0.1],           # bottom left
        [0.8, 0.1],           # bottom right
        [0.35, 0.3],          # middle left
        [0.65, 0.3],          # middle right
        [0.5, 0.4],           # middle center
        [0.25, 0.5],          # upper left
        [0.75, 0.5],          # upper right
        [0.4, 0.65],          # top left
        [0.6, 0.65],          # top right
        [0.5, 0.75],          # top center
    ])
    initial_configs.append(sym_pts)
    
    # Phase 1: Push-apart + Nelder-Mead from each initial config
    for idx, init_pts in enumerate(initial_configs):
        if _elapsed(start_time) > DEADLINE - 30:
            break
        
        # Push-apart local search
        pushed, push_score = _push_apart(init_pts, strength=0.005, max_iter=2000, start_time=start_time, deadline=DEADLINE)
        
        if push_score > best_score:
            best_score = push_score
            best_points = pushed.copy()
        
        # Nelder-Mead polish
        if _elapsed(start_time) > DEADLINE - 20:
            break
        
        polished, nm_score = _nelder_mead_optimize(pushed, start_time, DEADLINE)
        
        if nm_score > best_score:
            best_score = nm_score
            best_points = polished.copy()
    
    # Phase 2: SLP iterations from best result
    if best_points is not None and _elapsed(start_time) < DEADLINE - 15:
        slp_pts, slp_score = _slp_iteration(best_points, start_time, DEADLINE, n_iters=200)
        
        if slp_score > best_score:
            best_score = slp_score
            best_points = slp_pts.copy()
        
        # Final Nelder-Mead polish
        if _elapsed(start_time) < DEADLINE - 5:
            polished, nm_score = _nelder_mead_optimize(best_points, start_time, DEADLINE)
            if nm_score > best_score:
                best_score = nm_score
                best_points = polished.copy()
    
    # Phase 3: Additional multi-start if time permits
    if _elapsed(start_time) < DEADLINE - 25:
        for _ in range(5):
            if _elapsed(start_time) > DEADLINE - 10:
                break
            
            # Perturb best and re-optimize
            perturbed = best_points + rng.normal(0, 0.03, size=(N, 2))
            perturbed = _project_to_triangle(perturbed)
            
            pushed, push_score = _push_apart(perturbed, strength=0.003, max_iter=1000, start_time=start_time, deadline=DEADLINE)
            
            if push_score > best_score:
                best_score = push_score
                best_points = pushed.copy()
    
    # Final polish
    if best_points is not None and _elapsed(start_time) < DEADLINE - 3:
        polished, nm_score = _nelder_mead_optimize(best_points, start_time, DEADLINE)
        if nm_score > best_score:
            best_score = nm_score
            best_points = polished.copy()
    
    # Fallback
    if best_points is None:
        best_points = _random_in_triangle(rng, N)
    
    return best_points
