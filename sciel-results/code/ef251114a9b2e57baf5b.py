import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0

_indices = list(itertools.combinations(range(N), 3))
_idx_arr = np.array(_indices)
N_TRI = len(_indices)

def _compute_areas(points):
    """Compute all 165 triangle areas from 11 points using vectorized operations."""
    i = _idx_arr[:, 0]
    j = _idx_arr[:, 1]
    k = _idx_arr[:, 2]
    
    xi, yi = points[i, 0], points[i, 1]
    xj, yj = points[j, 0], points[j, 1]
    xk, yk = points[k, 0], points[k, 1]
    
    det = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
    areas = 0.5 * np.abs(det)
    return areas

def _min_area(points):
    """Return the minimum triangle area."""
    return float(np.min(_compute_areas(points)))

def _min_area_and_indices(points):
    """Return (min_area, list of indices of points involved in min triangles)."""
    areas = _compute_areas(points)
    min_area = float(np.min(areas))
    threshold = min_area + 1e-10
    min_indices = np.where(areas <= threshold)[0]
    involved = set()
    for idx in min_indices:
        involved.add(int(_idx_arr[idx, 0]))
        involved.add(int(_idx_arr[idx, 1]))
        involved.add(int(_idx_arr[idx, 2]))
    return min_area, sorted(involved)

def _bary_to_cart(bary):
    """Convert barycentric coords (u,v) array to Cartesian points."""
    u = np.clip(bary[:, 0], 0.0, None)
    v = np.clip(bary[:, 1], 0.0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
        
    pts = np.empty((N, 2))
    pts[:, 0] = v * 1.0 + (1.0 - u - v) * 0.5
    pts[:, 1] = (1.0 - u - v) * H
    return pts

def _cart_to_bary(pts):
    """Convert Cartesian points to barycentric coords (u, v)."""
    bary = np.empty((N, 2))
    for m in range(N):
        x = pts[m, 0]
        y = pts[m, 1]
        bary[m, 0] = 1.0 - x - y / (2.0 * H)
        bary[m, 1] = x - y / (2.0 * H)
    return bary

def _project_to_triangle(points):
    """Project points that are slightly outside the triangle back inside."""
    eps = 1e-10
    for m in range(N):
        x, y = points[m, 0], points[m, 1]
        if y < -eps:
            points[m, 1] = 0.0
        if y + SQRT3 * x > SQRT3 + eps:
            d = (y + SQRT3 * x - SQRT3) / 2.0
            if d > 0:
                points[m, 0] -= SQRT3 * d
                points[m, 1] -= d
        if y > SQRT3 * x + eps:
            d = (SQRT3 * x - y) / 2.0
            if d < 0:
                points[m, 0] += SQRT3 * d
                points[m, 1] -= d
    for m in range(N):
        x, y = points[m, 0], points[m, 1]
        if y < 0:
            points[m, 1] = 0
        if y > SQRT3 * x:
            if x > 0:
                points[m, 1] = SQRT3 * x
            else:
                points[m, 1] = 0
        if y > SQRT3 * (1 - x):
            if x < 1:
                points[m, 1] = SQRT3 * (1 - x)
            else:
                points[m, 1] = 0
    return points

def _sobol_points_in_triangle(n, seed=42):
    """Generate n quasi-random points in the equilateral triangle using Sobol sequence."""
    from scipy.stats import qmc
    rng = qmc.Sobol(d=2, seed=seed)
    n_gen = n * 4
    samples = rng.random(n_gen)
    u = samples[:, 0]
    v = samples[:, 1]
    mask = (u >= 0) & (v >= 0) & (u + v <= 1)
    valid_u = u[mask]
    valid_v = v[mask]
    if len(valid_u) >= n:
        u = valid_u[:n]
        v = valid_v[:n]
    else:
        while len(valid_u) < n:
            more = rng.random(n_gen)
            u2, v2 = more[:, 0], more[:, 1]
            mask2 = (u2 >= 0) & (v2 >= 0) & (u2 + v2 <= 1)
            valid_u = np.concatenate([valid_u, u2[mask2]])
            valid_v = np.concatenate([valid_v, v2[mask2]])
        u = valid_u[:n]
        v = valid_v[:n]
    bary = np.column_stack([u, v])
    return _bary_to_cart(bary)

def _soft_min_area(points, beta=50.0):
    """
    Smooth approximation of the minimum area using log-sum-exp.
    """
    areas = _compute_areas(points)
    min_a = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - min_a))))
    return min_a - np.log(np.sum(np.exp(-beta * (areas - min_a)))) / beta

def _objective(params, beta=50.0):
    """
    Objective function for optimization: negative of soft min area.
    """
    points = params.reshape(N, 2)
    points = _project_to_triangle(points.copy())
    return -_soft_min_area(points, beta)

def _nelder_mead_optimize(points, time_limit=10.0, beta=50.0):
    """
    Use Nelder-Mead to optimize the soft min area objective.
    """
    start_t = time.time()
    x0 = points.flatten()
    
    result = minimize(
        _objective,
        x0,
        method='Nelder-Mead',
        args=(beta,),
        options={
            'maxiter': 2000,
            'xatol': 1e-8,
            'fatol': 1e-10,
            'adaptive': True
        }
    )
    
    elapsed = time.time() - start_t
    if elapsed > time_limit:
        return points, _min_area(points)
    
    new_points = result.x.reshape(N, 2)
    new_points = _project_to_triangle(new_points)
    score = _min_area(new_points)
    
    if score > _min_area(points):
        return new_points, score
    else:
        return points, _min_area(points)

def _local_nudge_search(points, max_iter=500, step_sizes=None, time_limit=None):
    """
    Local search that only moves points involved in the smallest triangles.
    """
    if step_sizes is None:
        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    current_points = points.copy()
    current_min, _ = _min_area_and_indices(current_points)
    best_points = current_points.copy()
    best_min = current_min
    
    if time_limit is None:
        time_limit = 60.0
    start_t = time.time()
    
    for step_size in step_sizes:
        if time.time() - start_t > time_limit:
            break
        
        for _ in range(max_iter):
            if time.time() - start_t > time_limit:
                break
            
            min_area, involved = _min_area_and_indices(current_points)
            
            if min_area > best_min:
                best_min = min_area
                best_points = current_points.copy()
            
            improved = False
            for idx in involved:
                if time.time() - start_t > time_limit:
                    break
                
                for _ in range(20):
                    du = np.random.uniform(-step_size, step_size)
                    dv = np.random.uniform(-step_size, step_size)
                    
                    bary = _cart_to_bary(current_points)
                    new_u = bary[idx, 0] + du
                    new_v = bary[idx, 1] + dv
                    
                    new_u = max(0.0, new_u)
                    new_v = max(0.0, new_v)
                    s = new_u + new_v
                    if s > 1.0:
                        new_u /= s
                        new_v /= s
                    
                    new_x = new_v * 1.0 + (1.0 - new_u - new_v) * 0.5
                    new_y = (1.0 - new_u - new_v) * H
                    
                    if new_y < -1e-10 or new_y > SQRT3 * new_x + 1e-10 or new_y > SQRT3 * (1 - new_x) + 1e-10:
                        continue
                    
                    test_points = current_points.copy()
                    test_points[idx, 0] = new_x
                    test_points[idx, 1] = new_y
                    test_min = _min_area(test_points)
                    
                    if test_min > min_area + 1e-12:
                        current_points = test_points
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
    
    return best_points, best_min

def _initial_guess_grid():
    """Generate a structured grid-like initial configuration."""
    positions = []
    for i in range(3):
        x = 0.1 + i * 0.3
        positions.append((x, 0.05))
    for i in range(4):
        x = 0.05 + i * 0.25
        y = 0.25
        if y <= SQRT3 * x and y <= SQRT3 * (1 - x):
            positions.append((x, y))
    for i in range(3):
        x = 0.15 + i * 0.25
        y = 0.45
        if y <= SQRT3 * x and y <= SQRT3 * (1 - x):
            positions.append((x, y))
    positions.append((0.5, 0.6))
    
    valid_positions = []
    for p in positions:
        x, y = p
        if y >= -1e-10 and y <= SQRT3 * x + 1e-10 and y <= SQRT3 * (1 - x) + 1e-10:
            valid_positions.append(p)
    
    if len(valid_positions) >= N:
        return np.array(valid_positions[:N])
    else:
        return _sobol_points_in_triangle(N, seed=0)

def _push_apart_gradient(points):
    """
    Compute a gradient-like direction that pushes points apart,
    focusing on the smallest triangles.
    """
    areas = _compute_areas(points)
    min_a = np.min(areas)
    
    # Weight each triangle by how close it is to the minimum
    weights = np.exp(-100.0 * (areas - min_a))
    weights /= np.sum(weights)
    
    grad = np.zeros((N, 2))
    for t in range(N_TRI):
        i, j, k = int(_idx_arr[t, 0]), int(_idx_arr[t, 1]), int(_idx_arr[t, 2])
        w = weights[t]
        
        pi = points[i]
        pj = points[j]
        pk = points[k]
        
        # Area = 0.5 * |det(pj-pi, pk-pi)|
        # To increase area, we want to increase |det|
        # The sign of the determinant tells us the orientation
        det = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
        sign = np.sign(det)
        
        # Gradient of det with respect to each point
        # d(det)/d(pi) = -(pk - pj) rotated 90 deg
        # d(det)/d(pj) = (pi - pk) rotated 90 deg
        # d(det)/d(pk) = (pj - pi) rotated 90 deg
        
        def rot90(v):
            return np.array([-v[1], v[0]])
        
        gpi = rot90(pk - pj)
        gpj = rot90(pi - pk)
        gpk = rot90(pj - pi)
        
        grad[i] += w * sign * gpi
        grad[j] += w * sign * gpj
        grad[k] += w * sign * gpk
    
    return grad

def _active_set_optimize(points, time_limit=15.0):
    """
    Active-set style optimization: identify critical triangles,
    then optimize by pushing those triangles' area up while
    keeping others from decreasing.
    """
    start_t = time.time()
    current = points.copy()
    best = points.copy()
    best_score = _min_area(current)
    
    max_iter = 200
    
    for it in range(max_iter):
        if time.time() - start_t > time_limit:
            break
        
        min_area, involved = _min_area_and_indices(current)
        
        if min_area > best_score:
            best_score = min_area
            best = current.copy()
        
        # Get the gradient direction
        grad = _push_apart_gradient(current)
        
        # Try to move in the gradient direction with adaptive step size
        improved = False
        for step in [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
            new_pts = current + step * grad
            new_pts = _project_to_triangle(new_pts)
            new_min = _min_area(new_pts)
            
            if new_min > min_area + 1e-12:
                current = new_pts
                improved = True
                break
        
        if not improved:
            # Try local nudge on involved points
            for idx in involved:
                for _ in range(5):
                    du = np.random.uniform(-0.005, 0.005)
                    dv = np.random.uniform(-0.005, 0.005)
                    
                    bary = _cart_to_bary(current)
                    new_u = bary[idx, 0] + du
                    new_v = bary[idx, 1] + dv
                    
                    new_u = max(0.0, new_u)
                    new_v = max(0.0, new_v)
                    s = new_u + new_v
                    if s > 1.0:
                        new_u /= s
                        new_v /= s
                    
                    new_x = new_v * 1.0 + (1.0 - new_u - new_v) * 0.5
                    new_y = (1.0 - new_u - new_v) * H
                    
                    if new_y < -1e-10 or new_y > SQRT3 * new_x + 1e-10 or new_y > SQRT3 * (1 - new_x) + 1e-10:
                        continue
                    
                    test_pts = current.copy()
                    test_pts[idx, 0] = new_x
                    test_pts[idx, 1] = new_y
                    test_min = _min_area(test_pts)
                    
                    if test_min > min_area + 1e-12:
                        current = test_pts
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best, best_score

def _delone_like_refine(points, time_limit=10.0):
    """
    Refine using a Delaunay-like approach: identify the worst triangles
    and try to improve them by moving points to equalize areas.
    """
    start_t = time.time()
    current = points.copy()
    best = points.copy()
    best_score = _min_area(current)
    
    for it in range(100):
        if time.time() - start_t > time_limit:
            break
        
        areas = _compute_areas(current)
        min_area = float(np.min(areas))
        
        if min_area > best_score:
            best_score = min_area
            best = current.copy()
        
        # Find the smallest triangles
        threshold = min_area + 1e-8
        min_tri_indices = np.where(areas <= threshold)[0]
        
        if len(min_tri_indices) == 0:
            break
        
        # For each smallest triangle, try to increase its area
        # by moving one of its vertices
        improved = False
        for tri_idx in min_tri_indices:
            i, j, k = int(_idx_arr[tri_idx, 0]), int(_idx_arr[tri_idx, 1]), int(_idx_arr[tri_idx, 2])
            
            for move_idx in [i, j, k]:
                # Try moving this point in the direction that increases this triangle's area
                pi = current[move_idx]
                
                # Compute the direction that increases the area of triangle (i,j,k)
                if move_idx == i:
                    other1 = current[j]
                    other2 = current[k]
                    # Move i perpendicular to (j-k) in the direction away from the triangle
                    edge = other2 - other1
                    perp = np.array([-edge[1], edge[0]])
                    # Determine which direction increases area
                    det = np.cross(other1 - pi, other2 - pi)
                    if det > 0:
                        direction = perp
                    else:
                        direction = -perp
                elif move_idx == j:
                    other1 = current[i]
                    other2 = current[k]
                    edge = other2 - other1
                    perp = np.array([-edge[1], edge[0]])
                    det = np.cross(other1 - pi, other2 - pi)
                    if det > 0:
                        direction = -perp
                    else:
                        direction = perp
                else:  # move_idx == k
                    other1 = current[i]
                    other2 = current[j]
                    edge = other2 - other1
                    perp = np.array([-edge[1], edge[0]])
                    det = np.cross(other1 - pi, other2 - pi)
                    if det > 0:
                        direction = perp
                    else:
                        direction = -perp
                
                direction = direction / (np.linalg.norm(direction) + 1e-10)
                
                for step in [0.01, 0.005, 0.002, 0.001]:
                    new_pts = current.copy()
                    new_pts[move_idx] = pi + step * direction
                    new_pts = _project_to_triangle(new_pts)
                    new_min = _min_area(new_pts)
                    
                    if new_min > min_area + 1e-12:
                        current = new_pts
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    np.random.seed(42)
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Multiple diverse initializations
    initializations = []
    
    for seed in range(5):
        init_points = _sobol_points_in_triangle(N, seed=seed)
        init_points = _project_to_triangle(init_points)
        initializations.append(init_points)
    
    grid_points = _initial_guess_grid()
    grid_points = _project_to_triangle(grid_points)
    initializations.append(grid_points)
    
    for seed in range(3):
        np.random.seed(seed + 100)
        bary = np.random.dirichlet([1, 1, 1], size=N)
        u = bary[:, 0]
        v = bary[:, 1]
        init_points = _bary_to_cart(np.column_stack([u, v]))
        init_points = _project_to_triangle(init_points)
        initializations.append(init_points)
    
    # Optimize each initialization with multiple methods
    for i, init_points in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        # Method 1: Nelder-Mead with soft min
        nm_points, nm_score = _nelder_mead_optimize(
            init_points,
            time_limit=min(10.0, deadline - time.time() - 10),
            beta=50.0
        )
        
        # Method 2: Active-set optimization
        as_points, as_score = _active_set_optimize(
            nm_points,
            time_limit=min(10.0, deadline - time.time() - 10)
        )
        
        # Method 3: Delaunay-like refine
        dl_points, dl_score = _delone_like_refine(
            as_points,
            time_limit=min(8.0, deadline - time.time() - 8)
        )
        
        # Method 4: Local nudge search
        refined_points, refined_score = _local_nudge_search(
            dl_points,
            max_iter=200,
            time_limit=min(8.0, deadline - time.time() - 5)
        )
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points
    
    # Phase 2: Refine best result with more aggressive optimization
    if best_points is not None and time.time() < deadline - 20:
        # Nelder-Mead with higher beta
        nm_points, nm_score = _nelder_mead_optimize(
            best_points,
            time_limit=min(10.0, deadline - time.time() - 15),
            beta=100.0
        )
        
        # Active-set
        as_points, as_score = _active_set_optimize(
            nm_points,
            time_limit=min(10.0, deadline - time.time() - 10)
        )
        
        # Delaunay-like
        dl_points, dl_score = _delone_like_refine(
            as_points,
            time_limit=min(8.0, deadline - time.time() - 8)
        )
        
        # Local nudge with smaller steps
        refined_points, refined_score = _local_nudge_search(
            dl_points,
            max_iter=300,
            step_sizes=[0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001],
            time_limit=min(10.0, deadline - time.time() - 5)
        )
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points
    
    # Fallback
    if best_points is None:
        best_points = _sobol_points_in_triangle(N, seed=0)
        best_points = _project_to_triangle(best_points)
    
    # Final projection
    best_points = _project_to_triangle(best_points)
    
    return best_points
