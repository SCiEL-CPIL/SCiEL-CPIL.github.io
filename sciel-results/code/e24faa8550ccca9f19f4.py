import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0

# Precompute the 165 combinations
_indices = list(itertools.combinations(range(N), 3))
_idx_arr = np.array(_indices)

def _compute_areas(points):
    """Compute all 165 triangle areas from 11 points using vectorized operations."""
    i = _idx_arr[:, 0]
    j = _idx_arr[:, 1]
    k = _idx_arr[:, 2]
    
    xi, yi = points[i, 0], points[i, 1]
    xj, yj = points[j, 0], points[j, 1]
    xk, yk = points[k, 0], points[k, 1]
    
    # Area = 0.5 * |xi(yj-yk) + xj(yk-yi) + xk(yi-yj)|
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
    # Find all triangles with area close to min
    threshold = min_area + 1e-10
    min_indices = np.where(areas <= threshold)[0]
    # Collect unique point indices involved
    involved = set()
    for idx in min_indices:
        involved.add(int(_idx_arr[idx, 0]))
        involved.add(int(_idx_arr[idx, 1]))
        involved.add(int(_idx_arr[idx, 2]))
    return min_area, sorted(involved)

def _bary_to_cart(bary):
    """Convert barycentric coords (u,v) array to Cartesian points, ensuring validity.
    bary has shape (N, 2) with u, v >= 0 and u + v <= 1.
    """
    u = np.clip(bary[:, 0], 0.0, None)
    v = np.clip(bary[:, 1], 0.0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
        
    pts = np.empty((N, 2))
    # Point = u * A + v * B + (1-u-v) * C
    # A = (0,0), B = (1,0), C = (0.5, H)
    pts[:, 0] = u * 0.0 + v * 1.0 + (1.0 - u - v) * 0.5
    pts[:, 1] = u * 0.0 + v * 0.0 + (1.0 - u - v) * H
    return pts

def _project_to_simplex(bary):
    """Project barycentric coordinates onto the simplex u,v >= 0, u+v <= 1."""
    u = np.clip(bary[:, 0], 0.0, None)
    v = np.clip(bary[:, 1], 0.0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack([u, v])

def _local_nudge_search_bary(bary, max_iter=500, step_sizes=None, time_limit=10.0, rng=None):
    """
    Local search in barycentric coords that only moves points involved in the smallest triangles.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    if step_sizes is None:
        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    current_bary = _project_to_simplex(bary.copy())
    current_points = _bary_to_cart(current_bary)
    current_min, _ = _min_area_and_indices(current_points)
    best_bary = current_bary.copy()
    best_min = current_min
    
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
                best_bary = current_bary.copy()
                
            if len(involved) == 0:
                break
                
            # Try nudging each involved point in random directions
            improved = False
            for idx in involved:
                if time.time() - start_t > time_limit:
                    break
                    
                # Try several random perturbations for this point
                for _ in range(30):
                    du = rng.uniform(-step_size, step_size)
                    dv = rng.uniform(-step_size, step_size)
                    
                    new_u = current_bary[idx, 0] + du
                    new_v = current_bary[idx, 1] + dv
                    
                    # Project to simplex
                    new_u = max(0.0, new_u)
                    new_v = max(0.0, new_v)
                    s = new_u + new_v
                    if s > 1.0:
                        new_u /= s
                        new_v /= s
                        
                    test_bary = current_bary.copy()
                    test_bary[idx, 0] = new_u
                    test_bary[idx, 1] = new_v
                    test_points = _bary_to_cart(test_bary)
                    test_min = _min_area(test_points)
                    
                    if test_min > min_area + 1e-12:
                        current_bary = test_bary
                        current_points = test_points
                        min_area = test_min
                        improved = True
                        break
                        
                if improved:
                    break
                    
            if not improved:
                break
                
    return best_bary, best_min

def _greedy_grid_search(grid_res=15, time_limit=5.0):
    """
    Greedy construction: Place points one by one on a grid to maximize min area.
    """
    start_t = time.time()
    # Generate grid points in barycentric coords
    u_vals = np.linspace(0, 1, grid_res)
    v_vals = np.linspace(0, 1, grid_res)
    grid_points = []
    for u in u_vals:
        for v in v_vals:
            if u + v <= 1.0 + 1e-9:
                grid_points.append((u, v))
    
    grid_bary = np.array(grid_points)
    grid_cart = _bary_to_cart(grid_bary)
    
    # Start with no points
    placed_bary = []
    placed_cart = []
    
    for i in range(N):
        best_idx = -1
        best_score = -1.0
        
        # If no points placed, pick center or random?
        # For i=0, any point is equivalent by symmetry. Pick center.
        if i == 0:
            # Center of triangle in barycentric is (1/3, 1/3)
            # Find closest grid point to (1/3, 1/3)
            dists = (grid_bary[:, 0] - 1/3)**2 + (grid_bary[:, 1] - 1/3)**2
            best_idx = np.argmin(dists)
        else:
            # Evaluate each grid point
            for idx in range(len(grid_bary)):
                if time.time() - start_t > time_limit:
                    break
                
                # Tentatively add this point
                new_bary = placed_bary + [grid_bary[idx]]
                new_cart = placed_cart + [grid_cart[idx]]
                
                # Calculate min area
                # We only need to check triangles involving the new point
                # Triangles among existing points are already >= current_min
                # New triangles: (new, p_j, p_k) for j < k < i
                # And (new, p_j, p_k) is not enough, we need all triples including new
                # Actually, we just compute full min area for simplicity if N is small
                # But N=11, grid=225. 11 * 225 * 165 ops is fine.
                
                temp_points = np.array(new_cart)
                min_a = _min_area(temp_points)
                
                if min_a > best_score:
                    best_score = min_a
                    best_idx = idx
        
        if best_idx != -1:
            placed_bary.append(grid_bary[best_idx])
            placed_cart.append(grid_cart[best_idx])
        else:
            # Fallback
            placed_bary.append([0.5, 0.5])
            placed_cart.append([0.5, H/2])
            
    return np.array(placed_bary)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    rng = np.random.default_rng(42)
    
    best_bary = None
    best_score = -np.inf
    
    # Phase 1: Greedy Grid Search
    if time.time() < deadline - 30:
        greedy_bary = _greedy_grid_search(grid_res=20, time_limit=10.0)
        greedy_points = _bary_to_cart(greedy_bary)
        greedy_score = _min_area(greedy_points)
        
        if greedy_score > best_score:
            best_bary = greedy_bary
            best_score = greedy_score
            
        # Polish greedy result
        if time.time() < deadline - 25:
            pol_bary, pol_score = _local_nudge_search_bary(greedy_bary, max_iter=200, time_limit=10.0, rng=rng)
            if pol_score > best_score:
                best_bary = pol_bary
                best_score = pol_score

    # Phase 2: Differential Evolution (Global Search)
    # We run DE for a few iterations, then polish, then restart if time permits.
    bounds = [(0.0, 1.0)] * (2 * N)
    
    def _de_objective(x):
        bary = x.reshape(N, 2)
        bary = _project_to_simplex(bary)
        points = _bary_to_cart(bary)
        # Return negative min area
        return -_min_area(points)
    
    # Run DE
    de_time_limit = 30.0
    if time.time() < deadline - de_time_limit - 10:
        try:
            de_result = differential_evolution(
                _de_objective,
                bounds,
                maxiter=100,
                popsize=20,
                tol=1e-8,
                seed=42,
                polish=False, # We will polish manually
                init='sobol'
            )
            
            de_bary = _project_to_simplex(de_result.x.reshape(N, 2))
            de_points = _bary_to_cart(de_bary)
            de_score = _min_area(de_points)
            
            if de_score > best_score:
                best_bary = de_bary
                best_score = de_score
                
        except Exception:
            pass

    # Phase 3: Local Polish on Best
    if best_bary is not None and time.time() < deadline - 10:
        pol_bary, pol_score = _local_nudge_search_bary(best_bary, max_iter=500, time_limit=15.0, rng=rng)
        if pol_score > best_score:
            best_bary = pol_bary
            best_score = pol_score

    # Phase 4: Second DE Run from Best (Perturbed) if time left
    if best_bary is not None and time.time() < deadline - 20:
        try:
            # Perturb best solution
            perturbed = best_bary.copy()
            perturbed += rng.normal(0, 0.1, perturbed.shape)
            perturbed = _project_to_simplex(perturbed)
            
            # Run DE with initial population centered around perturbed best
            # scipy DE doesn't support custom init easily without 'sobol' or 'latinhypercube'
            # We can just run another DE with a different seed
            de_result2 = differential_evolution(
                _de_objective,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                seed=123,
                polish=False
            )
            
            de_bary2 = _project_to_simplex(de_result2.x.reshape(N, 2))
            de_points2 = _bary_to_cart(de_bary2)
            de_score2 = _min_area(de_points2)
            
            if de_score2 > best_score:
                best_bary = de_bary2
                best_score = de_score2
                
        except Exception:
            pass

    # Final Polish
    if best_bary is not None and time.time() < deadline - 5:
        pol_bary, pol_score = _local_nudge_search_bary(best_bary, max_iter=300, time_limit=5.0, rng=rng)
        if pol_score > best_score:
            best_bary = pol_bary
            best_score = pol_score

    # Fallback
    if best_bary is None:
        # Random valid barycentric
        bary = np.random.dirichlet([1, 1, 1], size=N)
        best_bary = bary[:, :2] # u, v (w is 1-u-v)
        # Wait, dirichlet gives 3 coords summing to 1.
        # u = bary[:,0], v = bary[:,1].
        # Check validity
        best_bary = _project_to_simplex(best_bary)

    final_bary = _project_to_simplex(best_bary)
    final_points = _bary_to_cart(final_bary)
    
    return final_points
