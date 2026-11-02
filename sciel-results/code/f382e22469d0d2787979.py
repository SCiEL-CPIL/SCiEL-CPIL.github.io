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

def _soft_min_area(points, beta=50.0):
    """
    Smooth approximation of the minimum area using log-sum-exp.
    -log(sum(exp(-beta * areas))) / beta approximates min(areas) for large beta.
    """
    areas = _compute_areas(points)
    # Use log-sum-exp trick for numerical stability
    min_a = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - min_a))))
    return min_a - log_sum / beta

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

def _cart_to_bary(pts):
    """Convert Cartesian points to barycentric coords (u, v) where u,v >= 0, u+v <= 1.
    u = weight of A=(0,0), v = weight of B=(1,0), 1-u-v = weight of C=(0.5,H)
    """
    bary = np.empty((N, 2))
    for m in range(N):
        x = pts[m, 0]
        y = pts[m, 1]
        # 1-u-v = y/H
        # v = x - (1-u-v)*0.5 = x - 0.5*y/H
        # u = 1 - y/H - v
        w_c = y / H  # weight of C
        w_b = x - 0.5 * w_c  # weight of B
        w_a = 1.0 - w_c - w_b  # weight of A
        bary[m, 0] = w_a
        bary[m, 1] = w_b
    return bary

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

def _objective_bary(bary_flat, beta=50.0):
    """
    Objective function for optimization: negative of soft min area.
    bary_flat is a flat array of 22 barycentric coordinates (u0,v0,u1,v1,...,u10,v10).
    """
    bary = bary_flat.reshape(N, 2)
    bary = _project_to_simplex(bary)
    points = _bary_to_cart(bary)
    return -_soft_min_area(points, beta)

def _nelder_mead_optimize_bary(bary, time_limit=10.0, beta=50.0):
    """
    Use Nelder-Mead to optimize the soft min area objective in barycentric coords.
    """
    start_t = time.time()
    x0 = bary.flatten()
    
    result = minimize(
        _objective_bary,
        x0,
        method='Nelder-Mead',
        args=(beta,),
        options={
            'maxiter': 3000,
            'xatol': 1e-9,
            'fatol': 1e-11,
            'adaptive': True
        }
    )
    
    elapsed = time.time() - start_t
    if elapsed > time_limit:
        return bary, _min_area(_bary_to_cart(bary))
    
    new_bary = _project_to_simplex(result.x.reshape(N, 2))
    new_points = _bary_to_cart(new_bary)
    score = _min_area(new_points)
    
    old_score = _min_area(_bary_to_cart(bary))
    if score > old_score:
        return new_bary, score
    else:
        return bary, old_score

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

def _initial_guess_grid():
    """Generate a structured initial guess based on a grid pattern in barycentric coords.
    Returns (N, 2) array with u, v >= 0, u+v <= 1.
    """
    # Use a hexagonal-like pattern in the simplex
    positions = [
        (0.0, 0.0), (1.0, 0.0), (0.0, 1.0),  # Corners
        (0.33, 0.33), (0.67, 0.33), (0.33, 0.67),  # Midpoints of edges
        (0.5, 0.25), (0.25, 0.5), (0.75, 0.25),  # Inner points
        (0.4, 0.4), (0.6, 0.4),  # More inner
    ]
    bary = np.empty((N, 2))
    for m, (u, v) in enumerate(positions):
        if u + v > 1.0:
            s = 1.0 / (u + v)
            u *= s
            v *= s
        bary[m, 0] = u
        bary[m, 1] = v
    return bary

def _random_valid_barycentric(rng):
    """Generate random valid barycentric coordinates."""
    bary = np.empty((N, 2))
    for m in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1.0:
                bary[m, 0] = u
                bary[m, 1] = v
                break
    return bary

def _sobol_barycentric(n, seed=42):
    """Generate n quasi-random points in the simplex using Sobol sequence."""
    from scipy.stats import qmc
    rng = qmc.Sobol(d=2, seed=seed)
    n_gen = n * 6
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
    return np.column_stack([u, v])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_bary = None
    best_score = -np.inf
    
    def _try_nm_optimize_bary(bary, beta, time_limit):
        """Try to optimize from initial barycentric points using Nelder-Mead, return (bary, score) or (None, -inf)."""
        try:
            bary_out, score = _nelder_mead_optimize_bary(bary, time_limit=time_limit, beta=beta)
            # Verify validity
            bary_out = _project_to_simplex(bary_out)
            pts = _bary_to_cart(bary_out)
            if np.all(pts[:, 0] >= -1e-8) and np.all(pts[:, 0] <= 1.0 + 1e-8) and \
               np.all(pts[:, 1] >= -1e-8) and np.all(pts[:, 1] <= H + 1e-8):
                x = pts[:, 0]
                y = pts[:, 1]
                if np.all(y >= -1e-8) and np.all(y <= SQRT3 * x + 1e-8) and \
                   np.all(y <= -SQRT3 * x + SQRT3 + 1e-8):
                    score = _min_area(pts)
                    return bary_out, score
        except Exception:
            pass
        return None, -np.inf
    
    # Phase 1: Annealing schedule with Nelder-Mead
    # Start with low beta for global exploration, increase for local refinement
    
    # Start with a good structured guess
    initial_bary = _initial_guess_grid()
    
    # Annealing: beta values to try
    beta_schedule = [10.0, 25.0, 50.0, 100.0, 200.0, 500.0]
    
    current_bary = initial_bary.copy()
    
    for beta in beta_schedule:
        if time.time() > deadline - 30:
            break
        
        # Determine time budget based on remaining time
        remaining = deadline - time.time()
        time_limit = min(15.0, remaining * 0.3)
        
        if time_limit < 5.0:
            break
        
        bary_out, score = _try_nm_optimize_bary(current_bary, beta=beta, time_limit=time_limit)
        if bary_out is not None and score > best_score:
            best_bary = bary_out
            best_score = score
            current_bary = bary_out
    
    # Apply local nudge search to the best result
    if best_bary is not None and time.time() < deadline - 25:
        nudged_bary, nudged_score = _local_nudge_search_bary(best_bary, max_iter=400, time_limit=15.0, rng=rng)
        if nudged_score > best_score:
            best_bary = nudged_bary
            best_score = nudged_score

    # Phase 2: Multiple restarts from diverse initial points
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # Alternate between random and Sobol initial points
        if i % 2 == 0:
            random_bary = _random_valid_barycentric(rng)
        else:
            random_bary = _sobol_barycentric(N, seed=42 + i * 7)
        
        # Use moderate beta for these restarts
        beta = 30.0 + 10.0 * (i % 5)
        
        bary_out, score = _try_nm_optimize_bary(random_bary, beta=beta, time_limit=8.0)
        if bary_out is not None and score > best_score:
            best_bary = bary_out
            best_score = score
            
        # Apply local nudge search to each random restart result
        if bary_out is not None and time.time() < deadline - 15:
            nudged_bary, nudged_score = _local_nudge_search_bary(bary_out, max_iter=200, time_limit=5.0, rng=rng)
            if nudged_score > best_score:
                best_bary = nudged_bary
                best_score = nudged_score
    
    # Phase 3: High-precision refinement
    if best_bary is not None:
        if time.time() > deadline - 15:
            pass  # Fall through to final return
        
        # Refine with very high beta
        if time.time() < deadline - 12:
            bary_out, score = _try_nm_optimize_bary(best_bary, beta=500.0, time_limit=10.0)
            if bary_out is not None and score > best_score:
                best_bary = bary_out
                best_score = score
        
        # Apply nudge search to refined result
        if time.time() < deadline - 8:
            nudged_bary, nudged_score = _local_nudge_search_bary(best_bary, max_iter=500, time_limit=8.0, rng=rng)
            if nudged_score > best_score:
                best_bary = nudged_bary
                best_score = nudged_score

        # Even higher beta for final polish
        if time.time() < deadline - 5:
            bary_out, score = _try_nm_optimize_bary(best_bary, beta=1000.0, time_limit=5.0)
            if bary_out is not None and score > best_score:
                best_bary = bary_out
                best_score = score
    
    # Phase 4: If we still have time, try differential evolution for global search
    if best_bary is None or time.time() < deadline - 20:
        if time.time() > deadline - 15:
            if best_bary is not None:
                return _bary_to_cart(_project_to_simplex(best_bary))
            # Fallback: return a valid configuration
            bary = _random_valid_barycentric(rng)
            return _bary_to_cart(bary)
        
        # Try differential evolution with a moderate budget
        if time.time() < deadline - 10:
            try:
                bounds = [(0.0, 1.0)] * (2 * N)
                
                def _de_objective(x):
                    bary = x.reshape(N, 2)
                    bary = _project_to_simplex(bary)
                    points = _bary_to_cart(bary)
                    return -_soft_min_area(points, beta=50.0)
                
                de_result = differential_evolution(
                    _de_objective,
                    bounds,
                    maxiter=50,
                    popsize=15,
                    tol=1e-8,
                    seed=42,
                    polish=True
                )
                
                de_bary = _project_to_simplex(de_result.x.reshape(N, 2))
                de_points = _bary_to_cart(de_bary)
                de_score = _min_area(de_points)
                
                if de_score > best_score:
                    best_bary = de_bary
                    best_score = de_score
                    
            except Exception:
                pass
    
    # Final check and strict validation
    if best_bary is None:
        # Fallback: return a valid configuration
        bary = _random_valid_barycentric(rng)
        best_bary = bary
    
    # Strictly enforce containment by projecting the final output
    final_bary = _project_to_simplex(best_bary)
    final_points = _bary_to_cart(final_bary)
    
    return final_points
