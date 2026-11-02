"""Heilbronn 11-point problem: maximize minimum triangle area in unit equilateral triangle.
Uses barycentric coordinates and SLP (Sequence of Linear Programs) approach.
"""
import numpy as np
import time
from itertools import combinations
from scipy.optimize import linprog, minimize, differential_evolution

N = 11
S3H = np.sqrt(3) / 2
TRI_AREA = 0.5 * 1.0 * S3H  # area of equilateral triangle with side 1

# Precompute all (i,j,k) triples
_TRIPLES = list(combinations(range(N), 3))
_N_TRIPLES = len(_TRIPLES)  # 165

# Convert triple indices to flat indices for vectorized area computation
_I_IDX = np.array([t[0] for t in _TRIPLES])
_J_IDX = np.array([t[1] for t in _TRIPLES])
_K_IDX = np.array([t[2] for t in _TRIPLES])


def _barycentric_to_cartesian(u, v):
    """Convert barycentric coords to Cartesian. u,v are arrays of length N."""
    x = u + 0.5 * v
    y = S3H * v
    return np.column_stack([x, y])


def _cartesian_to_barycentric(pts):
    """Convert Cartesian to barycentric coords. pts is (N,2) array."""
    v = pts[:, 1] / S3H
    u = pts[:, 0] - 0.5 * v
    return u, v


def _all_areas_from_bary(u, v):
    """Compute all triangle areas given barycentric coords."""
    pts = _barycentric_to_cartesian(u, v)
    p_i = pts[_I_IDX]
    p_j = pts[_J_IDX]
    p_k = pts[_K_IDX]
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    return 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)


def _min_area_from_bary(u, v):
    """Compute minimum triangle area given barycentric coords."""
    return np.min(_all_areas_from_bary(u, v))


def _is_valid_bary(u, v, tol=1e-9):
    """Check if barycentric coordinates are valid (inside triangle)."""
    if np.any(u < -tol) or np.any(v < -tol) or np.any(u + v > 1.0 + tol):
        return False
    return True


def _clamp_bary(u, v):
    """Clamp barycentric coordinates to valid region."""
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    mask = (u + v) > 1.0
    if np.any(mask):
        s = u[mask] + v[mask]
        u[mask] /= s
        v[mask] /= s
    return u, v


def _slp_step(u, v, t_min, max_iter=20, tol=1e-10):
    """
    SLP iteration: linearize the min-area constraint and solve LP.
    Returns improved (u, v, t_min) or original if no improvement.
    """
    for _ in range(max_iter):
        areas = _all_areas_from_bary(u, v)
        
        # Find the active constraints (triangles with smallest area)
        # Use the current areas to determine which are binding
        active_mask = areas <= t_min + 1e-8
        if not np.any(active_mask):
            active_mask = areas <= np.min(areas) + 1e-8
        
        # Get active triples
        active_triples = [t for t, a in zip(_TRIPLES, areas) if a <= t_min + 1e-8 or a <= np.min(areas) + 1e-8]
        if not active_triples:
            active_triples = [_TRIPLES[np.argmin(areas)]]
        
        # Build LP: maximize t subject to area constraints
        # Variables: u_0,...,u_{N-1}, v_0,...,v_{N-1}, t
        # Total: 2N + 1 variables
        
        n_vars = 2 * N + 1
        # Objective: maximize t -> minimize -t
        c = np.zeros(n_vars)
        c[-1] = -1.0  # minimize -t
        
        # Inequality constraints: A_ub @ x <= b_ub
        # Barycentric constraints:
        # u_i >= 0 -> -u_i <= 0
        # v_i >= 0 -> -v_i <= 0
        # u_i + v_i <= 1
        A_ub = []
        b_ub = []
        
        for i in range(N):
            # -u_i <= 0
            row = np.zeros(n_vars)
            row[i] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)
            
            # -v_i <= 0
            row = np.zeros(n_vars)
            row[N + i] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)
            
            # u_i + v_i <= 1
            row = np.zeros(n_vars)
            row[i] = 1.0
            row[N + i] = 1.0
            A_ub.append(row)
            b_ub.append(1.0)
        
        # Area constraints: area(i,j,k) >= t
        # area = 0.5 * |cross product|
        # In barycentric coords, the area of triangle formed by points i,j,k
        # can be expressed as a function of barycentric coords.
        # 
        # For points P_i, P_j, P_k with barycentric coords (u_i,v_i), (u_j,v_j), (u_k,v_k):
        # Cartesian: P_i = (u_i + 0.5*v_i, S3H*v_i)
        # 
        # The signed area is 0.5 * det([P_j - P_i, P_k - P_i])
        # 
        # Let's express this linearly in u,v.
        # P_j - P_i = ((u_j - u_i) + 0.5*(v_j - v_i), S3H*(v_j - v_i))
        # P_k - P_i = ((u_k - u_i) + 0.5*(v_k - v_i), S3H*(v_k - v_i))
        # 
        # det = [(u_j - u_i) + 0.5*(v_j - v_i)] * S3H*(v_k - v_i) 
        #       - S3H*(v_j - v_i) * [(u_k - u_i) + 0.5*(v_k - v_i)]
        # 
        # = S3H * [(u_j - u_i + 0.5*(v_j - v_i))*(v_k - v_i) 
        #          - (v_j - v_i)*(u_k - u_i + 0.5*(v_k - v_i))]
        # 
        # = S3H * [(u_j - u_i)*(v_k - v_i) + 0.5*(v_j - v_i)*(v_k - v_i)
        #          - (v_j - v_i)*(u_k - u_i) - 0.5*(v_j - v_i)*(v_k - v_i)]
        # 
        # = S3H * [(u_j - u_i)*(v_k - v_i) - (v_j - v_i)*(u_k - u_i)]
        # 
        # This is still quadratic in the variables. However, if we linearize around
        # the current solution, we can write:
        # area(P_i, P_j, P_k) ≈ area(P_i^0, P_j^0, P_k^0) + gradient terms
        # 
        # Actually, let's use the fact that for fixed i,j,k, the area is a 
        # bilinear/quadratic function. For SLP, we linearize.
        # 
        # Alternative: use the fact that area = 0.5 * |det|, and at the current 
        # point, the sign is known. So area ≈ 0.5 * sign * det_linearized
        
        # Let's compute the linearization properly.
        # The area (signed) is:
        # A = 0.5 * S3H * [(u_j - u_i)*(v_k - v_i) - (v_j - v_i)*(u_k - u_i)]
        # 
        # Expanding:
        # A = 0.5 * S3H * [u_j*v_k - u_j*v_i - u_i*v_k + u_i*v_i 
        #                   - v_j*u_k + v_j*u_i + v_i*u_k - v_i*u_i]
        # 
        # = 0.5 * S3H * [u_j*v_k - u_j*v_i - u_i*v_k + u_i*v_i 
        #                   - v_j*u_k + v_j*u_i + v_i*u_k - v_i*u_i]
        # 
        # This is bilinear. For linearization at (u^0, v^0):
        # A ≈ A^0 + sum of partial derivatives * (delta)
        
        # Let's just use a simpler approach: for each active triple, add a constraint
        # that the area at the new point is >= t, using the linearization.
        
        # Actually, for a cleaner approach, let's note that the area can be written as:
        # A(i,j,k) = 0.5 * S3H * |M| where M is a matrix involving the barycentric coords.
        # 
        # Let me define the signed area more carefully.
        # Points in Cartesian:
        # P_i = (u_i + v_i/2, S3H * v_i)
        # 
        # Vector P_j - P_i = (u_j - u_i + (v_j - v_i)/2, S3H*(v_j - v_i))
        # Vector P_k - P_i = (u_k - u_i + (v_k - v_i)/2, S3H*(v_k - v_i))
        # 
        # Cross product (2D): (P_j - P_i)_x * (P_k - P_i)_y - (P_j - P_i)_y * (P_k - P_i)_x
        # = [u_j - u_i + (v_j - v_i)/2] * S3H*(v_k - v_i) - S3H*(v_j - v_i) * [u_k - u_i + (v_k - v_i)/2]
        # = S3H * { [u_j - u_i + (v_j - v_i)/2] * (v_k - v_i) - (v_j - v_i) * [u_k - u_i + (v_k - v_i)/2] }
        # = S3H * { (u_j - u_i)(v_k - v_i) + (v_j - v_i)(v_k - v_i)/2 - (v_j - v_i)(u_k - u_i) - (v_j - v_i)(v_k - v_i)/2 }
        # = S3H * { (u_j - u_i)(v_k - v_i) - (v_j - v_i)(u_k - u_i) }
        
        # So signed area = 0.5 * S3H * { (u_j - u_i)(v_k - v_i) - (v_j - v_i)(u_k - u_i) }
        
        # This is bilinear in the variables. For linearization at current point (u^0, v^0):
        # Let f(u,v) = (u_j - u_i)(v_k - v_i) - (v_j - v_i)(u_k - u_i)
        # 
        # Partial derivatives:
        # df/du_i = -(v_k - v_i^0) + (v_j - v_i^0) = v_j - v_k
        # df/du_j = v_k - v_i^0
        # df/du_k = -(v_j - v_i^0)
        # df/dv_i = -(u_j - u_i^0) + (u_k - u_i^0) = u_k - u_j
        # df/dv_j = -(u_k - u_i^0)
        # df/dv_k = u_j - u_i^0
        
        # So the linearized constraint is:
        # 0.5 * S3H * [f^0 + df/du_i*(u_i - u_i^0) + df/du_j*(u_j - u_j^0) + df/du_k*(u_k - u_k^0)
        #               + df/dv_i*(v_i - v_i^0) + df/dv_j*(v_j - v_j^0) + df/dv_k*(v_k - v_k^0)] >= t
        
        # For each active triple (i,j,k):
        for idx, (i, j, k) in enumerate(active_triples):
            # Compute current value
            f0 = (u[j] - u[i]) * (v[k] - v[i]) - (v[j] - v[i]) * (u[k] - u[i])
            area0 = 0.5 * S3H * abs(f0)
            
            # Determine sign
            sign = 1.0 if f0 >= 0 else -1.0
            
            # Partial derivatives
            df_du_i = v[j] - v[k]
            df_du_j = v[k] - v[i]
            df_du_k = -(v[j] - v[i])
            df_dv_i = u[k] - u[j]
            df_dv_j = -(u[k] - u[i])
            df_dv_k = u[j] - u[i]
            
            # Constraint: 0.5 * S3H * sign * [f0 + linear terms] >= t
            # => -0.5 * S3H * sign * [linear terms] - t <= -0.5 * S3H * sign * f0
            
            row = np.zeros(n_vars)
            row[i] = -0.5 * S3H * sign * df_du_i
            row[j] = -0.5 * S3H * sign * df_du_j
            row[k] = -0.5 * S3H * sign * df_du_k
            row[N + i] = -0.5 * S3H * sign * df_dv_i
            row[N + j] = -0.5 * S3H * sign * df_dv_j
            row[N + k] = -0.5 * S3H * sign * df_dv_k
            row[-1] = -1.0  # coefficient of t
            
            A_ub.append(row)
            b_ub.append(-0.5 * S3H * sign * f0)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds
        bounds = [(0, 1)] * N + [(0, 1)] * N + [(t_min * 0.5, None)]
        
        # Solve LP
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if res.status == 0 and res.x is not None:
                u_new = res.x[:N]
                v_new = res.x[N:2*N]
                t_new = res.x[2*N]
                
                # Verify
                u_new, v_new = _clamp_bary(u_new, v_new)
                min_a = _min_area_from_bary(u_new, v_new)
                
                if min_a > t_min + tol:
                    u, v = u_new, v_new
                    t_min = min_a
                elif min_a > t_min:
                    u, v = u_new, v_new
                    t_min = min_a
                else:
                    break
            else:
                break
        except Exception:
            break
    
    return u, v, t_min


def _local_optimize(u, v, max_iter=200):
    """Local optimization using smooth approximation of min area."""
    # Use a smooth min approximation
    def objective(x):
        u_x = x[0::2]
        v_x = x[1::2]
        u_x, v_x = _clamp_bary(u_x, v_x)
        areas = _all_areas_from_bary(u_x, v_x)
        # Smooth min using log-sum-exp
        beta = 50.0
        max_arg = np.max(-beta * areas)
        log_sum_exp = max_arg + np.log(np.sum(np.exp(-beta * areas - max_arg)))
        soft_min = -log_sum_exp / beta
        return -soft_min
    
    x_init = np.column_stack([u, v]).flatten()
    bounds = [(0, 1)] * (2 * N)
    
    try:
        res = minimize(objective, x_init, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        if res.x is not None:
            u_opt = res.x[0::2]
            v_opt = res.x[1::2]
            u_opt, v_opt = _clamp_bary(u_opt, v_opt)
            return u_opt, v_opt
    except Exception:
        pass
    return u, v


def _slp_optimize(u, v, time_start, time_limit):
    """Run SLP iterations with periodic local optimization."""
    t_min = _min_area_from_bary(u, v)
    best_u, best_v, best_t = u.copy(), v.copy(), t_min
    
    for slp_round in range(50):
        if time.time() - time_start > time_limit:
            break
        
        # SLP step
        u, v, t_min = _slp_step(u, v, t_min, max_iter=10)
        
        # Check improvement
        if t_min > best_t + 1e-12:
            best_u, best_v, best_t = u.copy(), v.copy(), t_min
        
        # Periodic local optimization
        if slp_round % 5 == 4:
            u, v = _local_optimize(u, v, max_iter=100)
            t_min = _min_area_from_bary(u, v)
            if t_min > best_t + 1e-12:
                best_u, best_v, best_t = u.copy(), v.copy(), t_min
        
        # Check for convergence
        if t_min <= best_t + 1e-10:
            # Try a few more SLP iterations
            pass
    
    return best_u, best_v, best_t


def construct():
    start_time = time.time()
    TIME_LIMIT = 140  # safety margin before 150s kill
    
    def _time_up():
        return (time.time() - start_time) > TIME_LIMIT
    
    # Initial guess: use a structured initialization
    # Place points in a hexagonal-like pattern
    rng = np.random.RandomState(42)
    
    # Start with a reasonable initial configuration
    # Use farthest-point sampling in barycentric coords
    u_init = np.zeros(N)
    v_init = np.zeros(N)
    
    # First point at center
    u_init[0] = 1/3
    v_init[0] = 1/3
    
    # Add points using farthest-point sampling
    for i in range(1, N):
        best_dist = -1
        best_u, best_v = 0.5, 0.5
        for _ in range(1000):
            u_try = rng.uniform(0, 1)
            v_try = rng.uniform(0, 1 - u_try)
            
            # Compute min distance to existing points
            dists = np.sqrt((u_try - u_init[:i])**2 + (v_try - v_init[:i])**2)
            min_dist = np.min(dists)
            if min_dist > best_dist:
                best_dist = min_dist
                best_u, best_v = u_try, v_try
        
        u_init[i] = best_u
        v_init[i] = best_v
    
    # Run SLP optimization
    u_best, v_best, t_best = _slp_optimize(u_init, v_init, start_time, TIME_LIMIT)
    
    if _time_up():
        return _barycentric_to_cartesian(u_best, v_best)
    
    # Try multiple random restarts
    best_overall_u, best_overall_v, best_overall_t = u_best, v_best, t_best
    
    for restart in range(10):
        if _time_up():
            break
        
        # Random initialization
        u_rand = rng.uniform(0, 1, N)
        v_rand = rng.uniform(0, 1, N)
        u_rand, v_rand = _clamp_bary(u_rand, v_rand)
        
        # Quick SLP
        u_r, v_r, t_r = _slp_optimize(u_rand, v_rand, start_time, TIME_LIMIT)
        
        if t_r > best_overall_t + 1e-12:
            best_overall_u, best_overall_v, best_overall_t = u_r, v_r, t_r
    
    # Final local polish
    if not _time_up():
        best_overall_u, best_overall_v = _local_optimize(best_overall_u, best_overall_v, max_iter=300)
        final_t = _min_area_from_bary(best_overall_u, best_overall_v)
        if final_t > best_overall_t:
            best_overall_t = final_t
    
    # Convert to Cartesian
    pts = _barycentric_to_cartesian(best_overall_u, best_overall_v)
    
    # Verify all points are inside the triangle
    for i in range(N):
        px, py = pts[i]
        v_i = py / S3H
        u_i = px - 0.5 * v_i
        if u_i < -1e-9 or v_i < -1e-9 or u_i + v_i > 1.0 + 1e-9:
            # Clamp
            u_i = max(0, min(1, u_i))
            v_i = max(0, min(1, v_i))
            if u_i + v_i > 1:
                s = u_i + v_i
                u_i /= s
                v_i /= s
            pts[i, 0] = u_i + 0.5 * v_i
            pts[i, 1] = S3H * v_i
    
    return pts
