import numpy as np
import time
from scipy.optimize import linprog, minimize
import itertools

N = 11

# EVOLVE-BLOCK-START
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

# Safety margin from 150s limit.
DEADLINE = 140.0

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
NUM_TRIP = len(_TRIPLETS)  # 165

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])


def _bary_to_cart(uv_flat):
    """
    Convert flat array of barycentric coords (u, v) for N points to Cartesian (N, 2).
    w = 1 - u - v.
    P = u*V0 + v*V1 + w*V2.
    Since V0=(0,0), P = v*(1,0) + w*(0.5, S3H) = (v + 0.5w, w*S3H).
    """
    uv = uv_flat.reshape(-1, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * S3H
    return np.column_stack((x, y))


def _cart_to_bary(pts):
    """
    Convert Cartesian points to barycentric coords (u, v) where w = 1-u-v.
    Points must be inside the triangle.
    """
    uv = np.zeros((len(pts), 2))
    for i in range(len(pts)):
        x, y = pts[i]
        # w = y / S3H
        w = y / S3H
        # x = v + 0.5 * w  => v = x - 0.5 * w
        v = x - 0.5 * w
        # u = 1 - v - w
        u = 1.0 - v - w
        uv[i] = [u, v]
    return uv


def _vectorized_min_area_cart(pts):
    """Vectorized computation of min triangle area from Cartesian points."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _soft_min_area_cart(pts, beta):
    """
    Soft minimum of triangle areas using log-sum-exp for numerical stability.
    """
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def _project_cartesian(pts):
    """Project points into the triangle if they are slightly outside."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _is_valid_cart(pts, tol=1e-6):
    """Check if all points are inside the triangle."""
    for i in range(len(pts)):
        x, y = pts[i]
        if y < -tol or y > S3H + tol:
            return False
        x_min = y / S3
        x_max = 1 - y / S3
        if x < x_min - tol or x > x_max + tol:
            return False
    return True


def _solve_lp(uv_fixed, t_lower=0.0):
    """
    Solve the LP:
    Maximize t
    Subject to:
      For each triplet (i,j,k):
        s_{ijk} * (area_{ijk}) >= t
      where area_{ijk} is the signed area (twice the actual area)
      and s_{ijk} is a sign variable (+1 or -1) that we fix based on current configuration.
    
    Actually, the area constraint is |cross| >= 2t.
    This is equivalent to:
      cross >= 2t  OR  cross <= -2t
    This is non-convex.
    
    However, if we fix the "orientation" of each triangle (i.e., whether cross is positive or negative),
    we get a linear constraint:
      s_{ijk} * cross >= 2t
    where s_{ijk} is +1 if we expect cross > 0, and -1 if cross < 0.
    
    We can determine s_{ijk} from the current configuration.
    Then we solve the LP.
    The solution will satisfy the constraints for the fixed orientations.
    If the optimal t is such that some triangles flip orientation, we detect this and re-solve with updated orientations.
    """
    # Variables:
    # x[0..2N-1]: barycentric coordinates (u_i, v_i) for i=0..N-1
    # x[2N]: t
    
    n_vars = 2 * N + 1
    t_idx = 2 * N
    
    # Objective: maximize t => minimize -t
    c = np.zeros(n_vars)
    c[t_idx] = -1.0
    
    # Inequality constraints: A_ub @ x <= b_ub
    # For each triplet, we have: -s_{ijk} * cross_{ijk} <= -2t
    # cross_{ijk} = (x_j - x_i) * (y_k - y_i) - (y_j - y_i) * (x_k - x_i)
    # In barycentric coords:
    # x_i = v_i + 0.5*w_i, y_i = S3H * w_i
    # w_i = 1 - u_i - v_i
    # x_i = v_i + 0.5*(1 - u_i - v_i) = 0.5 - 0.5*u_i + 0.5*v_i
    # y_i = S3H * (1 - u_i - v_i)
    
    # Let's derive the linear form of cross in terms of u, v.
    # This is complex. Let's use the Cartesian representation for the LP variables?
    # No, let's stick to barycentric but derive the coefficients.
    
    # Alternatively, we can solve the LP in Cartesian coordinates with linear constraints for the triangle.
    # Variables: x_i, y_i for i=0..N-1, and t.
    # Constraints:
    # 1. Points inside triangle:
    #    y_i >= 0
    #    y_i <= S3H
    #    x_i >= y_i / S3
    #    x_i <= 1 - y_i / S3
    # 2. Area constraints:
    #    s_{ijk} * cross_{ijk} >= 2t  =>  -s_{ijk} * cross_{ijk} <= -2t
    
    # Let's use Cartesian variables.
    n_vars_cart = 2 * N + 1
    t_idx_cart = 2 * N
    
    c_cart = np.zeros(n_vars_cart)
    c_cart[t_idx_cart] = -1.0
    
    A_ub_list = []
    b_ub_list = []
    
    # Triangle containment constraints
    for i in range(N):
        xi = 2 * i
        yi = 2 * i + 1
        
        # y_i >= 0  =>  -y_i <= 0
        row = np.zeros(n_vars_cart)
        row[yi] = -1.0
        A_ub_list.append(row)
        b_ub_list.append(0.0)
        
        # y_i <= S3H  =>  y_i <= S3H
        row = np.zeros(n_vars_cart)
        row[yi] = 1.0
        A_ub_list.append(row)
        b_ub_list.append(S3H)
        
        # x_i >= y_i / S3  =>  x_i - y_i/S3 >= 0  =>  -x_i + y_i/S3 <= 0
        row = np.zeros(n_vars_cart)
        row[xi] = -1.0
        row[yi] = 1.0 / S3
        A_ub_list.append(row)
        b_ub_list.append(0.0)
        
        # x_i <= 1 - y_i / S3  =>  x_i + y_i/S3 <= 1
        row = np.zeros(n_vars_cart)
        row[xi] = 1.0
        row[yi] = 1.0 / S3
        A_ub_list.append(row)
        b_ub_list.append(1.0)
    
    # Area constraints
    # We need to determine the sign s_{ijk} from the current configuration.
    pts_cart = _bary_to_cart(uv_fixed)
    
    for m in range(NUM_TRIP):
        i, j, k = _TRIPLETS[m]
        pi = pts_cart[i]
        pj = pts_cart[j]
        pk = pts_cart[k]
        
        dx1 = pj[0] - pi[0]
        dy1 = pj[1] - pi[1]
        dx2 = pk[0] - pi[0]
        dy2 = pk[1] - pi[1]
        cross_val = dx1 * dy2 - dy1 * dx2
        
        # Determine sign
        if cross_val >= 0:
            s = 1.0
        else:
            s = -1.0
        
        # Constraint: s * cross >= 2t  =>  -s * cross <= -2t
        # cross = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
        # This is quadratic in Cartesian variables!
        # We cannot use Cartesian variables for LP.
        
        # We MUST use barycentric variables where the area is linear.
        # Let's go back to barycentric.
        pass
    
    # Re-derive for barycentric coordinates.
    # x_i = 0.5 - 0.5*u_i + 0.5*v_i
    # y_i = S3H * (1 - u_i - v_i)
    
    # cross_{ijk} = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    
    # Let's compute the coefficients for u and v.
    # This is tedious but doable.
    # Alternatively, we can note that the area is linear in each coordinate if the other two are fixed?
    # No, it's bilinear.
    
    # Wait, the area of a triangle in barycentric coordinates is proportional to the determinant of the matrix of barycentric coordinates.
    # Area(P, Q, R) = TRI_AREA * |det([[u_P, v_P, w_P], [u_Q, v_Q, w_Q], [u_R, v_R, w_R]])|
    
    # Since w = 1 - u - v, the determinant is linear in each row's u, v, w.
    # So the area is a linear function of the barycentric coordinates of each point, given the other two.
    # Actually, it is multilinear.
    
    # Let's use the determinant form.
    # D = det(M) where M is 3x3 matrix of barycentric coords.
    # D is linear in each row.
    
    # Let's build the LP in barycentric coordinates.
    # Variables: u_i, v_i for i=0..N-1, and t.
    # Constraints:
    # 1. u_i >= 0, v_i >= 0, u_i + v_i <= 1
    # 2. s_{ijk} * D_{ijk} >= 2 * t * TRI_AREA  (since Area = TRI_AREA * |D|)
    
    # Let's compute the coefficients of D_{ijk} in terms of u, v.
    # D = det([[u_i, v_i, 1-u_i-v_i], [u_j, v_j, 1-u_j-v_j], [u_k, v_k, 1-u_k-v_k]])
    
    # This is a linear function of u_i, v_i, u_j, v_j, u_k, v_k.
    # We can precompute the coefficients for each triplet.
    
    # Let's define a function to get the coefficients of D_{ijk} for a given triplet.
    # D = sum_{p,q,r} C_{pqr} * coord_p * coord_q * coord_r ... no, it's a determinant.
    
    # Actually, we can just evaluate the gradient of D with respect to each variable.
    # Or, since D is linear in each row, we can write:
    # D = a_i * u_i + b_i * v_i + c_i * (1-u_i-v_i) + ...
    
    # Let's use the property that D is the sum of 6 terms.
    # D = u_i(v_j(1-u_k-v_k) - (1-u_j-v_j)v_k) - v_i(u_j(1-u_k-v_k) - (1-u_j-v_j)u_k) + (1-u_i-v_i)(u_j v_k - v_j u_k)
    
    # This is getting complicated. Let's just use the Cartesian LP but realize that the area constraint is quadratic.
    # We cannot use LP for Cartesian.
    
    # So we must use barycentric.
    # Let's precompute the coefficients for D_{ijk} for each triplet.
    
    # Coefficients for u_i, v_i, u_j, v_j, u_k, v_k.
    # D = det(M)
    # Let M = [[u_i, v_i, w_i], [u_j, v_j, w_j], [u_k, v_k, w_k]]
    # where w = 1 - u - v.
    
    # D = u_i(v_j w_k - w_j v_k) - v_i(u_j w_k - w_j u_k) + w_i(u_j v_k - v_j u_k)
    
    # Substitute w = 1 - u - v.
    # This will yield a linear expression in u_i, v_i, u_j, v_j, u_k, v_k.
    
    # Let's compute the coefficients programmatically for each triplet.
    
    A_ub_bary = []
    b_ub_bary = []
    
    n_vars_bary = 2 * N + 1
    t_idx_bary = 2 * N
    
    c_bary = np.zeros(n_vars_bary)
    c_bary[t_idx_bary] = -1.0
    
    # Containment constraints in barycentric:
    # u_i >= 0  =>  -u_i <= 0
    # v_i >= 0  =>  -v_i <= 0
    # u_i + v_i <= 1
    for i in range(N):
        ui = 2 * i
        vi = 2 * i + 1
        
        row = np.zeros(n_vars_bary)
        row[ui] = -1.0
        A_ub_bary.append(row)
        b_ub_bary.append(0.0)
        
        row = np.zeros(n_vars_bary)
        row[vi] = -1.0
        A_ub_bary.append(row)
        b_ub_bary.append(0.0)
        
        row = np.zeros(n_vars_bary)
        row[ui] = 1.0
        row[vi] = 1.0
        A_ub_bary.append(row)
        b_ub_bary.append(1.0)
    
    # Area constraints
    for m in range(NUM_TRIP):
        i, j, k = _TRIPLETS[m]
        
        # Compute coefficients of D_{ijk}
        # D = u_i(v_j w_k - w_j v_k) - v_i(u_j w_k - w_j u_k) + w_i(u_j v_k - v_j u_k)
        
        # Let's expand this in terms of u and v.
        # w_j = 1 - u_j - v_j
        # w_k = 1 - u_k - v_k
        
        # Term 1: u_i * (v_j(1-u_k-v_k) - (1-u_j-v_j)v_k)
        # = u_i * (v_j - v_j u_k - v_j v_k - v_k + v_j u_k + v_j v_k)
        # = u_i * (v_j - v_k)
        
        # Term 2: -v_i * (u_j(1-u_k-v_k) - (1-u_j-v_j)u_k)
        # = -v_i * (u_j - u_j u_k - u_j v_k - u_k + u_j u_k + u_j v_k)
        # = -v_i * (u_j - u_k)
        
        # Term 3: w_i * (u_j v_k - v_j u_k)
        # = (1-u_i-v_i) * (u_j v_k - v_j u_k)
        
        # So D = u_i(v_j - v_k) - v_i(u_j - u_k) + (1-u_i-v_i)(u_j v_k - v_j u_k)
        
        # Let A = u_j v_k - v_j u_k
        # D = u_i(v_j - v_k) - v_i(u_j - u_k) + A - u_i A - v_i A
        # D = A + u_i(v_j - v_k - A) - v_i(u_j - u_k + A)
        
        # Coefficients:
        # const: A
        # u_i: (v_j - v_k - A)
        # v_i: -(u_j - u_k + A)
        # u_j: (v_k - A)  [from A = u_j v_k - v_j u_k]
        # v_j: -u_k       [from A]
        # u_k: -v_j       [from A]
        # v_k: u_j        [from A]
        
        # Wait, A depends on j and k.
        # A = u_j v_k - v_j u_k
        
        # Let's re-evaluate the coefficients carefully.
        # D = A + u_i(v_j - v_k - A) - v_i(u_j - u_k + A)
        
        # Coeff of u_i: (v_j - v_k - A)
        # Coeff of v_i: -(u_j - u_k + A)
        # Coeff of u_j: v_k (from A)
        # Coeff of v_j: -u_k (from A)
        # Coeff of u_k: -v_j (from A)
        # Coeff of v_k: u_j (from A)
        
        # But A is not a constant, it's a variable.
        # A = u_j v_k - v_j u_k
        # So D = (u_j v_k - v_j u_k) + u_i(v_j - v_k - u_j v_k + v_j u_k) - v_i(u_j - u_k + u_j v_k - v_j u_k)
        
        # This is still bilinear.
        # I made a mistake. D is NOT linear in all variables simultaneously.
        # It is linear in the coordinates of each point if the other two are fixed.
        # But for LP, we need it to be linear in ALL variables.
        
        # This means we CANNOT use a single LP to solve this problem directly.
        # We must use the SLP approach:
        # 1. Fix the signs s_{ijk}.
        # 2. Linearize the area constraints around the current point.
        # 3. Solve the LP.
        # 4. Update the signs and repeat.
        
        # The linearization of |cross| at a point where cross > 0 is cross.
        # So if we fix the sign, the constraint s * cross >= 2t becomes linear in the variables IF cross is linear.
        # But cross is NOT linear in all variables.
        
        # This implies that the SLP approach with LP is not straightforward.
        
        # Let's abandon the LP approach and use the soft-min annealing approach from the inspiration, which is robust and effective.
        pass
    
    # Since LP is not viable, we return None to indicate failure.
    return None


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0
    
    def _elapsed_ok(margin=0):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    def _update_best(pts):
        nonlocal best_pts, best_area
        if not _is_valid_cart(pts):
            return
        area = _vectorized_min_area_cart(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()
    
    # Initial configuration: High symmetry guess
    init_cart = np.array([
        [0.0, 0.0],          # Corner 0
        [1.0, 0.0],          # Corner 1
        [0.5, S3H],          # Corner 2
        [0.5, 0.0],          # Mid bottom
        [0.75, S3H / 2],     # Mid right
        [0.25, S3H / 2],     # Mid left
        [1/3, S3H/3],        # Centroid
        [0.333, 0.0],        # Bottom edge
        [0.667, 0.0],        # Bottom edge
        [0.5, S3H/3],        # Median
        [0.2, 0.1],          # Interior
    ], dtype=float)
    
    init_cart = _project_cartesian(init_cart)
    _update_best(init_cart)
    
    init_uv = _cart_to_bary(init_cart)
    
    def _objective_bary(uv_flat, beta):
        pts = _bary_to_cart(uv_flat)
        uv = uv_flat.reshape(-1, 2)
        u = uv[:, 0]
        v = uv[:, 1]
        w = 1.0 - u - v
        
        viol = np.maximum(0, -w)
        penalty = 1000.0 * np.sum(viol ** 2)
        
        ma = _soft_min_area_cart(pts, beta)
        return -ma + penalty

    def _project_to_simplex(uv_flat):
        uv = uv_flat.reshape(-1, 2).copy()
        for i in range(len(uv)):
            u, v = uv[i]
            u = max(0.0, min(1.0, u))
            v = max(0.0, min(1.0, v))
            if u + v > 1.0:
                scale = 1.0 / (u + v)
                u *= scale
                v *= scale
            uv[i] = [u, v]
        return uv.flatten()

    # Multi-start with annealing
    beta_schedule = [10.0, 30.0, 80.0, 200.0, 500.0]
    
    # Generate starts
    starts = [init_uv]
    for _ in range(5):
        uv_rand = rng.random((N, 2))
        # Ensure u+v <= 1
        for i in range(N):
            s = uv_rand[i, 0] + uv_rand[i, 1]
            if s > 1.0:
                uv_rand[i] /= s
        starts.append(uv_rand.flatten())
    
    for start_uv in starts:
        if not _elapsed_ok(20):
            break
        
        uv_curr = start_uv.copy()
        for beta in beta_schedule:
            if not _elapsed_ok(10):
                break
            try:
                bounds = [(0, 1), (0, 1)] * N
                result = minimize(
                    lambda x: _objective_bary(x, beta),
                    uv_curr,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                uv_curr = result.x
                uv_curr = _project_to_simplex(uv_curr)
                pts = _bary_to_cart(uv_curr)
                _update_best(pts)
            except Exception:
                break
    
    # Final polish with high beta
    if best_pts is not None and _elapsed_ok(10):
        uv_best = _cart_to_bary(best_pts)
        for beta in [1000.0, 5000.0]:
            if not _elapsed_ok(5):
                break
            try:
                bounds = [(0, 1), (0, 1)] * N
                result = minimize(
                    lambda x: _objective_bary(x, beta),
                    uv_best,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                uv_best = result.x
                uv_best = _project_to_simplex(uv_best)
                pts = _bary_to_cart(uv_best)
                _update_best(pts)
            except Exception:
                break
    
    if best_pts is None:
        # Fallback
        best_pts = init_cart
    
    return best_pts
# EVOLVE-BLOCK-END
