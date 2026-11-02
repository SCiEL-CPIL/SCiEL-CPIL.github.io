"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Use a sequence of Linear Programs (LPs) where, for each point in turn, we
   maximize t subject to linear constraints on the (signed) area of every
   triangle being at least t (after fixing the orientation sign per triplet).
   This is a structural change from the non-smooth min formulation.
2. Alternate between LP-based point updates and a final SLSQP polish.
3. Multi-start with diverse initializations.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize, linprog

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 140.0  # Safety margin before 150s limit

# Pre-compute triplet indices
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]

def _min_triangle_area(pts):
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_j_idx]  # intentional: will fix below
    p_k = pts[_k_idx]
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    return float(np.min(areas))

def _check_inside(pts, eps=1e-9):
    for p in pts:
        x, y = p
        if y < -eps or y > S3H + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
    return True

def _project_to_triangle(pts):
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0: y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y: y = max_y
        if x < 0: x = 0
        if x > 1: x = 1
        pts[i] = [x, y]
    return pts

def _random_point_in_triangle(rng):
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]

def _greedy_init(rng):
    pts = [np.array(_random_point_in_triangle(rng))]
    for _ in range(N - 1):
        best_pos = None
        best_min_dist = -1
        for _ in range(300):
            cand = np.array(_random_point_in_triangle(rng))
            dists = np.linalg.norm(pts - cand, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand
        pts.append(best_pos)
    return np.array(pts)

def _structured_init():
    pts = [
        [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
        [0.5, 0.0], [0.75, S3H/2], [0.25, S3H/2],
        [0.5, S3H/3],
        [0.25, 0.0], [0.75, 0.0],
        [0.125, S3H/4], [0.875, S3H/4]
    ]
    return np.array(pts)

def _build_area_matrix(pts, fixed_pt):
    """
    For each triplet, compute the linear coefficients (a, b, c, d) such that
    the signed area = a*x_fixed + b*y_fixed + c*other_x + d*other_y + e
    where (x_fixed, y_fixed) are the coordinates of the point being optimized.
    
    For a triplet (i, j, k), the signed area is:
    0.5 * ((x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i))
    
    If one of i,j,k is the fixed point m, then the area is linear in x_m, y_m.
    If none is m, then the area is constant (given current pts).
    """
    n_trip = len(_triplets)
    # Coefficients for x_m and y_m
    ax = np.zeros(n_trip)
    ay = np.zeros(n_trip)
    # Constant term
    c0 = np.zeros(n_trip)
    
    for idx, (i, j, k) in enumerate(_triplets):
        # Area = 0.5 * [(xj - xi)(yk - yi) - (yj - yi)(xk - xi)]
        # Expand in terms of the three points' coordinates.
        # If m is i:
        if fixed_pt == i:
            # x_i is variable. Terms involving x_i:
            # 0.5 * [xj*yk - xj*yi - xi*yk + xi*yi - yj*xk + yj*xi + yi*xk - yi*xi]
            # x_i terms: -yk*xi + yi*xi + yj*xi = xi*(yi - yk + yj)
            # So coeff of x_i: 0.5*(yj - yk + yi) -> but yi is variable too
            # Let me do this more carefully.
            # Area = 0.5 * [ (xj - xi)(yk - yi) - (yj - yi)(xk - xi) ]
            # Let xi, yi be variables.
            # = 0.5 * [ xj*yk - xj*yi - xi*yk + xi*yi - yj*xk + yj*xi + yi*xk - yi*xi ]
            # = 0.5 * [ xj*yk - yj*xk - xj*yi + yi*xk + xi*yi - xi*yk + yj*xi ]
            # Group xi terms: xi*(yi - yk + yj)
            # Group yi terms: yi*(xk - xj + xi)
            # Constant: xj*yk - yj*xk
            xi = pts[i]  # current value for constant computation, but we're solving for xi,yi
            xj, yj = pts[j]
            xk, yk = pts[k]
            # ax (coeff of xi): 0.5 * (yj - yk + yi_current?) No, yi is also variable.
            # This is getting complex. Let me use the determinant form directly.
            # Signed area * 2 = det[[1, xi, yi], [1, xj, yj], [1, xk, yk]]
            # = 1*(xj*yk - yj*xk) - xi*(yj - yk) + yi*(xj - xk)
            # So: area2 = const - xi*(yj - yk) + yi*(xj - xk)
            # Where const = xj*yk - yj*xk
            const = xj * yk - yj * xk
            ax[idx] = -0.5 * (yj - yk)
            ay[idx] = 0.5 * (xj - xk)
            c0[idx] = 0.5 * const
        elif fixed_pt == j:
            xk, yk = pts[k]
            xi, yi = pts[i]
            # area2 = 1*(xi*yk - yi*xk) - xj*(yi - yk) + yj*(xi - xk)
            # = const + xj*yk - xj*yi + yj*xi - yj*xk
            # xj terms: yk*xj - yi*xj = xj*(yk - yi)
            # yj terms: xi*yj - xk*yj = yj*(xi - xk)
            # const: xi*yk - yi*xk
            const = xi * yk - yi * xk
            ax[idx] = 0.5 * (yk - yi)
            ay[idx] = 0.5 * (xi - xk)
            c0[idx] = 0.5 * const
        elif fixed_pt == k:
            xi, yi = pts[i]
            xj, yj = pts[j]
            # area2 = 1*(xi*yj - yi*xj) - xk*(yi - yj) + yk*(xi - xj)
            # xk terms: -yi*xk + yj*xk = xk*(yj - yi)
            # yk terms: xi*yk - xj*yk = yk*(xi - xj)
            # const: xi*yj - yi*xj
            const = xi * yj - yi * xj
            ax[idx] = 0.5 * (yj - yi)
            ay[idx] = 0.5 * (xi - xj)
            c0[idx] = 0.5 * const
        else:
            # None of i,j,k is fixed_pt, so area is constant
            xi, yi = pts[i]
            xj, yj = pts[j]
            xk, yk = pts[k]
            area2 = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
            c0[idx] = 0.5 * area2
            ax[idx] = 0
            ay[idx] = 0
    
    return ax, ay, c0

def _lp_optimize_point(pts, m):
    """
    For point m, maximize t such that for all triplets:
    sign_m * (ax*xm + ay*ym + c0) >= t
    where sign_m is determined by the current orientation.
    Also: t >= 0, and xm, ym within triangle bounds.
    """
    ax, ay, c0 = _build_area_matrix(pts, m)
    n_trip = len(_triplets)
    
    # Determine sign for each triplet based on current configuration
    signed_areas = ax * pts[m, 0] + ay * pts[m, 1] + c0
    # Avoid zero
    signed_areas[np.abs(signed_areas) < 1e-15] = 1e-15
    signs = np.sign(signed_areas)
    
    # We want: signs[i] * (ax[i]*xm + ay[i]*ym + c0[i]) >= t  for all i
    # => -signs[i]*ax[i]*xm - signs[i]*ay[i]*ym + t <= -signs[i]*c0[i]
    # Minimize -t (maximize t)
    # Variables: [xm, ym, t]
    c_obj = np.array([0.0, 0.0, -1.0])
    
    # Inequality constraints: A_ub @ x <= b_ub
    A_ub = np.zeros((n_trip + 1, 3))  # +1 for t >= 0 => -t <= 0
    b_ub = np.zeros(n_trip + 1)
    
    for i in range(n_trip):
        A_ub[i, 0] = -signs[i] * ax[i]
        A_ub[i, 1] = -signs[i] * ay[i]
        A_ub[i, 2] = 1.0
        b_ub[i] = -signs[i] * c0[i]
    
    # t >= 0 => -t <= 0
    A_ub[n_trip, 2] = -1.0
    b_ub[n_trip] = 0.0
    
    # Bounds: xm in [0,1], ym in [0, S3H], t in [0, 0.5]
    bounds = [(0, 1), (0, S3H), (0, 0.5)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            new_xm, new_ym, new_t = res.x
            # Check triangle constraints for new point
            if new_ym <= np.sqrt(3) * new_xm + 1e-9 and new_ym <= np.sqrt(3) * (1 - new_xm) + 1e-9:
                new_pt = np.array([new_xm, new_ym])
                return new_pt, new_t
    except Exception:
        pass
    return pts[m].copy(), 0.0

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    initializations = []
    try:
        initializations.append(_greedy_init(rng))
    except:
        pass
    initializations.append(np.array([_random_point_in_triangle(rng) for _ in range(N)]))
    initializations.append(_structured_init())
    for _ in range(3):
        initializations.append(np.array([_random_point_in_triangle(rng) for _ in range(N)]))
    
    for init_pts in initializations:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        current_pts = _project_to_triangle(init_pts.copy())
        current_score = _min_triangle_area(current_pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = current_pts.copy()
        
        # LP-based coordinate descent
        for cycle in range(8):
            if time.time() - start_time > DEADLINE - 25:
                break
            improved = False
            for m in range(N):
                if time.time() - start_time > DEADLINE - 25:
                    break
                new_pt, _ = _lp_optimize_point(current_pts, m)
                if not np.allclose(new_pt, current_pts[m], atol=1e-8):
                    current_pts[m] = new_pt
                    improved = True
            
            new_score = _min_triangle_area(current_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = current_pts.copy()
            
            if not improved:
                break
        
        # SLSQP polish
        if time.time() - start_time > DEADLINE - 10:
            continue
        
        def obj(x):
            pts = x[:2*N].reshape(N, 2)
            return -_min_triangle_area(pts)
        
        def tri_constr(x):
            pts = x[:2*N].reshape(N, 2)
            c1 = pts[:, 1]
            c2 = np.sqrt(3) * pts[:, 0] - pts[:, 1]
            c3 = np.sqrt(3) * (1 - pts[:, 0]) - pts[:, 1]
            c4 = pts[:, 0]
            c5 = 1 - pts[:, 0]
            return np.concatenate([c1, c2, c3, c4, c5])
        
        bounds = [(0, 1)] * (2 * N)
        cons = [{'type': 'ineq', 'fun': tri_constr}]
        
        try:
            res = minimize(obj, current_pts.flatten(), method='SLSQP', bounds=bounds,
                          constraints=cons, options={'maxiter': 50, 'ftol': 1e-10})
            if res.success:
                pts_opt = _project_to_triangle(res.x[:2*N].reshape(N, 2))
                if _check_inside(pts_opt):
                    s = _min_triangle_area(pts_opt)
                    if s > best_score:
                        best_score = s
                        best_pts = pts_opt.copy()
        except:
            pass
    
    if best_pts is None:
        best_pts = _structured_init()
    
    if not _check_inside(best_pts):
        best_pts = _project_to_triangle(best_pts)
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
