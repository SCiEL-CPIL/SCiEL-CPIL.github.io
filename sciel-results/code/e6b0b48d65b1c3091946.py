"""
Heilbronn triangles: maximize the smallest triangle area formed by 11 points in an equilateral triangle.
Strategy:
1. QP formulation via trust-constr: maximize t subject to det^2 >= t^2 for all triples.
2. Soft-min (log-sum-exp) surrogate as auxiliary path.
3. FPS and structured initializations.
4. Final polishing.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint, NonlinearConstraint

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples (i, j, k) with i < j < k
# Number of triples is C(11, 3) = 165
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_IDX_I = _IDX_TRIPLES[:, 0]
_IDX_J = _IDX_TRIPLES[:, 1]
_IDX_K = _IDX_TRIPLES[:, 2]


def _min_triangle_area_vectorized(points):
    """
    Compute the smallest area among all triangles formed by triples of points, vectorized.
    points: (N, 2) array
    Returns: float min area
    """
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    
    # Cross product magnitude for 2D vectors
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    if len(areas) == 0:
        return 0.0
        
    return np.min(areas)

def _all_triangle_areas(points):
    """
    Compute all triangle areas, vectorized.
    """
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _param_to_points(params):
    """
    Map parameters in [0, 1]^{2N} to points inside the triangle.
    Barycentric coordinates:
    lambda_0 = 1 - s_i
    lambda_1 = s_i * (1 - t_i)
    lambda_2 = s_i * t_i
    Point = lambda_1 * V1 + lambda_2 * V2 (since V0 is origin)
    """
    s = params[0::2]
    t = params[1::2]
    
    # Clip to avoid numerical issues at boundaries
    s = np.clip(s, 1e-12, 1.0 - 1e-12)
    t = np.clip(t, 1e-12, 1.0 - 1e-12)
    
    lam1 = s * (1.0 - t)
    lam2 = s * t
    
    x = lam1 * V1[0] + lam2 * V2[0]
    y = lam1 * V1[1] + lam2 * V2[1]
    
    return np.column_stack([x, y])


def _points_to_param(points):
    """
    Map points in the triangle to parameters (s, t).
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Inverse barycentric transformation
    # y = lam1 * V1[1] + lam2 * V2[1] = 0 + lam2 * (sqrt(3)/2) => lam2 = 2*y/sqrt(3)
    lam2 = 2.0 * y / np.sqrt(3)
    # x = lam1 * V1[0] + lam2 * V2[0] = lam1 * 1 + lam2 * 0.5 => lam1 = x - 0.5*lam2
    lam1 = x - 0.5 * lam2
    
    # s = lam1 + lam2, t = lam2 / s
    s = lam1 + lam2
    t = np.zeros_like(s)
    mask = s > 1e-12
    t[mask] = lam2[mask] / s[mask]
    
    s = np.clip(s, 1e-12, 1.0 - 1e-12)
    t = np.clip(t, 1e-12, 1.0 - 1e-12)
    
    params = np.zeros(2 * N)
    params[0::2] = s
    params[1::2] = t
    return params


def _soft_min_objective_and_grad(params, beta):
    """
    Objective: negative soft-min of triangle areas.
    Gradient: computed via chain rule.
    
    soft_min(A) = -1/beta * log(sum(exp(-beta * A_i)))
    We want to maximize min(A), so we minimize -soft_min(A).
    f(params) = -soft_min(A) = 1/beta * log(sum(exp(-beta * A_i)))
    
    grad f = 1/beta * 1/S * sum(-beta * exp(-beta * A_i) * grad A_i)
           = - 1/S * sum(w_i * grad A_i)
    where w_i = exp(-beta * A_i), S = sum(w_i)
    """
    points = _param_to_points(params)
    areas = _all_triangle_areas(points)
    
    # Numerical stability for log-sum-exp
    min_a = np.min(areas)
    # Shift areas to prevent underflow/overflow in exp
    # We compute sum(exp(-beta * (A_i - min_a)))
    shifted = -beta * (areas - min_a)
    # Max shifted is 0 (for min area), so exp is <= 1. Safe.
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    # Objective value
    # -soft_min = min_a - (1/beta) * log(sum(exp(-beta(A_i - min_a))))
    # Wait, the standard soft min is -1/beta log(sum exp(-beta A_i)).
    # Let's stick to the definition:
    # soft_min = min_a - (1/beta) * log( sum( exp(-beta (A_i - min_a)) ) )
    # We minimize -soft_min.
    obj = -(min_a - np.log(sum_exp) / beta)
    
    # Gradient
    # Weights w_i proportional to exp(-beta (A_i - min_a))
    weights = exp_shifted / sum_exp
    
    # Compute grad A_i w.r.t points
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    
    # A_i = 0.5 * |cross_i|
    # cross_i = (pj_x - pi_x)*(pk_y - pi_y) - (pj_y - pi_y)*(pk_x - pi_x)
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    
    # sign of cross to handle absolute value
    # d|cross|/dparams = sign(cross) * dcross/dparams
    sign_cross = np.sign(cross)
    # Avoid division by zero or undefined gradient when cross=0
    sign_cross[np.abs(cross) < 1e-15] = 0.0
    
    # Gradient of cross w.r.t points
    # d(cross)/d(pi) = (-(pk_y - pi_y) - (pj_y - pi_y), -(pj_x - pi_x) + (pk_x - pi_x)) ?
    # Let's re-derive carefully.
    # cross = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    # d/dx_i: -(y_k - y_i) - (y_j - y_i) = y_i - y_k - y_j + y_i = 2y_i - y_j - y_k
    # d/dy_i: (x_j - x_i) - (x_k - x_i) = x_j - x_i - x_k + x_i = x_j - x_k
    # d/dx_j: (y_k - y_i)
    # d/dy_j: -(x_k - x_i)
    # d/dx_k: -(y_j - y_i)
    # d/dy_k: (x_j - x_i)
    
    # Actually, simpler:
    # d(cross)/d(p_i) = p_j - p_k (rotated 90 deg? No.)
    # Let's use the explicit components derived in inspiration, they were correct.
    # d_cross_pi_x = -(pk[:, 1] - pi[:, 1]) + (pj[:, 1] - pi[:, 1])
    # d_cross_pi_y = -(pj[:, 0] - pi[:, 0]) + (pk[:, 0] - pi[:, 0])
    # d_cross_pj_x = pk[:, 1] - pi[:, 1]
    # d_cross_pj_y = -(pk[:, 0] - pi[:, 0])
    # d_cross_pk_x = -(pj[:, 1] - pi[:, 1])
    # d_cross_pk_y = pj[:, 0] - pi[:, 0]
    
    d_cross_pi = np.column_stack([
        -(pk[:, 1] - pi[:, 1]) + (pj[:, 1] - pi[:, 1]),
        -(pj[:, 0] - pi[:, 0]) + (pk[:, 0] - pi[:, 0])
    ])
    d_cross_pj = np.column_stack([
        pk[:, 1] - pi[:, 1],
        -(pk[:, 0] - pi[:, 0])
    ])
    d_cross_pk = np.column_stack([
        -(pj[:, 1] - pi[:, 1]),
        pj[:, 0] - pi[:, 0]
    ])
    
    # Gradient of Area A_i = 0.5 * |cross_i|
    # grad_A_i = 0.5 * sign(cross_i) * d_cross_i
    
    # Accumulate gradient for each point
    grad_points = np.zeros((N, 2))
    
    # Weighted sum of gradients
    w = weights * sign_cross * 0.5
    
    np.add.at(grad_points, _IDX_I, w[:, np.newaxis] * d_cross_pi)
    np.add.at(grad_points, _IDX_J, w[:, np.newaxis] * d_cross_pj)
    np.add.at(grad_points, _IDX_K, w[:, np.newaxis] * d_cross_pk)
    
    # Chain rule: grad_params = J^T * grad_points
    # params[2i] = s_i, params[2i+1] = t_i
    # points[i] = [lam1 * V1_x + lam2 * V2_x, lam1 * V1_y + lam2 * V2_y]
    # lam1 = s(1-t), lam2 = st
    # dx/ds = (1-t)V1_x + tV2_x
    # dx/dt = -sV1_x + sV2_x
    # dy/ds = (1-t)V1_y + tV2_y
    # dy/dt = -sV1_y + sV2_y
    
    s = params[0::2]
    t = params[1::2]
    
    dx_ds = (1 - t) * V1[0] + t * V2[0]
    dx_dt = -s * V1[0] + s * V2[0]
    dy_ds = (1 - t) * V1[1] + t * V2[1]
    dy_dt = -s * V1[1] + s * V2[1]
    
    grad_params = np.zeros(2 * N)
    grad_params[0::2] = grad_points[:, 0] * dx_ds + grad_points[:, 1] * dy_ds
    grad_params[1::2] = grad_points[:, 0] * dx_dt + grad_points[:, 1] * dy_dt
    
    # Objective is -soft_min, so we minimize it.
    # The gradient of the objective function to be minimized is -grad(soft_min).
    # My derivation above:
    # obj = -soft_min
    # grad obj = - grad soft_min
    # grad soft_min = sum(w_i * grad A_i) ... wait.
    # soft_min = -1/beta log(sum exp(-beta A_i))
    # d(soft_min)/dA_i = -1/beta * 1/S * (-beta exp(-beta A_i)) = w_i
    # So grad soft_min = sum(w_i * grad A_i)
    # Therefore grad obj = - sum(w_i * grad A_i)
    # My grad_points accumulated sum(w_i * grad A_i) (with sign handled).
    # So I need to negate grad_params.
    
    return obj, -grad_params

def _raw_objective(params):
    """Objective for final polishing: negative min area."""
    points = _param_to_points(params)
    return -_min_triangle_area_vectorized(points)

def _generate_fps_initialization(rng, n_points=N, n_candidates=300):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling.
    """
    points = np.zeros((n_points, 2))
    # Start with centroid
    points[0] = (V0 + V1 + V2) / 3.0
    
    for k in range(1, n_points):
        r1 = rng.uniform(0, 1, n_candidates)
        r2 = rng.uniform(0, 1, n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        
        candidates = np.column_stack([
            r1 * V1[0] + r2 * V2[0],
            r1 * V1[1] + r2 * V2[1]
        ])
        
        # Vectorized distance calculation
        # candidates: (M, 2), points[:k]: (k, 2)
        diffs = candidates[:, np.newaxis, :] - points[np.newaxis, :k, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        
        best_idx = np.argmax(min_dists)
        points[k] = candidates[best_idx]
        
    return points

def _cross_areas_from_points(points):
    """Compute all 165 triangle areas from points (N,2)."""
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * cross  # signed areas, no abs


def _qp_objective_and_grad(x):
    """
    QP formulation: x = [s_0..s_10, t_0..t_10, tau]
    Minimize -tau (maximize tau)
    s_i, t_i in [0,1], tau >= 0
    """
    n_pts = 2 * N
    s = x[0:N]
    t = x[N:2*N]
    tau = x[2*N]
    
    s_c = np.clip(s, 1e-12, 1.0 - 1e-12)
    t_c = np.clip(t, 1e-12, 1.0 - 1e-12)
    
    lam1 = s_c * (1.0 - t_c)
    lam2 = s_c * t_c
    x_coords = lam1 * V1[0] + lam2 * V2[0]
    y_coords = lam1 * V1[1] + lam2 * V2[1]
    points = np.column_stack([x_coords, y_coords])
    
    areas = _cross_areas_from_points(points)
    
    # Objective: -tau
    obj = -tau
    
    # Gradient w.r.t. x
    grad = np.zeros(2 * N + 1)
    grad[2*N] = -1.0  # d(-tau)/d(tau)
    
    # We need gradient of areas w.r.t. points, then chain rule
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    
    # For the gradient of the constraint g(x) = areas^2 - tau^2 >= 0,
    # we compute it in the constraint function. For the objective, 
    # trust-constr will use the constraint gradients.
    # Actually for trust-constr, we provide objective + gradient, and constraints.
    # The objective is just -tau, so its gradient is simple.
    # The constraints handle the area conditions.
    
    return obj, grad


def _qp_constraints(x):
    """
    Constraints: for each triple, area_i^2 - tau^2 >= 0
    i.e., (signed_area_i)^2 - tau^2 >= 0
    """
    n_pts = 2 * N
    s = x[0:N]
    t = x[N:2*N]
    tau = x[2*N]
    
    s_c = np.clip(s, 1e-12, 1.0 - 1e-12)
    t_c = np.clip(t, 1e-12, 1.0 - 1e-12)
    
    lam1 = s_c * (1.0 - t_c)
    lam2 = s_c * t_c
    x_coords = lam1 * V1[0] + lam2 * V2[0]
    y_coords = lam1 * V1[1] + lam2 * V2[1]
    points = np.column_stack([x_coords, y_coords])
    
    areas = _cross_areas_from_points(points)
    
    # Constraint: area_i^2 - tau^2 >= 0 for all i
    return areas**2 - tau**2


def _qp_jacobian_constraints(x):
    """
    Jacobian of constraint functions w.r.t. x.
    g_i(x) = area_i^2 - tau^2
    dg_i/dx = 2*area_i * d(area_i)/dx
    """
    n_pts = 2 * N
    s = x[0:N]
    t = x[N:2*N]
    tau = x[2*N]
    
    s_c = np.clip(s, 1e-12, 1.0 - 1e-12)
    t_c = np.clip(t, 1e-12, 1.0 - 1e-12)
    
    lam1 = s_c * (1.0 - t_c)
    lam2 = s_c * t_c
    x_coords = lam1 * V1[0] + lam2 * V2[0]
    y_coords = lam1 * V1[1] + lam2 * V2[1]
    points = np.column_stack([x_coords, y_coords])
    
    areas = _cross_areas_from_points(points)
    
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    
    # d(area_i)/d(point) for each triple
    # area_i = 0.5 * cross_i
    # d(cross_i)/d(p_i) = [-(y_k-y_i)+(y_j-y_i), -(x_j-x_i)+(x_k-x_i)]
    # d(cross_i)/d(p_j) = [y_k-y_i, -(x_k-x_i)]
    # d(cross_i)/d(p_k) = [-(y_j-y_i), x_j-x_i]
    
    d_cross_pi = np.column_stack([
        -(pk[:, 1] - pi[:, 1]) + (pj[:, 1] - pi[:, 1]),
        -(pj[:, 0] - pi[:, 0]) + (pk[:, 0] - pi[:, 0])
    ])
    d_cross_pj = np.column_stack([
        pk[:, 1] - pi[:, 1],
        -(pk[:, 0] - pi[:, 0])
    ])
    d_cross_pk = np.column_stack([
        -(pj[:, 1] - pi[:, 1]),
        pj[:, 0] - pi[:, 0]
    ])
    
    # d(area)/d(point) = 0.5 * d(cross)/d(point)
    d_area_pi = 0.5 * d_cross_pi
    d_area_pj = 0.5 * d_cross_pj
    d_area_pk = 0.5 * d_cross_pk
    
    # d(area_i)/d(s_i), d(area_i)/d(t_i) via chain rule
    # point_i = [lam1*V1x + lam2*V2x, lam1*V1y + lam2*V2y]
    # lam1 = s(1-t), lam2 = st
    # d(x_i)/ds = (1-t)*V1x + t*V2x
    # d(x_i)/dt = -s*V1x + s*V2x
    # d(y_i)/ds = (1-t)*V1y + t*V2y
    # d(y_i)/dt = -s*V1y + s*V2y
    
    dx_ds = (1 - t_c) * V1[0] + t_c * V2[0]
    dx_dt = -s_c * V1[0] + s_c * V2[0]
    dy_ds = (1 - t_c) * V1[1] + t_c * V2[1]
    dy_dt = -s_c * V1[1] + s_c * V2[1]
    
    # For each triple, accumulate d(area_i)/d(s_j), d(area_i)/d(t_j) for j in {i,j,k}
    # Build Jacobian: shape (165, 22+1)
    jac = np.zeros((165, 2 * N + 1))
    
    # d(g_i)/d(tau) = -2*tau
    jac[:, 2*N] = -2.0 * tau
    
    # For each triple m, and each point p in the triple:
    # d(g_m)/d(s_p) = 2 * area_m * (d_area_m/dx_p * dx_ds_p + d_area_m/dy_p * dy_ds_p)
    # d(g_m)/d(t_p) = 2 * area_m * (d_area_m/dx_p * dx_dt_p + d_area_m/dy_p * dy_dt_p)
    
    two_areas = 2.0 * areas
    
    # For point i in triple
    jac[:, _IDX_I * 2 + 0] = two_areas * (d_area_pi[:, 0] * dx_ds[_IDX_I] + d_area_pi[:, 1] * dy_ds[_IDX_I])
    jac[:, _IDX_I * 2 + 1] = two_areas * (d_area_pi[:, 0] * dx_dt[_IDX_I] + d_area_pi[:, 1] * dy_dt[_IDX_I])
    
    # For point j in triple
    jac[:, _IDX_J * 2 + 0] = two_areas * (d_area_pj[:, 0] * dx_ds[_IDX_J] + d_area_pj[:, 1] * dy_ds[_IDX_J])
    jac[:, _IDX_J * 2 + 1] = two_areas * (d_area_pj[:, 0] * dx_dt[_IDX_J] + d_area_pj[:, 1] * dy_dt[_IDX_J])
    
    # For point k in triple
    jac[:, _IDX_K * 2 + 0] = two_areas * (d_area_pk[:, 0] * dx_ds[_IDX_K] + d_area_pk[:, 1] * dy_ds[_IDX_K])
    jac[:, _IDX_K * 2 + 1] = two_areas * (d_area_pk[:, 0] * dx_dt[_IDX_K] + d_area_pk[:, 1] * dy_dt[_IDX_K])
    
    return jac


def _qp_hessian_lagrangian(x, g, lam):
    """
    Hessian of Lagrangian for trust-constr.
    L = -tau + sum_m lam_m * (area_m^2 - tau^2)
    Hessian w.r.t. x.
    
    This is complex. For simplicity, we'll use a finite-difference approximation
    or return zero and let trust-constr use its own approximation.
    Actually, let's compute it properly for the key terms.
    
    For now, return zeros to let trust-constr approximate.
    """
    return np.zeros((2 * N + 1, 2 * N + 1))


def _structured_init_qp():
    """Generate a good initial guess for the QP formulation."""
    # Use a symmetric configuration
    # 3 vertices + 3 midpoints + 3 near-centroid + 2 more
    pts = np.zeros((N, 2))
    # Vertices
    pts[0] = V0 + 0.05 * (V1 + V2 - 2*V0)  # slightly inside from V0
    pts[1] = V1 + 0.05 * (V0 + V2 - 2*V1)  # slightly inside from V1
    pts[2] = V2 + 0.05 * (V0 + V1 - 2*V2)  # slightly inside from V2
    # Midpoints of edges
    pts[3] = (V0 + V1) / 2.0
    pts[4] = (V1 + V2) / 2.0
    pts[5] = (V2 + V0) / 2.0
    # Centroid region
    centroid = (V0 + V1 + V2) / 3.0
    pts[6] = centroid
    # Two more points
    pts[7] = centroid + 0.15 * (V0 - centroid)
    pts[8] = centroid + 0.15 * (V1 - centroid)
    # One more
    pts[9] = centroid + 0.15 * (V2 - centroid)
    # Tenth point - adjust
    pts[10] = centroid + 0.08 * (V0 - centroid)
    
    return _points_to_param(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    def time_left():
        return deadline - time.time()

    best_params = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # ===== Phase 1: QP with trust-constr =====
    if time_left() > 60.0:
        # Initial guesses for QP
        qp_inits = []
        
        # Structured initialization
        qp_inits.append(_structured_init_qp())
        
        # FPS initializations
        for i in range(3):
            if time_left() < 50.0: break
            pts = _generate_fps_initialization(rng)
            qp_inits.append(_points_to_param(pts))
        
        # Random initializations
        for i in range(3):
            if time_left() < 40.0: break
            s = rng.uniform(0.2, 0.8, size=N)
            t = rng.uniform(0.2, 0.8, size=N)
            params = np.zeros(2 * N)
            params[0::2] = s
            params[1::2] = t
            qp_inits.append(params)
        
        # For QP, x = [s_0..s_10, t_0..t_10, tau]
        # bounds: s,t in [0,1], tau in [0, 0.05] (max possible area is ~0.433/11^2)
        qp_bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.05)]
        
        for init_params in qp_inits:
            if time_left() < 15.0: break
            
            # Build x vector: [s..., t..., tau]
            s_init = init_params[0::2]
            t_init = init_params[1::2]
            # Initial tau: estimate from current min area
            pts_init = _param_to_points(init_params)
            tau_init = max(0.0, _min_triangle_area_vectorized(pts_init) * 0.9)
            
            x0 = np.concatenate([s_init, t_init, [tau_init]])
            
            try:
                res = minimize(
                    _qp_objective_and_grad,
                    x0,
                    method='trust-constr',
                    jac=True,
                    bounds=qp_bounds,
                    constraints=[
                        NonlinearConstraint(
                            _qp_constraints,
                            0.0,
                            np.inf,
                            jac=_qp_jacobian_constraints
                        )
                    ],
                    options={
                        'maxiter': 100,
                        'gtol': 1e-6,
                        'xtol': 1e-8,
                        'barrier_tol': 1e-4
                    }
                )
                
                if res.x is not None:
                    # Extract params from x
                    s_res = res.x[0:N]
                    t_res = res.x[N:2*N]
                    params_res = np.zeros(2 * N)
                    params_res[0::2] = s_res
                    params_res[1::2] = t_res
                    
                    score = -_raw_objective(params_res)
                    if score > best_score:
                        best_score = score
                        best_params = params_res.copy()
                        
            except Exception:
                pass
    
    # ===== Phase 2: Soft-min annealing (original approach) =====
    if time_left() > 30.0:
        # Generate initial configurations
        initial_params = []
        
        # Use best from QP if available
        if best_params is not None:
            initial_params.append(best_params.copy())
        
        # FPS initializations
        for i in range(3):
            if time_left() < 20.0: break
            pts = _generate_fps_initialization(rng)
            initial_params.append(_points_to_param(pts))
        
        # Random initializations
        for i in range(3):
            if time_left() < 15.0: break
            s = rng.uniform(0.1, 0.9, size=N)
            t = rng.uniform(0.1, 0.9, size=N)
            params = np.zeros(2 * N)
            params[0::2] = s
            params[1::2] = t
            initial_params.append(params)
        
        bounds = [(0.0, 1.0)] * (2 * N)
        betas = [5.0, 15.0, 40.0, 100.0]
        
        for start_p in initial_params:
            if time_left() < 10.0: break
            
            current_params = start_p.copy()
            
            for beta in betas:
                if time_left() < 5.0: break
                
                def obj_grad(params):
                    return _soft_min_objective_and_grad(params, beta)
                
                try:
                    res = minimize(
                        obj_grad, current_params,
                        method='L-BFGS-B',
                        jac=True,
                        bounds=bounds,
                        options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    current_params = res.x
                except Exception:
                    pass
            
            # Final polishing
            if time_left() < 3.0: break
            
            try:
                res = minimize(
                    _raw_objective, current_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 80, 'ftol': 1e-12}
                )
                current_params = res.x
            except Exception:
                pass
            
            score = -_raw_objective(current_params)
            if score > best_score:
                best_score = score
                best_params = current_params.copy()
    
    if best_params is None:
        # Fallback
        return np.array([
            [0.5, 0.0], [0.0, 0.0], [1.0, 0.0],
            [0.25, np.sqrt(3)/4], [0.75, np.sqrt(3)/4], [0.5, np.sqrt(3)/2],
            [0.5, np.sqrt(3)/8], [0.25, np.sqrt(3)/8], [0.75, np.sqrt(3)/8],
            [0.5, 3*np.sqrt(3)/8], [0.5, np.sqrt(3)/4]
        ])
        
    best_points = _param_to_points(best_params)
    
    return best_points
