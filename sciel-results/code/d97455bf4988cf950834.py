import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14
DEADLINE = 142.0  # wall-clock guard (evaluator kills at 150s)

def _triangle_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx, dtype=np.int32)

_TRI = _triangle_indices(N)
_I = _TRI[:, 0]
_J = _TRI[:, 1]
_K = _TRI[:, 2]
_NTRI = _TRI.shape[0]

def _min_triangle_area(points):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    try:
        return float(scipy.spatial.ConvexHull(points).volume)
    except Exception:
        return 0.0

def _score(points):
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        return _min_triangle_area(points) / ha
    except Exception:
        return 0.0

def _neg_score(flat):
    return -_score(flat.reshape(-1, 2))

def _normalize(points):
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(a), radius * np.sin(a)])

def _two_heptagons(r_inner, phi):
    """Two nested regular heptagons: outer at radius 1, inner at r_inner rotated by phi."""
    outer = _regular_polygon(7, 1.0, 0.0)
    inner = _regular_polygon(7, r_inner, phi)
    return np.vstack([outer, inner])

def _score_two_heptagons(params):
    """Score for two-nested-heptagons parameterization."""
    try:
        r_inner, phi = params
        if r_inner < 0:
            r_inner = 0.0
        points = _two_heptagons(r_inner, phi)
        return _score(points)
    except Exception:
        return 0.0

def _neg_score_two_heptagons(params):
    return -_score_two_heptagons(params)

def _exact_maxmin_obj_grad(points):
    """
    Returns (value, grad) for f(points) = -min_triangle_area(points) / hull_area(points).
    Uses the EXACT minimum (no LSE). The subgradient of min_j a_j is a unit vector on
    the set of minimizers, which is valid for maximizing the minimum.
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

    ha = _hull_area(points)
    if ha < 1e-12:
        return 1e10, np.zeros((N, 2))

    m = areas.min()
    tol = 1e-10 * max(1.0, m)
    mask = areas <= (m + tol)
    cnt = mask.sum()
    w = np.zeros(_NTRI)
    w[mask] = 1.0 / cnt

    # Gradient of cross w.r.t. points
    g_a_x = -v2[:, 1]
    g_a_y = v2[:, 0]
    g_b_x = v2[:, 1]
    g_b_y = -v2[:, 0]
    g_c_x = -v1[:, 1]
    g_c_y = v1[:, 0]

    sign_cross = np.sign(cross)
    sign_cross[np.abs(cross) < 1e-15] = 1.0

    wh = 0.5 * w * sign_cross
    grad = np.zeros((N, 2))
    np.add.at(grad, (_I, 0), wh * g_a_x)
    np.add.at(grad, (_I, 1), wh * g_a_y)
    np.add.at(grad, (_J, 0), wh * g_b_x)
    np.add.at(grad, (_J, 1), wh * g_b_y)
    np.add.at(grad, (_K, 0), wh * g_c_x)
    np.add.at(grad, (_K, 1), wh * g_c_y)

    val = -m / ha
    grad = -grad / ha
    return val, grad

def _soft_min_score(x, tau):
    """
    Compute -soft_min(areas) / hull_area.
    soft_min(A) = -tau * log(sum(exp(-A/tau)))
    """
    points = x.reshape(-1, 2)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    areas = np.maximum(areas, 1e-12)
    
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    sum_exp = np.sum(np.exp(scaled - max_scaled))
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    
    return -soft_min_val / ha

def _soft_min_score_grad(x, tau):
    """Returns (value, grad) for soft-min objective."""
    points = x.reshape(-1, 2)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0, np.zeros((N, 2))
    except Exception:
        return 0.0, np.zeros((N, 2))
    
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    areas = np.maximum(areas, 1e-12)
    
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    probs = exp_scaled / sum_exp
    
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    
    g_areas = -probs
    
    g_a_x = -v2[:, 1]
    g_a_y = v2[:, 0]
    g_b_x = v2[:, 1]
    g_b_y = -v2[:, 0]
    g_c_x = -v1[:, 1]
    g_c_y = v1[:, 0]
    
    sign_cross = np.sign(cross)
    sign_cross[np.abs(cross) < 1e-15] = 1.0
    
    g_areas_scaled = 0.5 * g_areas * sign_cross
    
    grad = np.zeros((N, 2))
    np.add.at(grad, (_I, 0), g_areas_scaled * g_a_x)
    np.add.at(grad, (_I, 1), g_areas_scaled * g_a_y)
    np.add.at(grad, (_J, 0), g_areas_scaled * g_b_x)
    np.add.at(grad, (_J, 1), g_areas_scaled * g_b_y)
    np.add.at(grad, (_K, 0), g_areas_scaled * g_c_x)
    np.add.at(grad, (_K, 1), g_areas_scaled * g_c_y)
    
    val = -soft_min_val / ha
    grad = -grad / ha
    return val, grad

def _ring_configuration(params):
    r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4 = params
    n1 = int(round(n1))
    n2 = int(round(n2))
    n3 = int(round(n3))
    n4 = int(round(n4))
    
    total = n1 + n2 + n3 + n4
    while total > N:
        if n4 >= 1:
            n4 -= 1
        elif n3 >= 1:
            n3 -= 1
        elif n2 >= 1:
            n2 -= 1
        else:
            n1 -= 1
        total -= 1
    while total < N:
        n4 += 1
        total += 1
    
    points = []
    
    if n1 > 0:
        angles = th1 + np.linspace(0, 2 * np.pi, n1, endpoint=False)
        points.append(np.column_stack((r1 * np.cos(angles), r1 * np.sin(angles))))
    
    if n2 > 0:
        angles = th2 + np.linspace(0, 2 * np.pi, n2, endpoint=False)
        points.append(np.column_stack((r2 * np.cos(angles), r2 * np.sin(angles))))
    
    if n3 > 0:
        angles = th3 + np.linspace(0, 2 * np.pi, n3, endpoint=False)
        points.append(np.column_stack((r3 * np.cos(angles), r3 * np.sin(angles))))
    
    if n4 > 0:
        angles = th4 + np.linspace(0, 2 * np.pi, n4, endpoint=False)
        points.append(np.column_stack((r4 * np.cos(angles), r4 * np.sin(angles))))
    
    if not points:
        return np.zeros((N, 2))
    
    result = np.vstack(points)
    
    if len(result) < N:
        extra = np.zeros((N - len(result), 2))
        result = np.vstack([result, extra])
    elif len(result) > N:
        result = result[:N]
    
    return result

def _neg_ring_score(params):
    try:
        points = _ring_configuration(params)
        return -_score(points)
    except Exception:
        return 0.0

def _convex_polygon_from_edge_lengths(edge_lengths, angles=None):
    n = len(edge_lengths)
    if angles is None:
        angles = np.full(n, 2 * np.pi / n)
    
    angle_sum = np.sum(angles)
    if angle_sum < 1e-10:
        angles = np.full(n, 2 * np.pi / n)
    else:
        angles = angles / angle_sum * 2 * np.pi
    
    directions = np.cumsum(angles)
    directions = np.roll(directions, -1) - angles / 2
    
    edge_vectors = np.column_stack([
        edge_lengths * np.cos(directions),
        edge_lengths * np.sin(directions)
    ])
    
    vertices = np.cumsum(edge_vectors, axis=0)
    vertices -= vertices.mean(axis=0)
    
    return vertices

def _neg_convex_poly_score(params):
    try:
        n = N
        e = params[:n]
        a = params[n:n + n - 1]
        
        e = np.maximum(e, 0.01)
        a = np.maximum(a, 0.01)
        
        a_full = np.append(a, 2 * np.pi - np.sum(a))
        a_full = np.maximum(a_full, 0.01)
        
        a_sum = np.sum(a_full)
        a_full = a_full / a_sum * 2 * np.pi
        
        vertices = _convex_polygon_from_edge_lengths(e, a_full)
        
        try:
            hull = scipy.spatial.ConvexHull(vertices)
            if len(hull.vertices) < n:
                return 1e10
        except Exception:
            return 1e10
        
        return -_score(vertices)
    except Exception:
        return 1e10

def _edge_length_score(e_flat, angles=None):
    try:
        e = np.maximum(e_flat, 0.01)
        e = e / e.sum() * N
        if angles is None:
            angles = np.full(N, 2 * np.pi / N)
        verts = _convex_polygon_from_edge_lengths(e, angles)
        try:
            hull = scipy.spatial.ConvexHull(verts)
            if len(hull.vertices) < N:
                return 1e10
        except Exception:
            return 1e10
        return -_score(verts)
    except Exception:
        return 1e10

def _angle_optimization(e_fixed):
    def neg_angle_score(a_flat):
        try:
            a = np.maximum(a_flat, 0.01)
            a_full = np.append(a, 2 * np.pi - np.sum(a))
            a_full = np.maximum(a_full, 0.01)
            a_sum = np.sum(a_full)
            a_full = a_full / a_sum * 2 * np.pi
            verts = _convex_polygon_from_edge_lengths(e_fixed, a_full)
            try:
                hull = scipy.spatial.ConvexHull(verts)
                if len(hull.vertices) < N:
                    return 1e10
            except Exception:
                return 1e10
            return -_score(verts)
        except Exception:
            return 1e10
    
    a_init = np.full(N - 1, 2 * np.pi / N)
    bounds = [(0.01, 2 * np.pi - 0.01)] * (N - 1)
    
    try:
        res = scipy.optimize.minimize(neg_angle_score, a_init, method='L-BFGS-B',
                                       options={'maxiter': 500, 'ftol': 1e-12})
        if res.fun < 1e9:
            a_opt = np.maximum(res.x, 0.01)
            a_full = np.append(a_opt, 2 * np.pi - np.sum(a_opt))
            a_full = np.maximum(a_full, 0.01)
            a_full = a_full / a_full.sum() * 2 * np.pi
            return a_full, -res.fun
    except Exception:
        pass
    
    return np.full(N, 2 * np.pi / N), 0.0

def _epigraph_objective(flat, t):
    """
    Epigraph formulation: maximize t subject to all triangle areas >= t * hull_area.
    We minimize -t with a penalty for constraint violations.
    points = flat.reshape(-1,2), t is the minimum area / hull area we want.
    """
    points = flat.reshape(-1, 2)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 1e10
    except Exception:
        return 1e10
    
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Constraint: areas >= t * ha  =>  areas - t*ha >= 0
    # Penalty for violations
    violations = np.maximum(0, t * ha - areas)
    penalty = np.sum(violations ** 2)
    
    return -t + 100.0 * penalty

def _epigraph_objective_grad(flat, t):
    """
    Returns (value, grad) for the epigraph objective with respect to points only.
    t is treated as fixed during this gradient computation.
    """
    points = flat.reshape(-1, 2)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 1e10, np.zeros((N, 2))
    except Exception:
        return 1e10, np.zeros((N, 2))
    
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    threshold = t * ha
    violations = np.maximum(0, threshold - areas)
    
    # Only triangles that are violating contribute to the penalty gradient
    mask = violations > 1e-12
    if not np.any(mask):
        # No violations, gradient is 0 w.r.t. points (t is fixed)
        val = -t
        return val, np.zeros((N, 2))
    
    # Penalty = sum(violations^2) where violations = threshold - areas (for violating ones)
    # d(penalty)/d(points) = sum over violating triangles of 2*violations * d(threshold-areas)/d(points)
    # = sum of 2*violations * (-d(areas)/d(points))
    # = -2 * sum(violations * d(areas)/d(points))
    
    sign_cross = np.sign(cross)
    sign_cross[np.abs(cross) < 1e-15] = 1.0
    
    g_a_x = -v2[:, 1]
    g_a_y = v2[:, 0]
    g_b_x = v2[:, 1]
    g_b_y = -v2[:, 0]
    g_c_x = -v1[:, 1]
    g_c_y = v1[:, 0]
    
    # d(areas)/d(points) = 0.5 * sign(cross) * d(cross)/d(points)
    # For violating triangles, we want to increase areas, so gradient of penalty w.r.t. points:
    # d(penalty)/d(p) = -2 * violations * 0.5 * sign(cross) * d(cross)/d(p)
    #                  = -violations * sign(cross) * d(cross)/d(p)
    
    w = np.zeros(_NTRI)
    w[mask] = -violations[mask] * sign_cross[mask]
    
    grad = np.zeros((N, 2))
    np.add.at(grad, (_I, 0), w * g_a_x)
    np.add.at(grad, (_I, 1), w * g_a_y)
    np.add.at(grad, (_J, 0), w * g_b_x)
    np.add.at(grad, (_J, 1), w * g_b_y)
    np.add.at(grad, (_K, 0), w * g_c_x)
    np.add.at(grad, (_K, 1), w * g_c_y)
    
    # Multiply by 100 (the penalty coefficient)
    grad *= 100.0
    
    val = -t + 100.0 * np.sum(violations ** 2)
    return val, grad

def _alternating_optimize(points, t, max_iter=200, tol=1e-10):
    """
    Alternating optimization:
    1. Fix t, optimize points to minimize penalty (satisfy constraints)
    2. Fix points, update t to the actual min area / hull area
    """
    current_pts = points.copy()
    current_t = t
    
    for it in range(max_iter):
        # Step 1: Optimize points with fixed t
        def obj_and_grad(flat):
            val, grad = _epigraph_objective_grad(flat, current_t)
            return val, grad.flatten()
        
        try:
            res = scipy.optimize.minimize(obj_and_grad, current_pts.flatten(), method='L-BFGS-B',
                                          jac=True, options={'maxiter': 500, 'ftol': 1e-12, 'xatol': 1e-11})
            new_pts = res.x.reshape(-1, 2)
        except Exception:
            new_pts = current_pts
        
        # Step 2: Update t
        new_t = _score(new_pts)
        
        if abs(new_t - current_t) < tol:
            current_t = new_t
            current_pts = new_pts
            break
        
        current_t = new_t
        current_pts = new_pts
    
    return current_pts, current_t

def construct():
    t0 = time.time()
    
    def left():
        return DEADLINE - (time.time() - t0)
    
    best_score = 0.0
    best_pts = _normalize(_regular_polygon(N, 1.0, 0.0))
    
    def try_update(points):
        nonlocal best_score, best_pts
        sc = _score(points)
        if sc > best_score:
            best_score = sc
            best_pts = points.copy()
        return sc
    
    # ---- Phase 0: Two-nested-heptagons 2D global search ----
    if left() > 5.0:
        r_vals = np.linspace(0.0, 1.0, 50)
        phi_vals = np.linspace(0.0, np.pi / 7, 50, endpoint=False)
        
        for r in r_vals:
            if left() <= 3.0:
                break
            for phi in phi_vals:
                pts = _two_heptagons(r, phi)
                try_update(pts)
        
        if left() > 5.0:
            best_r, best_phi, best_s = 0.0, 0.0, 0.0
            r_vals = np.linspace(0.0, 1.0, 30)
            phi_vals = np.linspace(0.0, np.pi / 7, 30, endpoint=False)
            for r in r_vals:
                for phi in phi_vals:
                    s = _score_two_heptagons([r, phi])
                    if s > best_s:
                        best_s, best_r, best_phi = s, r, phi
            
            gr = 0.6180339887498949
            a_r, b_r = max(0.0, best_r - 0.1), min(1.0, best_r + 0.1)
            c_r = b_r - gr * (b_r - a_r)
            d_r = a_r + gr * (b_r - a_r)
            fc = _score_two_heptagons([c_r, best_phi])
            fd = _score_two_heptagons([d_r, best_phi])
            for _ in range(40):
                if left() <= 2.0:
                    break
                if fc < fd:
                    a_r = c_r
                    c_r = d_r
                    fc = fd
                    d_r = b_r - gr * (b_r - a_r)
                    fd = _score_two_heptagons([d_r, best_phi])
                else:
                    b_r = d_r
                    d_r = c_r
                    fd = fc
                    c_r = a_r + gr * (b_r - a_r)
                    fc = _score_two_heptagons([c_r, best_phi])
            best_r = (a_r + b_r) / 2
            
            a_p, b_p = 0.0, np.pi / 7
            c_p = b_p - gr * (b_p - a_p)
            d_p = a_p + gr * (b_p - a_p)
            fc = _score_two_heptagons([best_r, c_p])
            fd = _score_two_heptagons([best_r, d_p])
            for _ in range(40):
                if left() <= 2.0:
                    break
                if fc < fd:
                    a_p = c_p
                    c_p = d_p
                    fc = fd
                    d_p = b_p - gr * (b_p - a_p)
                    fd = _score_two_heptagons([best_r, d_p])
                else:
                    b_p = d_p
                    d_p = c_p
                    fd = fc
                    c_p = a_p + gr * (b_p - a_p)
                    fc = _score_two_heptagons([best_r, c_p])
            best_phi = (a_p + b_p) / 2
            
            pts = _two_heptagons(best_r, best_phi)
            try_update(pts)
    
    # ---- Phase 1: Convex polygon edge-length optimization ----
    if left() > 40.0:
        reg_poly = _regular_polygon(N, 1.0, 0.0)
        
        if left() > 30.0:
            x0 = np.full(N, 1.0)
            bounds = [(0.1, 3.0)] * N
            try:
                res = scipy.optimize.differential_evolution(
                    _edge_length_score, bounds, seed=42, maxiter=50,
                    tol=1e-5, popsize=10, mutation=(0.3, 1.2),
                    recombination=0.9, polish=True, init='latinhypercube'
                )
                if res.fun < 1e9:
                    e_opt = np.maximum(res.x, 0.01)
                    e_opt = e_opt / e_opt.sum() * N
                    angles = np.full(N, 2 * np.pi / N)
                    verts = _convex_polygon_from_edge_lengths(e_opt, angles)
                    try_update(verts)
            except Exception:
                pass
        
        if left() > 20.0:
            e_init = np.full(N, 1.0)
            a_init = np.full(N - 1, 2 * np.pi / N)
            x0 = np.concatenate([e_init, a_init])
            
            bounds = [(0.1, 3.0)] * N + [(0.01, 2 * np.pi - 0.01)] * (N - 1)
            
            try:
                res = scipy.optimize.differential_evolution(
                    _neg_convex_poly_score, bounds, seed=123, maxiter=30,
                    tol=1e-4, popsize=8, mutation=(0.4, 1.5),
                    recombination=0.8, polish=True, init='latinhypercube'
                )
                if res.fun < 1e9:
                    e_opt = np.maximum(res.x[:N], 0.01)
                    a_opt = np.maximum(res.x[N:], 0.01)
                    a_full = np.append(a_opt, 2 * np.pi - np.sum(a_opt))
                    a_full = np.maximum(a_full, 0.01)
                    a_full = a_full / a_full.sum() * 2 * np.pi
                    verts = _convex_polygon_from_edge_lengths(e_opt, a_full)
                    try_update(verts)
            except Exception:
                pass
    
    # ---- Phase 1b: Coordinate descent on edge lengths and angles ----
    if left() > 25.0:
        e_current = np.full(N, 1.0)
        a_current = np.full(N, 2 * np.pi / N)
        
        try:
            hull = scipy.spatial.ConvexHull(best_pts)
            if len(hull.vertices) == N:
                verts_ordered = best_pts[hull.vertices]
                edges = np.diff(np.vstack([verts_ordered, verts_ordered[0]]), axis=0)
                edge_lengths = np.linalg.norm(edges, axis=1)
                e_current = edge_lengths.copy()
                
                for i in range(N):
                    v1 = edges[i]
                    v2 = edges[(i + 1) % N]
                    cross_z = v1[0] * v2[1] - v1[1] * v2[0]
                    dot = np.dot(v1, v2)
                    angle = np.arctan2(cross_z, dot)
                    if angle < 0:
                        angle += 2 * np.pi
                    a_current[i] = angle
        except Exception:
            pass
        
        for it in range(15):
            if left() <= 5.0:
                break
            
            if left() > 10.0:
                try:
                    def neg_e_score(e_flat):
                        return _edge_length_score(e_flat, a_current)
                    
                    res = scipy.optimize.minimize(neg_e_score, e_current, method='L-BFGS-B',
                                                   options={'maxiter': 300, 'ftol': 1e-12})
                    if res.fun < 1e9:
                        e_new = np.maximum(res.x, 0.01)
                        e_new = e_new / e_new.sum() * N
                        verts = _convex_polygon_from_edge_lengths(e_new, a_current)
                        sc = _score(verts)
                        if sc > best_score:
                            best_score = sc
                            best_pts = verts.copy()
                        e_current = e_new
                except Exception:
                    pass
            
            if left() > 5.0:
                a_opt, sc_opt = _angle_optimization(e_current)
                if sc_opt > best_score:
                    verts = _convex_polygon_from_edge_lengths(e_current, a_opt)
                    try_update(verts)
                    a_current = a_opt
    
    # ---- Phase 2: Differential evolution on 4-ring parameters ----
    if left() > 30.0:
        bounds = [
            (0.3, 1.5),
            (0.1, 1.0),
            (0.0, 0.7),
            (0.0, 0.4),
            (0, 2 * np.pi),
            (0, 2 * np.pi),
            (0, 2 * np.pi),
            (0, 2 * np.pi),
            (1, 14),
            (1, 14),
            (1, 14),
            (1, 14),
        ]
        
        rng = np.random.default_rng(42)
        for restart in range(2):
            if left() <= 20.0:
                break
            try:
                res = scipy.optimize.differential_evolution(
                    _neg_ring_score,
                    bounds,
                    seed=rng.integers(0, 2**31),
                    maxiter=40,
                    tol=1e-5,
                    popsize=8,
                    mutation=(0.3, 1.2),
                    recombination=0.9,
                    polish=True,
                    init='latinhypercube'
                )
                if res.success or res.fun < 0:
                    points = _ring_configuration(res.x)
                    try_update(points)
            except Exception:
                pass
    
    # ---- Phase 3: Structured starts + gradient-based local refinement ----
    if left() > 20.0:
        starts = [best_pts.copy()]
        starts.append(_normalize(_regular_polygon(N, 1.0, 0.0)))
        
        a_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a_in = a_out + np.pi / 7
        starts.append(_normalize(np.vstack((
            np.column_stack((np.cos(a_out), np.sin(a_out))),
            np.column_stack((0.5 * np.cos(a_in), 0.5 * np.sin(a_in)))
        ))))
        
        a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        a2 = a1 + np.pi / 5
        a3 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        starts.append(_normalize(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((0.6 * np.cos(a2), 0.6 * np.sin(a2))),
            np.column_stack((0.25 * np.cos(a3), 0.25 * np.sin(a3)))
        ))))
        
        a1 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        a2 = a1 + np.pi / 4
        a3 = np.linspace(0, 2 * np.pi, 3, endpoint=False) + np.pi / 3
        a4 = a3 + np.pi / 3
        starts.append(_normalize(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((0.65 * np.cos(a2), 0.65 * np.sin(a2))),
            np.column_stack((0.35 * np.cos(a3), 0.35 * np.sin(a3))),
            np.column_stack((0.1 * np.cos(a4), 0.1 * np.sin(a4)))
        ))))
        
        rng2 = np.random.default_rng(123)
        for _ in range(3):
            if left() <= 10.0:
                break
            jittered = best_pts + rng2.normal(0, 0.08, (N, 2))
            starts.append(_normalize(jittered))
        
        def try_opt(x0, method, maxiter, fatol, xatol, use_grad=False):
            nonlocal best_score, best_pts
            if left() <= 3.0:
                return
            try:
                if use_grad:
                    def obj_and_grad(flat):
                        val, grad = _exact_maxmin_obj_grad(flat.reshape(-1, 2))
                        return val, grad.flatten()
                    opts = {'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol,
                            'maxfev': 8000, 'maxls': 30}
                    res = scipy.optimize.minimize(obj_and_grad, x0.flatten(), method=method,
                                                  jac=True, options=opts)
                else:
                    opts = {'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol, 'maxfev': 12000}
                    if method == 'Nelder-Mead':
                        opts['adaptive'] = True
                    res = scipy.optimize.minimize(_neg_score, x0.flatten(), method=method, options=opts)
                opt = _normalize(res.x.reshape(-1, 2))
                sc = _score(opt)
                if sc > best_score:
                    best_score, best_pts = sc, opt.copy()
            except Exception:
                pass
        
        for start in starts:
            if left() <= 8.0:
                break
            try_opt(start, 'L-BFGS-B', 1500, 1e-12, 1e-11, use_grad=True)
            if left() <= 6.0:
                break
            try_opt(start, 'Nelder-Mead', 2000, 1e-12, 1e-9)
    
    # ---- Phase 4: Soft-min annealing ----
    if left() > 15.0:
        tau_vals = [0.05, 0.02, 0.01, 0.005, 0.002]
        for tau in tau_vals:
            if left() <= 4.0:
                break
            try:
                def obj_and_grad(flat):
                    val, grad = _soft_min_score_grad(flat, tau)
                    return val, grad.flatten()
                res = scipy.optimize.minimize(obj_and_grad, best_pts.flatten(), method='L-BFGS-B',
                                              jac=True, options={'maxiter': 800, 'fatol': 1e-12, 'xatol': 1e-11})
                opt = _normalize(res.x.reshape(-1, 2))
                sc = _score(opt)
                if sc > best_score:
                    best_score, best_pts = sc, opt.copy()
            except Exception:
                pass
    
    # ---- Phase 5: Epigraph alternating optimization (NEW) ----
    if left() > 15.0:
        # Start from current best
        t_init = best_score
        try:
            opt_pts, opt_t = _alternating_optimize(best_pts, t_init, max_iter=100, tol=1e-10)
            sc = _score(opt_pts)
            if sc > best_score:
                best_score = sc
                best_pts = opt_pts.copy()
        except Exception:
            pass
        
        # Try from a few other starts
        if left() > 10.0:
            other_starts = [
                _normalize(_regular_polygon(N, 1.0, 0.0)),
                _two_heptagons(0.5, np.pi / 7),
            ]
            for s in other_starts:
                if left() <= 5.0:
                    break
                try:
                    opt_pts, opt_t = _alternating_optimize(s, _score(s), max_iter=80, tol=1e-10)
                    sc = _score(opt_pts)
                    if sc > best_score:
                        best_score = sc
                        best_pts = opt_pts.copy()
                except Exception:
                    pass
    
    # ---- Phase 6: Aggressive high-precision polish ----
    polish_configs = [
        ('L-BFGS-B', 3000, 1e-14, 1e-13, True),
        ('Nelder-Mead', 4000, 1e-15, 1e-14, False),
        ('L-BFGS-B', 3000, 1e-15, 1e-14, True),
        ('Nelder-Mead', 5000, 1e-16, 1e-15, False),
        ('L-BFGS-B', 2500, 1e-16, 1e-15, True),
        ('Nelder-Mead', 6000, 1e-17, 1e-16, False),
    ]
    
    for method, maxiter, fatol, xatol, use_grad in polish_configs:
        if left() <= 2.0:
            break
        try_opt(best_pts, method, maxiter, fatol, xatol, use_grad=use_grad)
    
    # ---- Phase 7: Coordinate-wise optimization ----
    if left() > 5.0:
        current_pts = best_pts.copy()
        improved = True
        max_rounds = 3
        
        for round_num in range(max_rounds):
            if left() <= 2.0:
                break
            if not improved:
                break
            improved = False
            
            for i in range(N):
                if left() <= 1.0:
                    break
                
                def neg_point_score(xy):
                    pts = current_pts.copy()
                    pts[i] = xy
                    return -_score(pts)
                
                x0 = current_pts[i].copy()
                try:
                    res = scipy.optimize.minimize(neg_point_score, x0, method='Nelder-Mead',
                                                  options={'maxiter': 500, 'fatol': 1e-14, 'xatol': 1e-13})
                    new_pt = res.x
                    sc = _score(np.vstack([current_pts[:i], new_pt.reshape(1,2), current_pts[i+1:]]))
                    if sc > best_score:
                        best_score = sc
                        current_pts[i] = new_pt
                        improved = True
                except Exception:
                    pass
        
        if best_score > 0:
            best_pts = current_pts.copy()
    
    # ---- Phase 8: Final epigraph polish ----
    if left() > 5.0:
        try:
            opt_pts, opt_t = _alternating_optimize(best_pts, best_score, max_iter=50, tol=1e-12)
            sc = _score(opt_pts)
            if sc > best_score:
                best_score = sc
                best_pts = opt_pts.copy()
        except Exception:
            pass
    
    # Final check and return
    return _normalize(best_pts)
