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
        scaled = points / np.sqrt(ha)
        # Verify the hull area is close to 1
        ha_check = _hull_area(scaled)
        if ha_check < 1e-6:
            return points
        return scaled
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

    g_a_x = v1[:, 1] - v2[:, 1]
    g_a_y = v2[:, 0] - v1[:, 0]
    g_b_x = v2[:, 1]
    g_b_y = -v2[:, 0]
    g_c_x = -v1[:, 1]
    g_c_y = v1[:, 0]

    wh = 0.5 * w
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
    This is differentiable and approximates -min(A) as tau -> 0.
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

def _hull_area_grad(points):
    """Compute gradient of convex hull area w.r.t. vertex positions."""
    try:
        hull = scipy.spatial.ConvexHull(points)
        verts = hull.vertices
        n_v = len(verts)
        if n_v < 3:
            return np.zeros((N, 2))
        # Sort vertices in order along hull
        # Use the hull's ridge structure to get ordered vertices
        # Simpler: use the fact that hull.vertices are in CCW order from scipy
        # Actually scipy returns them in arbitrary order; let's sort by angle from centroid
        centroid = points[verts].mean(axis=0)
        angles = np.arctan2(points[verts, 1] - centroid[1], points[verts, 0] - centroid[0])
        order = np.argsort(angles)
        ordered_verts = verts[order]
        
        grad = np.zeros((N, 2))
        # Area = 0.5 * sum of cross products of consecutive vertices
        # d(area)/d(p_i) = 0.5 * (p_{i-1} x p_{i+1} - p_{i-1} x p_i + p_i x p_{i+1} - p_{i-1} x p_i)
        # More carefully: area = 0.5 * sum_j (x_j * y_{j+1} - x_{j+1} * y_j)
        # d(area)/d(x_j) = 0.5 * (y_{j+1} - y_{j-1})
        # d(area)/d(y_j) = 0.5 * (x_{j-1} - x_{j+1})
        for k in range(n_v):
            vi = ordered_verts[k]
            vp = ordered_verts[(k - 1) % n_v]
            vn = ordered_verts[(k + 1) % n_v]
            grad[vi, 0] += 0.5 * (points[vn, 1] - points[vp, 1])
            grad[vi, 1] += 0.5 * (points[vp, 0] - points[vn, 0])
        return grad
    except Exception:
        return np.zeros((N, 2))

def _soft_min_score_grad(x, tau):
    """Returns (value, grad) for -soft_min(areas)/hull_area with proper quotient rule."""
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
    probs = exp_scaled / sum_exp  # softmax weights
    
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    
    # Gradient of soft_min w.r.t. areas
    g_areas = -probs
    
    # Gradient of areas w.r.t. points
    g_a_x = v1[:, 1] - v2[:, 1]
    g_a_y = v2[:, 0] - v1[:, 0]
    g_b_x = v2[:, 1]
    g_b_y = -v2[:, 0]
    g_c_x = -v1[:, 1]
    g_c_y = v1[:, 0]
    
    grad_num = np.zeros((N, 2))
    np.add.at(grad_num, (_I, 0), g_areas * g_a_x)
    np.add.at(grad_num, (_I, 1), g_areas * g_a_y)
    np.add.at(grad_num, (_J, 0), g_areas * g_b_x)
    np.add.at(grad_num, (_J, 1), g_areas * g_b_y)
    np.add.at(grad_num, (_K, 0), g_areas * g_c_x)
    np.add.at(grad_num, (_K, 1), g_areas * g_c_y)
    
    # Quotient rule: f = soft_min / ha
    # grad(f) = (grad(soft_min) * ha - soft_min * grad(ha)) / ha^2
    grad_ha = _hull_area_grad(points)
    
    val = soft_min_val / ha
    grad = (grad_num * ha - soft_min_val * grad_ha) / (ha * ha)
    
    # We want to MINIMIZE -f, so return (-val, -grad)
    return -val, -grad

def _ring_configuration(params):
    """
    Generate points from 4-ring parameters.
    params: [r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4]
    where n1+n2+n3+n4=14, r_i >= 0, theta offsets in [0, 2pi)
    """
    r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4 = params
    n1 = int(round(n1))
    n2 = int(round(n2))
    n3 = int(round(n3))
    n4 = int(round(n4))
    
    # Ensure we have exactly 14 points
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
    """
    Construct a convex polygon from edge lengths and turning angles.
    For a convex polygon, the sum of exterior angles is 2*pi.
    We parameterize by the turning angles (which sum to 2*pi).
    """
    n = len(edge_lengths)
    if angles is None:
        angles = np.full(n, 2 * np.pi / n)
    
    # Normalize angles to sum to 2*pi
    angle_sum = np.sum(angles)
    if angle_sum < 1e-10:
        angles = np.full(n, 2 * np.pi / n)
    else:
        angles = angles / angle_sum * 2 * np.pi
    
    # Compute directions
    directions = np.cumsum(angles)
    directions = np.roll(directions, -1) - angles / 2
    
    # Compute edge vectors
    edge_vectors = np.column_stack([
        edge_lengths * np.cos(directions),
        edge_lengths * np.sin(directions)
    ])
    
    # Compute vertices (starting at origin, then close the polygon)
    vertices = np.cumsum(edge_vectors, axis=0)
    
    # Center the polygon
    vertices -= vertices.mean(axis=0)
    
    return vertices

def _neg_convex_poly_score(params):
    """
    Objective for convex polygon optimization.
    params: [e1, e2, ..., e14, a1, a2, ..., a13]
    where e_i are edge lengths and a_i are turning angles (a14 determined by closure)
    """
    try:
        n = N
        e = params[:n]
        a = params[n:n + n - 1]  # n-1 angles, last determined by closure
        
        # Ensure positive edge lengths
        e = np.maximum(e, 0.01)
        
        # Normalize angles
        a = np.maximum(a, 0.01)
        
        # Compute the last angle to close the polygon
        a_full = np.append(a, 2 * np.pi - np.sum(a))
        a_full = np.maximum(a_full, 0.01)
        
        # Normalize to sum to 2*pi
        a_sum = np.sum(a_full)
        a_full = a_full / a_sum * 2 * np.pi
        
        vertices = _convex_polygon_from_edge_lengths(e, a_full)
        
        # Check convexity
        try:
            hull = scipy.spatial.ConvexHull(vertices)
            if len(hull.vertices) < n:
                return 1e10  # Not convex
        except Exception:
            return 1e10
        
        return -_score(vertices)
    except Exception:
        return 1e10

def _dumbbell_points(centers, v):
    """
    Generate 14 points from 7 pair-centers and a separation vector v.
    centers: (7, 2) array
    v: (2,) array
    Returns (14, 2) array
    """
    p1 = centers + v / 2.0
    p2 = centers - v / 2.0
    return np.vstack([p1, p2])

def _neg_dumbbell_score(params):
    """
    Objective for dumbbell topology optimization.
    params: [c1x, c1y, c2x, c2y, ..., c7x, c7y, vx, vy]
    14 + 2 = 16 parameters
    """
    try:
        centers = params[:14].reshape(7, 2)
        v = params[14:16]
        # Clamp v magnitude to prevent degenerate configurations
        vmag = np.linalg.norm(v)
        if vmag > 0.5:
            v = v * 0.5 / vmag
        points = _dumbbell_points(centers, v)
        sc = _score(points)
        return -sc
    except Exception:
        return 1e10

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
    
    # ---- Phase 0: Two-nested-heptagons 2D global search (fast, exploits symmetry) ----
    if left() > 5.0:
        # Dense grid over (r_inner, phi)
        r_vals = np.linspace(0.0, 1.0, 50)
        phi_vals = np.linspace(0.0, np.pi / 7, 50, endpoint=False)
        
        for r in r_vals:
            if left() <= 3.0:
                break
            for phi in phi_vals:
                pts = _two_heptagons(r, phi)
                try_update(pts)
        
        # Golden-section refinement on the best region
        if left() > 5.0:
            # Find best (r, phi) from grid
            best_r, best_phi, best_s = 0.0, 0.0, 0.0
            r_vals = np.linspace(0.0, 1.0, 30)
            phi_vals = np.linspace(0.0, np.pi / 7, 30, endpoint=False)
            for r in r_vals:
                for phi in phi_vals:
                    s = _score_two_heptagons([r, phi])
                    if s > best_s:
                        best_s, best_r, best_phi = s, r, phi
            
            # 1D golden-section on r for fixed phi
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
            
            # 1D golden-section on phi for fixed r
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
    
    # ---- Phase 1: Expanded two-ring and three-ring parameterized search ----
    if left() > 35.0:
        # Two-ring: 7+7 with independent radius, rotation, and slight eccentricity
        def _two_ring_general(params):
            r1, r2, phi1, phi2, ecc1, ecc2 = params
            r1 = max(0.1, r1)
            r2 = max(0.0, r2)
            pts = []
            a1 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + phi1
            pts.append(np.column_stack((r1 * (1 + ecc1 * np.cos(a1)) * np.cos(a1),
                                        r1 * (1 + ecc1 * np.sin(a1)) * np.sin(a1))))
            a2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + phi2
            pts.append(np.column_stack((r2 * (1 + ecc2 * np.cos(a2)) * np.cos(a2),
                                        r2 * (1 + ecc2 * np.sin(a2)) * np.sin(a2))))
            return np.vstack(pts)
        
        def _neg_two_ring_general(params):
            try:
                pts = _two_ring_general(params)
                return -_score(pts)
            except Exception:
                return 1e10
        
        bounds_2ring = [
            (0.5, 2.0),   # r1
            (0.0, 1.5),   # r2
            (0, 2 * np.pi),  # phi1
            (0, 2 * np.pi),  # phi2
            (0, 0.3),     # ecc1
            (0, 0.3),     # ecc2
        ]
        
        try:
            res = scipy.optimize.differential_evolution(
                _neg_two_ring_general, bounds_2ring, seed=42, maxiter=60,
                tol=1e-6, popsize=12, mutation=(0.3, 1.2),
                recombination=0.9, polish=True, init='latinhypercube'
            )
            if res.fun < 0:
                pts = _two_ring_general(res.x)
                try_update(pts)
        except Exception:
            pass
        
        # Three-ring: 5+5+4
        if left() > 20.0:
            def _three_ring(params):
                r1, r2, r3, phi1, phi2, phi3 = params
                r1 = max(0.1, r1)
                r2 = max(0.0, r2)
                r3 = max(0.0, r3)
                pts = []
                a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + phi1
                pts.append(np.column_stack((r1 * np.cos(a1), r1 * np.sin(a1))))
                a2 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + phi2
                pts.append(np.column_stack((r2 * np.cos(a2), r2 * np.sin(a2))))
                a3 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + phi3
                pts.append(np.column_stack((r3 * np.cos(a3), r3 * np.sin(a3))))
                return np.vstack(pts)
            
            def _neg_three_ring(params):
                try:
                    pts = _three_ring(params)
                    return -_score(pts)
                except Exception:
                    return 1e10
            
            bounds_3ring = [
                (0.5, 2.0),
                (0.1, 1.5),
                (0.0, 1.0),
                (0, 2 * np.pi),
                (0, 2 * np.pi),
                (0, 2 * np.pi),
            ]
            
            try:
                res = scipy.optimize.differential_evolution(
                    _neg_three_ring, bounds_3ring, seed=99, maxiter=50,
                    tol=1e-6, popsize=10, mutation=(0.3, 1.2),
                    recombination=0.9, polish=True, init='latinhypercube'
                )
                if res.fun < 0:
                    pts = _three_ring(res.x)
                    try_update(pts)
            except Exception:
                pass
    
    # ---- Phase 2: Differential evolution on 4-ring parameters (global search) ----
    if left() > 30.0:
        bounds = [
            (0.3, 1.5),   # r1 (outermost)
            (0.1, 1.0),   # r2
            (0.0, 0.7),   # r3
            (0.0, 0.4),   # r4 (innermost)
            (0, 2 * np.pi),  # th1
            (0, 2 * np.pi),  # th2
            (0, 2 * np.pi),  # th3
            (0, 2 * np.pi),  # th4
            (1, 14),      # n1
            (1, 14),      # n2
            (1, 14),      # n3
            (1, 14),      # n4
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
        # Build diverse starting points
        starts = [best_pts.copy()]
        
        # Regular polygon
        starts.append(_normalize(_regular_polygon(N, 1.0, 0.0)))
        
        # Two rings 7+7 with different radii
        a_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a_in = a_out + np.pi / 7
        for r_in in [0.4, 0.6, 0.8]:
            starts.append(_normalize(np.vstack((
                np.column_stack((np.cos(a_out), np.sin(a_out))),
                np.column_stack((r_in * np.cos(a_in), r_in * np.sin(a_in)))
            ))))
        
        # Three rings 5+5+4
        a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        a2 = a1 + np.pi / 5
        a3 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        for (r2, r3) in [(0.5, 0.2), (0.6, 0.35)]:
            starts.append(_normalize(np.vstack((
                np.column_stack((np.cos(a1), np.sin(a1))),
                np.column_stack((r2 * np.cos(a2), r2 * np.sin(a2))),
                np.column_stack((r3 * np.cos(a3), r3 * np.sin(a3)))
            ))))
        
        # Jittered versions of best (smaller jitters for refinement)
        rng2 = np.random.default_rng(123)
        for _ in range(4):
            if left() <= 10.0:
                break
            jittered = best_pts + rng2.normal(0, 0.05, (N, 2))
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
        
        # Local refinement on each start
        for start in starts:
            if left() <= 8.0:
                break
            try_opt(start, 'L-BFGS-B', 1500, 1e-12, 1e-11, use_grad=True)
            if left() <= 6.0:
                break
            try_opt(start, 'Nelder-Mead', 2000, 1e-12, 1e-9)
    
    # ---- Phase 3.5: Dumbbell topology search (7 pairs with shared separation vector) ----
    if left() > 40.0:
        # Build initial dumbbell configurations from existing best points
        # Sort points by angle from centroid to identify natural pairs
        centroid = best_pts.mean(axis=0)
        angles = np.arctan2(best_pts[:, 1] - centroid[1], best_pts[:, 0] - centroid[0])
        sorted_idx = np.argsort(angles)
        sorted_pts = best_pts[sorted_idx]
        
        # Try different pairing strategies
        dumbbell_starts = []
        
        # Strategy 1: Adjacent pairs in angular order
        centers1 = np.vstack([sorted_pts[0:2:2].mean(axis=0), 
                              sorted_pts[1:2:2].mean(axis=0),
                              sorted_pts[2:4:2].mean(axis=0),
                              sorted_pts[3:4:2].mean(axis=0),
                              sorted_pts[4:6:2].mean(axis=0),
                              sorted_pts[5:6:2].mean(axis=0),
                              sorted_pts[6:8:2].mean(axis=0)])
        # Actually, let's do this more carefully
        centers1 = np.zeros((7, 2))
        for i in range(7):
            centers1[i] = (sorted_pts[2*i] + sorted_pts[2*i+1]) / 2.0
        v1 = (sorted_pts[1] - sorted_pts[0]) / 2.0
        dumbbell_starts.append(np.concatenate([centers1.flatten(), v1]))
        
        # Strategy 2: Opposite pairs
        centers2 = np.zeros((7, 2))
        for i in range(7):
            centers2[i] = (sorted_pts[i] + sorted_pts[(i+7)%14]) / 2.0
        v2 = (sorted_pts[7] - sorted_pts[0]) / 2.0
        dumbbell_starts.append(np.concatenate([centers2.flatten(), v2]))
        
        # Strategy 3: Random pairing
        rng_db = np.random.default_rng(42)
        perm = rng_db.permutation(14)
        centers3 = np.zeros((7, 2))
        for i in range(7):
            centers3[i] = (best_pts[perm[2*i]] + best_pts[perm[2*i+1]]) / 2.0
        v3 = (best_pts[perm[1]] - best_pts[perm[0]]) / 2.0
        dumbbell_starts.append(np.concatenate([centers3.flatten(), v3]))
        
        # Strategy 4: Regular heptagon centers with small v
        a_reg = np.linspace(0, 2*np.pi, 7, endpoint=False)
        centers4 = np.column_stack([np.cos(a_reg), np.sin(a_reg)])
        v4 = np.array([0.05, 0.0])
        dumbbell_starts.append(np.concatenate([centers4.flatten(), v4]))
        
        # Strategy 5: Two nested heptagons as dumbbell
        a_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
        a_in = a_out + np.pi/7
        centers5 = np.zeros((7, 2))
        for i in range(7):
            centers5[i] = (np.array([np.cos(a_out[i]), np.sin(a_out[i])]) + 
                          0.7 * np.array([np.cos(a_in[i]), np.sin(a_in[i])])) / 2.0
        v5 = (np.array([np.cos(a_out[0]), np.sin(a_out[0])]) - 
              0.7 * np.array([np.cos(a_in[0]), np.sin(a_in[0])])) / 2.0
        dumbbell_starts.append(np.concatenate([centers5.flatten(), v5]))
        
        # Differential evolution on dumbbell parameters
        bounds_db = []
        for i in range(7):
            bounds_db.append((-2.0, 2.0))  # center x
            bounds_db.append((-2.0, 2.0))  # center y
        bounds_db.append((-0.3, 0.3))  # vx
        bounds_db.append((-0.3, 0.3))  # vy
        
        # Run DE from multiple starts
        for start in dumbbell_starts:
            if left() <= 15.0:
                break
            try:
                # First try a quick local optimization from this start
                res_local = scipy.optimize.minimize(
                    _neg_dumbbell_score, start, method='L-BFGS-B',
                    bounds=bounds_db, options={'maxiter': 500, 'ftol': 1e-10, 'maxfev': 3000}
                )
                if res_local.fun < -best_score:
                    pts = _dumbbell_points(res_local.x[:14].reshape(7, 2), res_local.x[14:16])
                    try_update(pts)
            except Exception:
                pass
        
        # Full DE search
        if left() > 20.0:
            try:
                res_de = scipy.optimize.differential_evolution(
                    _neg_dumbbell_score, bounds_db, seed=123, maxiter=80,
                    tol=1e-6, popsize=15, mutation=(0.4, 1.0),
                    recombination=0.9, polish=True, init='latinhypercube'
                )
                if res_de.fun < -best_score:
                    pts = _dumbbell_points(res_de.x[:14].reshape(7, 2), res_de.x[14:16])
                    try_update(pts)
            except Exception:
                pass
        
        # Fine polish on best dumbbell result
        if left() > 10.0:
            # Find the best dumbbell configuration
            best_db_params = None
            best_db_score = -np.inf
            for start in dumbbell_starts:
                try:
                    res = scipy.optimize.minimize(
                        _neg_dumbbell_score, start, method='L-BFGS-B',
                        bounds=bounds_db, options={'maxiter': 2000, 'ftol': 1e-12, 'maxfev': 10000}
                    )
                    if res.fun < best_db_score:
                        best_db_score = res.fun
                        best_db_params = res.x
                except Exception:
                    pass
            
            if best_db_params is not None:
                pts = _dumbbell_points(best_db_params[:14].reshape(7, 2), best_db_params[14:16])
                sc = _score(pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = pts.copy()
                # Final high-precision polish
                if left() > 5.0:
                    try_opt(best_pts, 'L-BFGS-B', 5000, 1e-14, 1e-13, use_grad=True)
    
    # ---- Phase 4: Soft-min annealing for smooth global navigation ----
    if left() > 25.0:
        # Build diverse starting points for soft-min annealing
        soft_starts = [best_pts.copy()]
        
        # Add jittered versions of best
        rng_soft = np.random.default_rng(777)
        for jitter in [0.03, 0.06, 0.10, 0.15]:
            if left() <= 12.0:
                break
            jittered = best_pts + rng_soft.normal(0, jitter, (N, 2))
            soft_starts.append(_normalize(jittered))
        
        # Add ring-based starts
        a_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a_in = a_out + np.pi / 7
        for r_in in [0.2, 0.4, 0.6, 0.8]:
            if left() <= 10.0:
                break
            ring_pts = np.vstack((
                np.column_stack((np.cos(a_out), np.sin(a_out))),
                np.column_stack((r_in * np.cos(a_in), r_in * np.sin(a_in)))
            ))
            soft_starts.append(_normalize(ring_pts))
        
        # Three-ring starts
        a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        a2 = a1 + np.pi / 5
        a3 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        for (r2, r3) in [(0.5, 0.2), (0.6, 0.3), (0.7, 0.4)]:
            if left() <= 8.0:
                break
            ring_pts = np.vstack((
                np.column_stack((np.cos(a1), np.sin(a1))),
                np.column_stack((r2 * np.cos(a2), r2 * np.sin(a2))),
                np.column_stack((r3 * np.cos(a3), r3 * np.sin(a3)))
            ))
            soft_starts.append(_normalize(ring_pts))
        
        tau_vals = [0.08, 0.04, 0.02, 0.01, 0.005, 0.002, 0.001]
        for tau in tau_vals:
            if left() <= 6.0:
                break
            best_in_tau = None
            best_sc_in_tau = best_score
            for start in soft_starts[:]:
                if left() <= 4.0:
                    break
                try:
                    def obj_and_grad(flat, _tau=tau):
                        val, grad = _soft_min_score_grad(flat, _tau)
                        return val, grad.flatten()
                    res = scipy.optimize.minimize(obj_and_grad, start.flatten(), method='L-BFGS-B',
                                                  jac=True, options={'maxiter': 800, 'fatol': 1e-13, 'xatol': 1e-12, 'maxfev': 5000})
                    opt = _normalize(res.x.reshape(-1, 2))
                    sc = _score(opt)
                    if sc > best_sc_in_tau:
                        best_sc_in_tau = sc
                        best_in_tau = opt.copy()
                except Exception:
                    pass
            if best_in_tau is not None:
                best_score = best_sc_in_tau
                best_pts = best_in_tau
                # Use the best result as primary start for next tau, keep a couple others
                soft_starts = [best_in_tau.copy()]
                if len(soft_starts) < 3 and left() > 8.0:
                    soft_starts.append(best_pts + rng_soft.normal(0, 0.05, (N, 2)))
    
    # ---- Phase 5: Fine polish on the best configuration ----
    if left() > 15.0:
        try_opt(best_pts, 'L-BFGS-B', 4000, 1e-13, 1e-12, use_grad=True)
    if left() > 12.0:
        try_opt(best_pts, 'Nelder-Mead', 5000, 1e-14, 1e-12)
    if left() > 10.0:
        try_opt(best_pts, 'L-BFGS-B', 3000, 1e-14, 1e-13, use_grad=True)
    if left() > 8.0:
        try_opt(best_pts, 'Nelder-Mead', 4000, 1e-15, 1e-14)
    if left() > 6.0:
        try_opt(best_pts, 'L-BFGS-B', 2500, 1e-15, 1e-14, use_grad=True)
    if left() > 4.5:
        try_opt(best_pts, 'Nelder-Mead', 3000, 1e-16, 1e-15)
    if left() > 3.0:
        try_opt(best_pts, 'L-BFGS-B', 2000, 1e-16, 1e-15, use_grad=True)
    if left() > 2.0:
        try_opt(best_pts, 'Nelder-Mead', 2000, 1e-17, 1e-16)
    
    # ---- Phase 6: Final dumbbell refinement with tighter bounds ----
    if left() > 5.0:
        # Extract current best as dumbbell if possible
        centroid = best_pts.mean(axis=0)
        angles = np.arctan2(best_pts[:, 1] - centroid[1], best_pts[:, 0] - centroid[0])
        sorted_idx = np.argsort(angles)
        sorted_pts = best_pts[sorted_idx]
        
        centers_final = np.zeros((7, 2))
        for i in range(7):
            centers_final[i] = (sorted_pts[2*i] + sorted_pts[2*i+1]) / 2.0
        v_final = (sorted_pts[1] - sorted_pts[0]) / 2.0
        params_final = np.concatenate([centers_final.flatten(), v_final])
        
        bounds_tight = []
        for i in range(7):
            cx, cy = centers_final[i]
            bounds_tight.append((cx - 0.5, cx + 0.5))
            bounds_tight.append((cy - 0.5, cy + 0.5))
        bounds_tight.append((v_final[0] - 0.2, v_final[0] + 0.2))
        bounds_tight.append((v_final[1] - 0.2, v_final[1] + 0.2))
        
        try:
            res = scipy.optimize.minimize(
                _neg_dumbbell_score, params_final, method='L-BFGS-B',
                bounds=bounds_tight, options={'maxiter': 8000, 'ftol': 1e-14, 'maxfev': 20000}
            )
            pts = _dumbbell_points(res.x[:14].reshape(7, 2), res.x[14:16])
            try_update(pts)
        except Exception:
            pass
    
    return _normalize(best_pts)
