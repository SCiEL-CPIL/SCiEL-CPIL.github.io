import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

N = 14
START_TIME = time.time()
DEADLINE = START_TIME + 115.0

def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

TRIPLE_INDICES = _get_triple_indices(N)
I_IDX = TRIPLE_INDICES[:, 0]
J_IDX = TRIPLE_INDICES[:, 1]
K_IDX = TRIPLE_INDICES[:, 2]
NUM_TRIPLES = len(TRIPLE_INDICES)

def _score(points):
    """Compute min triangle area / convex hull area."""
    if len(points) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    p_i = points[I_IDX]
    p_j = points[J_IDX]
    p_k = points[K_IDX]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _neg_score(flat_points):
    points = flat_points.reshape((N, 2))
    return -_score(points)

def _make_regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_ring_config(rings):
    """
    Generate points from ring configuration.
    rings: list of (count, radius, angle_offset)
    """
    pts = []
    for count, radius, offset in rings:
        angles = np.linspace(offset, 2*np.pi + offset, count, endpoint=False)
        pts.append(np.column_stack([radius * np.cos(angles), radius * np.sin(angles)]))
    return np.vstack(pts)

def _superellipse_points(n_points, exponent, aspect_ratio, offset_angle=0.0):
    """
    Generate points on a superellipse |x/a|^n + |y/b|^n = 1
    by uniform arc length parameterization.
    
    Parameters:
    -----------
    n_points : int
        Number of points to generate
    exponent : float
        The Lamé exponent (n in the equation). 
        n=2 is a circle/ellipse, n->inf approaches a rectangle.
    aspect_ratio : float
        Ratio a/b. a is semi-axis in x, b in y.
    offset_angle : float
        Rotation of the entire configuration.
    
    Returns:
    --------
    points : array of shape (n_points, 2)
    """
    a = 1.0
    b = 1.0 / aspect_ratio
    
    # Sample parameter t uniformly in [0, 2*pi]
    # This is not uniform arc length, but it's a good starting point.
    # For a true uniform arc length, we'd need to integrate ds/dt.
    # However, for the purpose of this optimization, the regular polygon
    # in parameter space is often a good local optimum or close to it.
    # Let's try to be smarter: use a larger set of samples and resample?
    # No, let's stick to uniform t for now, as the optimizer will adjust
    # the global shape if needed, but the "uniformity" is fixed by the representation.
    # Actually, the directive says "distribute points uniformly by arc length".
    # Let's implement a simple arc-length resampling.
    
    # High resolution sampling to compute arc length
    t_high_res = np.linspace(0, 2*np.pi, 1000, endpoint=False)
    
    # x = a * sign(cos(t)) * |cos(t)|^(2/n)
    # y = b * sign(sin(t)) * |sin(t)|^(2/n)
    # Derivatives:
    # dx/dt = -a * (2/n) * |cos(t)|^(2/n - 2) * sin(t)
    # dy/dt = b * (2/n) * |sin(t)|^(2/n - 2) * cos(t)
    
    # Handle singularity when n < 2 (though we likely want n >= 2 for convexity)
    # For n >= 2, the derivatives are well-behaved at the axes (0 or finite).
    # At t=0, cos=1, sin=0. dx/dt = 0. dy/dt = b * (2/n) * 0^(2/n-2) * 1.
    # If n > 2, 2/n - 2 < 0, so 0^negative is infinity.
    # Wait, the formula for derivative of |u|^k is k|u|^(k-1)sign(u)u'.
    # Let's just compute numerically or use a robust formula.
    
    # Actually, for n >= 2, the curve is convex.
    # Let's compute points first.
    cos_t = np.cos(t_high_res)
    sin_t = np.sin(t_high_res)
    
    # Avoid 0^negative issues by clipping
    eps = 1e-10
    abs_cos = np.maximum(np.abs(cos_t), eps)
    abs_sin = np.maximum(np.abs(sin_t), eps)
    
    x_hr = a * np.sign(cos_t) * abs_cos ** (2.0 / exponent)
    y_hr = b * np.sign(sin_t) * abs_sin ** (2.0 / exponent)
    
    # Compute arc length element ds = sqrt(dx^2 + dy^2)
    # Use numerical differentiation for stability
    dx = np.gradient(x_hr, t_high_res)
    dy = np.gradient(y_hr, t_high_res)
    ds = np.sqrt(dx**2 + dy**2)
    
    # Cumulative arc length
    s = np.cumsum(ds)
    s = np.concatenate(([0], s)) # s[0]=0
    total_length = s[-1] + ds[-1] # approximate total
    
    # We want n_points equally spaced in arc length
    target_lengths = np.linspace(0, total_length, n_points, endpoint=False)
    
    # Interpolate t values for these target lengths
    # s is monotonically increasing
    t_points = np.interp(target_lengths, s, t_high_res)
    
    # If the last point wraps around, ensure it's handled correctly
    # np.interp clamps to the ends. Since s goes 0 to ~total, and target is < total, it's fine.
    # But s is defined on [0, 2pi]. The cumulative sum s[i] is length from 0 to t[i].
    # The last interval is from t[-1] to 2pi.
    # Let's just use the interpolated t values.
    
    cos_p = np.cos(t_points)
    sin_p = np.sin(t_points)
    
    abs_cos_p = np.maximum(np.abs(cos_p), eps)
    abs_sin_p = np.maximum(np.abs(sin_p), eps)
    
    x = a * np.sign(cos_p) * abs_cos_p ** (2.0 / exponent)
    y = b * np.sign(sin_p) * abs_sin_p ** (2.0 / exponent)
    
    points = np.column_stack([x, y])
    
    # Rotate by offset_angle
    if offset_angle != 0:
        c = np.cos(offset_angle)
        s_rot = np.sin(offset_angle)
        rot = np.array([[c, -s_rot], [s_rot, c]])
        points = points @ rot.T
        
    return points

def _neg_score_superellipse(params):
    exponent, aspect_ratio, offset_angle = params
    # Bounds: exponent [1.5, 10], aspect_ratio [0.5, 2.0], offset [0, 2*pi]
    # Note: aspect_ratio > 0
    try:
        pts = _superellipse_points(N, exponent, aspect_ratio, offset_angle)
        return -_score(pts)
    except Exception:
        return 0.0

def construct():
    global START_TIME, DEADLINE
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0
    
    np.random.seed(42)
    
    candidates = []
    
    # 1. Regular 14-gon (Superellipse with n=2, ar=1)
    candidates.append(_make_regular_polygon(N))
    
    # 2. Superellipse candidates
    # Explore different exponents and aspect ratios
    exponents = [2.0, 3.0, 4.0, 6.0, 10.0]
    aspect_ratios = [0.7, 0.8, 1.0, 1.2, 1.5]
    offsets = [0.0, np.pi/14, np.pi/7]
    
    for n_exp in exponents:
        for ar in aspect_ratios:
            for off in offsets:
                try:
                    pts = _superellipse_points(N, n_exp, ar, off)
                    candidates.append(pts)
                except Exception:
                    pass
    
    # 3. Concentric Rings (keeping some from original for robustness)
    splits = [
        (7, 7),
        (6, 8),
        (8, 6),
    ]
    radius_ratios = [0.5, 0.7, 0.9]
    for split in splits:
        n1, n2 = split
        for r1 in radius_ratios:
            for offset in [0, np.pi/n1]:
                rings = [(n1, 1.0, 0), (n2, r1, offset)]
                candidates.append(_generate_ring_config(rings))
    
    # 4. Random Points
    for _ in range(10):
        candidates.append(np.random.randn(N, 2) * 0.8)
        
    # Evaluate all candidates
    scored_candidates = []
    for c in candidates:
        s = _score(c)
        scored_candidates.append((s, c))
        
    scored_candidates.sort(key=lambda x: -x[0])
    
    best_score = -1
    best_points = None
    
    # Optimize top candidates
    # We have ~115 seconds. 
    # Let's take top 5 general candidates and optimize them.
    # Also, run a specific optimization on the superellipse parameter space.
    
    top_k = 5
    for i in range(top_k):
        if time.time() > DEADLINE - 20:
            break
            
        s, c = scored_candidates[i]
        if s > best_score:
            best_score = s
            best_points = c.copy()
            
        # Optimize this candidate
        try:
            res = minimize(
                _neg_score,
                c.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'maxfev': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            opt_points = res.x.reshape((N, 2))
            opt_score = _score(opt_points)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
                
        except Exception:
            pass
            
    # Specific optimization on Superellipse parameters
    if time.time() < DEADLINE - 20:
        # Use multiple starting points for the superellipse optimizer
        start_params = [
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [4.0, 1.2, 0.0],
            [6.0, 1.0, 0.0],
            [2.0, 1.5, 0.0],
            [2.0, 0.8, 0.0],
        ]
        
        for sp in start_params:
            if time.time() > DEADLINE - 10:
                break
            try:
                res = minimize(
                    _neg_score_superellipse,
                    sp,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'maxfev': 400, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                opt_params = res.x
                opt_pts = _superellipse_points(N, opt_params[0], opt_params[1], opt_params[2])
                opt_score = _score(opt_pts)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_pts.copy()
                    
            except Exception:
                pass

    # Final polish on the best point
    if best_points is not None and time.time() < DEADLINE - 5:
        try:
            res = minimize(
                _neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'maxfev': 400, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            final_points = res.x.reshape((N, 2))
            final_score = _score(final_points)
            if final_score > best_score:
                best_score = final_score
                best_points = final_points.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = _make_regular_polygon(N)
        
    return best_points
