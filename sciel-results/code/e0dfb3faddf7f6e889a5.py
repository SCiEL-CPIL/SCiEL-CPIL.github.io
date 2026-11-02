"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution, linprog
import time
import itertools

def _all_triangle_areas(pts):
    """Vectorized areas of all C(n,3) triangles."""
    n = len(pts)
    parts = []
    for i in range(n):
        for j in range(i + 1, n):
            v1 = pts[j] - pts[i]
            v2 = pts[j + 1:] - pts[i]
            if len(v2) == 0:
                continue
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            parts.append(0.5 * cross)
    return np.concatenate(parts)

def _min_triangle_area_fast(pts):
    """Compute only the minimum triangle area, with early stopping."""
    n = len(pts)
    best = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            v1 = pts[j] - pts[i]
            v2 = pts[j + 1:] - pts[i]
            if len(v2) == 0:
                continue
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas = 0.5 * cross
            a = np.min(areas)
            if a < best:
                best = a
                if best <= 1e-15:
                    return best
    return best

def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _convex_hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    min_area = _min_triangle_area_fast(pts)
    return min_area / hull_a

def _neg_score(params):
    pts = params.reshape(-1, 2)
    return -_score(pts)

def _soft_min_triangle_area(pts, alpha=50.0):
    """
    Smooth approximation of the minimum triangle area.
    Uses -log(sum(exp(-alpha * areas))) / alpha
    This is differentiable and guides optimizers better.
    """
    n = len(pts)
    if n < 3:
        return 0.0
    log_sum = -np.inf
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas = 0.5 * cross
            if len(areas) > 0:
                min_a = np.min(areas)
                if min_a > 0:
                    shifted = -alpha * (areas - min_a)
                    log_sum_partial = np.log(np.sum(np.exp(shifted))) - alpha * min_a
                    if log_sum_partial > log_sum:
                        log_sum = log_sum_partial
    if log_sum == -np.inf:
        return 0.0
    return -log_sum / alpha

def _hull_area_safe(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 1.0

def _smooth_score(pts):
    """Smooth score surrogate."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area_safe(pts)
    if hull_a <= 1e-12:
        return 0.0
    return _soft_min_triangle_area(pts, alpha=50.0) / hull_a

def _neg_smooth_score(params):
    pts = params.reshape(-1, 2)
    return -_smooth_score(pts)

def _fourier_points(coeffs):
    """
    Construct N points from low-frequency Fourier coefficients.

    coeffs layout:
      0: center_x
      1: center_y
      2: global radius scale
      3: angular phase
      4..: real/imag coefficients for harmonics 1..K
    """
    K = 3
    c = np.asarray(coeffs, dtype=float)
    cx, cy = c[0], c[1]
    r0 = c[2]
    phase = c[3]
    t = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    angles = t + phase

    radius = np.ones(N)
    angle_mod = np.ones(N)
    coeff_ptr = 4
    for k in range(1, K + 1):
        if coeff_ptr + 2 > len(c):
            break
        a = c[coeff_ptr]
        b = c[coeff_ptr + 1]
        radius += a * np.cos(k * t) + b * np.sin(k * t)
        angle_mod += 0.1 * (a * np.sin(k * t) - b * np.cos(k * t))
        coeff_ptr += 2

    radius = np.clip(radius, 0.05, 2.0)
    x = cx + r0 * radius * np.cos(angles * angle_mod)
    y = cy + r0 * radius * np.sin(angles * angle_mod)
    return np.column_stack([x, y])

def _fourier_bounds():
    K = 3
    b = [(-1.0, 1.0), (-1.0, 1.0), (0.2, 1.6), (-np.pi, np.pi)]
    for _ in range(K):
        b.append((-0.35, 0.35))
        b.append((-0.35, 0.35))
    return b

def _fourier_score(params):
    pts = _fourier_points(params)
    return -_score(pts)

def _generate_initializations():
    """Generate diverse starting points."""
    inits = []
    rng = np.random.RandomState(42)
    
    # 1. Regular Polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Regular Polygon + Center (if N is odd, N-1 on ring, 1 center)
    m = N - 1
    angles_m = np.linspace(0, 2 * np.pi, m, endpoint=False)
    outer = np.column_stack([np.cos(angles_m), np.sin(angles_m)])
    center = np.array([[0.0, 0.0]])
    inits.append(np.vstack([outer, center]))
    
    # 3. Concentric Rings
    # Try splitting N into two rings
    for n1 in range(3, N-2):
        n2 = N - n1
        ang1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
        ang2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
        r1 = 0.5
        r2 = 1.0
        ring1 = np.column_stack([r1 * np.cos(ang1), r1 * np.sin(ang1)])
        ring2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
        inits.append(np.vstack([ring1, ring2]))
        
    # 4. Jittered Regular Polygon
    for jitter in [0.01, 0.05, 0.1]:
        r = 1.0 + jitter * rng.randn(N)
        inits.append(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
        
    # 5. Random points in unit disk
    for _ in range(5):
        r = rng.rand(N) ** 0.5
        theta = rng.rand(N) * 2 * np.pi
        inits.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
        
    # 6. Grid-like
    xs = np.linspace(-0.5, 0.5, 4)
    ys = np.linspace(-0.5, 0.5, 4)
    xx, yy = np.meshgrid(xs, ys)
    pts_grid = np.column_stack([xx.flatten(), yy.flatten()])
    inits.append(pts_grid[:13])

    # 7. Fourier-parametrized initializations
    fourier_bounds = _fourier_bounds()
    # Symmetric circle-like starting coefficients
    base = np.zeros(len(fourier_bounds))
    base[2] = 1.0
    inits.append(_fourier_points(base))

    for seed in [1, 2, 3, 4, 5, 6]:
        rngf = np.random.RandomState(seed)
        params = np.array([rngf.uniform(lo, hi) for lo, hi in fourier_bounds])
        inits.append(_fourier_points(params))

    return inits

def _lp_refine(pts, deadline):
    """
    Refine using a linear programming approach on the log-scale of areas.
    This treats the problem as maximizing the minimum log-area, which is a convex problem
    in the log-domain if we fix the hull, but here we use it as a local polishing step
    by solving a sequence of LPs or using the LP structure to guide Nelder-Mead.
    
    Actually, a direct LP for this non-linear geometric problem is hard.
    Instead, we use a 'soft-min' guided Nelder-Mead with very tight tolerances,
    followed by a coordinate-descent-like step on the most critical triangles.
    """
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    # High-precision Nelder-Mead on the exact objective
    try:
        res = minimize(
            _neg_score,
            current_pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-16, 'maxfev': 200000, 'adaptive': True}
        )
        exact_pts = res.x.reshape(-1, 2)
        exact_score = _score(exact_pts)
        if exact_score > current_score:
            current_pts = exact_pts
            current_score = exact_score
    except Exception:
        pass
    
    # Coordinate descent on the critical triangle
    # Identify the triangle with the minimum area
    if time.time() < deadline:
        areas = _all_triangle_areas(current_pts)
        min_idx = np.argmin(areas)
        
        # Find the indices i, j, k corresponding to min_idx
        # This is a bit slow to reverse map, so we just try perturbing points slightly
        # towards the centroid of the critical triangle to increase its area,
        # while checking if other areas decrease.
        
        # A more robust local search: random jitter + accept if score improves
        best_local = current_pts.copy()
        best_local_score = current_score
        rng = np.random.RandomState(123)
        
        for _ in range(200):
            if time.time() > deadline:
                break
            # Random perturbation
            step_size = 1e-4
            perturbed = best_local + step_size * rng.randn(N, 2)
            s = _score(perturbed)
            if s > best_local_score:
                best_local = perturbed
                best_local_score = s
        
        if best_local_score > current_score:
            current_pts = best_local
            current_score = best_local_score
            
    return current_pts, current_score

def _refine_local(pts, deadline):
    """
    Perform local refinement using a multi-stage approach:
    1. Smooth optimization to navigate the landscape.
    2. High-precision exact optimization.
    3. Local search / coordinate descent polish.
    """
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    # Stage 1: Smooth refinement
    try:
        res_smooth = minimize(
            _neg_smooth_score,
            current_pts.flatten(),
            method='Powell',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 30000}
        )
        smooth_pts = res_smooth.x.reshape(-1, 2)
        smooth_score = _score(smooth_pts)
        if smooth_score > current_score:
            current_pts = smooth_pts
            current_score = smooth_score
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return current_pts, current_score

    # Stage 2: Exact refinement with high precision
    current_pts, current_score = _lp_refine(current_pts, deadline)
        
    return current_pts, current_score

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We stop at 140s to ensure return.
    deadline = start_time + 140.0

    best_score = 0.0
    best_pts = np.zeros((N, 2))

    # 1. Evaluate initializations
    inits = _generate_initializations()
    for init_pts in inits:
        if time.time() > deadline - 40.0:
            break
        s = _score(init_pts)
        if s > best_score:
            best_score = s
            best_pts = init_pts.copy()

    # 2. Fourier-coefficient global optimization
    # The Fourier parametrization enforces a smooth cyclic ordering and reduces
    # pathological self-intersections, making differential evolution more effective
    # in the lower-dimensional coefficient space.
    fourier_bounds = _fourier_bounds()
    remaining = deadline - time.time()
    if remaining > 25.0:
        try:
            # Use a time budget for DE; leave room for local refinement.
            de_time_budget = min(remaining - 10.0, 70.0)
            popsize = 12
            maxiter = max(10, int(de_time_budget / 1.2))

            res_de = differential_evolution(
                _fourier_score,
                fourier_bounds,
                seed=42,
                maxiter=maxiter,
                popsize=popsize,
                tol=1e-8,
                init='latinhypercube',
                polish=False,
                update_interval=1,
                callback=lambda x, convergence: (time.time() > deadline - 25.0)
            )
            de_pts = _fourier_points(res_de.x)
            de_score = _score(de_pts)
            if de_score > best_score:
                best_score = de_score
                best_pts = de_pts.copy()
        except Exception:
            pass

    # 3. Optimization Loop
    # We will try multiple restarts from the best few initializations and the current best
    candidates = []
    scored_inits = [(_score(p), p) for p in inits]
    scored_inits.sort(key=lambda x: -x[0])
    for s, p in scored_inits[:5]:
        candidates.append(p)
    candidates.append(best_pts)

    unique_candidates = []
    for c in candidates:
        if not any(np.allclose(c, u, atol=1e-5) for u in unique_candidates):
            unique_candidates.append(c)

    # Refine each candidate
    for init_pts in unique_candidates:
        if time.time() > deadline - 20.0:
            break

        refined_pts, refined_score = _refine_local(init_pts, deadline - 5.0)

        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined_pts.copy()

    # 4. Differential Evolution for global exploration if time permits
    if time.time() < deadline - 50.0:
        try:
            bounds = [(-1.5, 1.5)] * (2 * N)
            res_de = differential_evolution(
                _neg_score,
                bounds,
                seed=42,
                maxiter=30,
                popsize=15,
                tol=1e-8,
                init='latinhypercube',
                polish=False
            )
            de_pts = res_de.x.reshape(-1, 2)

            polished_pts, polished_score = _refine_local(de_pts, deadline - 5.0)

            if polished_score > best_score:
                best_score = polished_score
                best_pts = polished_pts.copy()
        except Exception:
            pass

    # Final polish on the global best
    if time.time() < deadline - 2.0:
        final_pts, final_score = _lp_refine(best_pts, deadline - 0.5)
        if final_score > best_score:
            best_pts = final_pts
            best_score = final_score

    return best_pts
# EVOLVE-BLOCK-END
