import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import sys

N = 13

# Precompute all triangle index triples
_TRIPLE_IDX = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _TRIPLE_IDX.append((i, j, k))
_TRIPLE_IDX = np.array(_TRIPLE_IDX, dtype=np.int32)
_NUM_TRIPLES = len(_TRIPLE_IDX)


def _triangle_areas_vectorized(points):
    """Compute all triangle areas using precomputed indices."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    """Compute convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute the score: min triangle area / hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _build_config(template_idx, params):
    """
    Build a point configuration from a template and parameters.
    Templates:
      0: 13-gon (all on hull)
      1: 12-1 (12 on hull, 1 center)
      2: 11-2 (11 on hull, 2 inner)
      3: 10-3 (10 on hull, 3 inner)
      4: 9-4 (9 on hull, 4 inner)
      5: 8-5 (8 on hull, 5 inner)
      6: 7-6 (7 on hull, 6 inner)
      7: 6-7 (6 on hull, 7 inner)
      8: 5-4-4 (5 hull, 4 mid, 4 inner)
    """
    if template_idx == 0:
        # 13-gon
        r = params[0]
        a0 = params[1]
        angles = np.linspace(a0, 2 * np.pi + a0, N, endpoint=False)
        return np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    elif template_idx == 1:
        # 12-1
        r_out, a_out, r_in, a_in = params[:4]
        outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 12, endpoint=False)
        outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
        inner_pts = np.array([[r_in * np.cos(a_in), r_in * np.sin(a_in)]])
        return np.vstack([outer_pts, inner_pts])
    elif template_idx == 2:
        # 11-2
        r_out, a_out, r_in, a_in = params[:4]
        outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 11, endpoint=False)
        outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
        inner_angles = np.linspace(a_in, 2 * np.pi + a_in, 2, endpoint=False)
        inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
        return np.vstack([outer_pts, inner_pts])
    elif template_idx == 3:
        # 10-3
        r_out, a_out, r_in, a_in = params[:4]
        outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 10, endpoint=False)
        outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
        inner_angles = np.linspace(a_in, 2 * np.pi + a_in, 3, endpoint=False)
        inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
        return np.vstack([outer_pts, inner_pts])
    elif template_idx == 4:
        # 9-4
        r_out, a_out, r_in, a_in = params[:4]
        outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 9, endpoint=False)
        outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
        inner_angles = np.linspace(a_in, 2 * np.pi + a_in, 4, endpoint=False)
        inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
        return np.vstack([outer_pts, inner_pts])
    elif template_idx == 5:
        # 8-5
        r_out, a_out, r_in, a_in = params[:4]
        outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 8, endpoint=False)
        outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
        inner_angles = np.linspace(a_in, 2 * np.pi + a_in, 5, endpoint=False)
        inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
        return np.vstack([outer_pts, inner_pts])
    elif template_idx == 6:
        # 7-6
        r_out, a_out, r_in, a_in = params[:4]
        outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 7, endpoint=False)
        outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
        inner_angles = np.linspace(a_in, 2 * np.pi + a_in, 6, endpoint=False)
        inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
        return np.vstack([outer_pts, inner_pts])
    elif template_idx == 7:
        # 6-7
        r_out, a_out, r_in, a_in = params[:4]
        outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 6, endpoint=False)
        outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
        inner_angles = np.linspace(a_in, 2 * np.pi + a_in, 7, endpoint=False)
        inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
        return np.vstack([outer_pts, inner_pts])
    elif template_idx == 8:
        # 5-4-4
        r_o, a_o, r_m, a_m, r_i, a_i = params[:6]
        o_angles = np.linspace(a_o, 2 * np.pi + a_o, 5, endpoint=False)
        m_angles = np.linspace(a_m, 2 * np.pi + a_m, 4, endpoint=False)
        i_angles = np.linspace(a_i, 2 * np.pi + a_i, 4, endpoint=False)
        o_pts = np.column_stack([r_o * np.cos(o_angles), r_o * np.sin(o_angles)])
        m_pts = np.column_stack([r_m * np.cos(m_angles), r_m * np.sin(m_angles)])
        i_pts = np.column_stack([r_i * np.cos(i_angles), r_i * np.sin(i_angles)])
        return np.vstack([o_pts, m_pts, i_pts])
    else:
        raise ValueError(f"Unknown template {template_idx}")


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin against 120s limit
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        if len(points) != N:
            return
        if not np.all(np.isfinite(points)):
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # Phase 1: Initial Guesses from Templates
    # Template 0: 13-gon
    _update_best(_regular_polygon(N, 1.0))
    
    # Template 1: 12-1
    r_out, a_out, r_in, a_in = 1.0, 0.0, 0.3, 0.0
    _update_best(_build_config(1, [r_out, a_out, r_in, a_in]))
    
    # Template 2: 11-2
    r_out, a_out, r_in, a_in = 1.0, 0.0, 0.4, 0.0
    _update_best(_build_config(2, [r_out, a_out, r_in, a_in]))
    
    # Template 3: 10-3
    r_out, a_out, r_in, a_in = 1.0, 0.0, 0.5, 0.0
    _update_best(_build_config(3, [r_out, a_out, r_in, a_in]))
    
    # Template 4: 9-4
    r_out, a_out, r_in, a_in = 1.0, 0.0, 0.5, 0.0
    _update_best(_build_config(4, [r_out, a_out, r_in, a_in]))
    
    # Template 5: 8-5
    r_out, a_out, r_in, a_in = 1.0, 0.0, 0.6, 0.0
    _update_best(_build_config(5, [r_out, a_out, r_in, a_in]))
    
    # Template 6: 7-6
    r_out, a_out, r_in, a_in = 1.0, 0.0, 0.6, 0.0
    _update_best(_build_config(6, [r_out, a_out, r_in, a_in]))
    
    # Template 7: 6-7
    r_out, a_out, r_in, a_in = 1.0, 0.0, 0.7, 0.0
    _update_best(_build_config(7, [r_out, a_out, r_in, a_in]))
    
    # Template 8: 5-4-4
    r_o, a_o, r_m, a_m, r_i, a_i = 1.0, 0.0, 0.6, 0.0, 0.3, 0.0
    _update_best(_build_config(8, [r_o, a_o, r_m, a_m, r_i, a_i]))
    
    # Random points
    rng = np.random.default_rng(42)
    for _ in range(10):
        pts = rng.uniform(-1, 1, (N, 2))
        _update_best(pts)

    # Phase 2: Optimize each template's parameters using Nelder-Mead
    # This is a structural change: instead of optimizing 26 raw coordinates,
    # we optimize 4-6 parameters per template, which is a much lower-dimensional
    # and smoother problem.
    
    def _optimize_template(template_idx, n_params):
        def objective(params):
            try:
                pts = _build_config(template_idx, params)
                if not np.all(np.isfinite(pts)):
                    return 1e10
                s = _score(pts)
                if s <= 0:
                    return 1e10
                return -s
            except Exception:
                return 1e10
        
        # Initial guess
        if template_idx == 0:
            x0 = np.array([1.0, 0.0])
        elif template_idx == 8:
            x0 = np.array([1.0, 0.0, 0.6, 0.0, 0.3, 0.0])
        else:
            x0 = np.array([1.0, 0.0, 0.5, 0.0])
        
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            opt_params = result.x
            opt_points = _build_config(template_idx, opt_params)
            _update_best(opt_points)
        except Exception:
            pass

    # Optimize templates 0-8
    for t in range(9):
        if time.time() > deadline - 10:
            break
        n_params = 2 if t == 0 else (6 if t == 8 else 4)
        _optimize_template(t, n_params)

    # Phase 3: Differential Evolution on full 26-dim space
    # Use the best template result as a good starting point for DE
    if best_points is not None and time.time() < deadline - 20:
        bounds = [(-1.5, 1.5)] * (N * 2)
        
        def _negative_score(x_flat):
            points = x_flat.reshape(N, 2)
            if not np.all(np.isfinite(points)):
                return 1e10
            s = _score(points)
            if s <= 0:
                return 1e10
            return -s
        
        try:
            result = differential_evolution(
                _negative_score,
                bounds=bounds,
                seed=42,
                maxiter=30,
                popsize=15,
                tol=1e-4,
                init='sobol'
            )
            opt_points = result.x.reshape(N, 2)
            _update_best(opt_points)
        except Exception:
            pass

    # Phase 4: Local Polish with Nelder-Mead on full space
    if best_points is not None and time.time() < deadline - 5:
        x0 = best_points.flatten()
        
        def _negative_score(x_flat):
            points = x_flat.reshape(N, 2)
            if not np.all(np.isfinite(points)):
                return 1e10
            s = _score(points)
            if s <= 0:
                return 1e10
            return -s
        
        try:
            result = minimize(
                _negative_score,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            opt_points = result.x.reshape(N, 2)
            _update_best(opt_points)
        except Exception:
            pass

    # Phase 5: Additional Random Restarts with DE if time permits
    if time.time() < deadline - 15:
        for seed in [123, 456]:
            if time.time() > deadline - 5:
                break
            try:
                bounds = [(-1.5, 1.5)] * (N * 2)
                result = differential_evolution(
                    _negative_score,
                    bounds=bounds,
                    seed=seed,
                    maxiter=20,
                    popsize=12,
                    tol=1e-3,
                    init='latinhypercube'
                )
                opt_points = result.x.reshape(N, 2)
                _update_best(opt_points)
                
                # Quick polish
                if time.time() < deadline - 3:
                    result2 = minimize(
                        _negative_score,
                        opt_points.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9}
                    )
                    _update_best(result2.x.reshape(N, 2))
            except Exception:
                pass

    # Final check
    if best_points is None:
        best_points = _regular_polygon(N, 1.0)
    
    return best_points
