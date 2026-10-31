"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations
import time
import warnings
warnings.filterwarnings('ignore')

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
    """Compute all triangle areas using vectorized operations."""
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
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        cx, cy = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
        sp = points[np.argsort(angles)]
        x, y = sp[:, 0], sp[:, 1]
        return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _score(points):
    """Score = min triangle area / convex hull area."""
    if len(points) < 3:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    hull_area = _hull_area(points)
    if hull_area <= 1e-15:
        return 0.0
    return min_area / hull_area


def _half_plane_intersection(normals, offsets):
    """
    Compute the convex polygon as intersection of half-planes.
    Each half-plane: dot(normal_i, x) <= offsets[i]
    Returns vertices of the resulting convex polygon (or None if empty).
    """
    n = len(normals)
    if n < 3:
        return None
    
    # Use scipy's half-space intersection via linear programming approach
    # Or implement Sutherland-Hodgman style clipping
    
    # Start with a large bounding box
    # Use a grid of candidate points and filter, then find convex hull
    # Better: use linear programming to find vertices
    
    # Alternative: use the fact that vertices are intersections of pairs of lines
    # that are "extreme"
    
    # For N half-planes, the polygon has at most N vertices
    # Each vertex is the intersection of two boundary lines
    # We need to find which pairs form actual vertices
    
    # Collect all pairwise intersections
    vertices = []
    for i in range(n):
        for j in range(i + 1, n):
            # Solve: normals[i] . x = offsets[i]
            #        normals[j] . x = offsets[j]
            A = np.array([normals[i], normals[j]])
            b = np.array([offsets[i], offsets[j]])
            det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
            if abs(det) < 1e-12:
                continue  # Parallel lines
            x = (b[0] * A[1, 1] - b[1] * A[0, 1]) / det
            y = (A[0, 0] * b[1] - A[1, 0] * b[0]) / det
            pt = np.array([x, y])
            
            # Check if this point satisfies all half-planes
            valid = True
            for k in range(n):
                if np.dot(normals[k], pt) > offsets[k] + 1e-10:
                    valid = False
                    break
            if valid:
                vertices.append(pt)
    
    if len(vertices) < 3:
        return None
    
    vertices = np.array(vertices)
    
    # Remove duplicates
    unique_verts = []
    for v in vertices:
        is_dup = False
        for uv in unique_verts:
            if np.linalg.norm(v - uv) < 1e-10:
                is_dup = True
                break
        if not is_dup:
            unique_verts.append(v)
    
    if len(unique_verts) < 3:
        return None
    
    unique_verts = np.array(unique_verts)
    
    # Compute convex hull to get the actual polygon vertices in order
    try:
        hull = ConvexHull(unique_verts)
        ordered = unique_verts[hull.vertices]
        return ordered
    except Exception:
        # Fallback: sort by angle
        cx, cy = np.mean(unique_verts, axis=0)
        angles = np.arctan2(unique_verts[:, 1] - cy, unique_verts[:, 0] - cx)
        return unique_verts[np.argsort(angles)]


def _support_line_objective(params):
    """
    Objective for support-line representation.
    params: [angles (N), offsets (N)] -> 2N parameters
    angles: normal directions of the N support lines
    offsets: distances from origin along each normal
    """
    angles = params[:N]
    offsets = params[N:]
    
    # Ensure offsets are positive (polygon contains origin)
    if np.any(offsets < 0.01):
        return 1e6
    
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    polygon = _half_plane_intersection(normals, offsets)
    if polygon is None or len(polygon) < N:
        # Penalize: we want exactly N vertices for best score
        if polygon is not None:
            # Use all available vertices
            pts = polygon
        else:
            return 1e6
    else:
        pts = polygon
    
    if len(pts) < 3:
        return 1e6
    
    # If we have more than N points, we need to select N
    # If fewer, pad or penalize
    if len(pts) >= N:
        # Select N points that maximize min triangle area
        # For simplicity, use all if exactly N, or subsample
        if len(pts) == N:
            selected = pts
        else:
            # Use convex hull vertices, take first N or evenly spaced
            idx = np.linspace(0, len(pts) - 1, N, dtype=int)
            selected = pts[idx]
    else:
        # Too few vertices - penalize
        return 1e6 + (N - len(pts))
    
    score = _score(selected)
    return -score


def _polar_to_cartesian(angles, radii):
    """Convert polar to cartesian."""
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _smooth_min(areas, temperature):
    """Soft minimum using log-sum-exp."""
    neg_areas = -areas
    max_neg = np.max(neg_areas)
    if max_neg > 500:
        return -max_neg
    shifted = neg_areas - max_neg
    log_sum = np.log(np.sum(np.exp(shifted))) + max_neg
    return -log_sum * temperature


def _smooth_objective(params, temperature=1.0):
    """Smooth objective using soft-min for triangle areas."""
    points = params.reshape(-1, 2)
    
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    if np.any(areas <= 0):
        return 1e10
    
    # Soft min of areas
    soft_min_area = _smooth_min(areas, temperature)
    
    hull_area = _hull_area(points)
    if hull_area <= 1e-15:
        return 1e10
    
    return -soft_min_area / hull_area


def _make_ring_config(radius_inner, radius_outer, n_inner, n_outer):
    """Create a ring configuration with inner and outer points."""
    angles_in = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
    angles_out = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    
    pts_in = np.column_stack([radius_inner * np.cos(angles_in), 
                               radius_inner * np.sin(angles_in)])
    pts_out = np.column_stack([radius_outer * np.cos(angles_out), 
                                radius_outer * np.sin(angles_out)])
    
    return np.vstack([pts_in, pts_out])


def _coordinate_descent_polish(points, max_iter=200, tol=1e-10):
    """Coordinate descent to polish a point configuration."""
    best = points.copy()
    best_score = _score(best)
    
    for it in range(max_iter):
        improved = False
        for i in range(N):
            # Try small perturbations in x and y
            for dx in [-0.01, -0.005, 0, 0.005, 0.01]:
                for dy in [-0.01, -0.005, 0, 0.005, 0.01]:
                    trial = best.copy()
                    trial[i, 0] += dx
                    trial[i, 1] += dy
                    s = _score(trial)
                    if s > best_score + tol:
                        best = trial
                        best_score = s
                        improved = True
        if not improved:
            break
    
    return best, best_score


def _local_polish(points, max_iter=5000):
    """Local refinement using Nelder-Mead."""
    try:
        result = minimize(
            lambda p: -_score(p.reshape(-1, 2)),
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        if result.fun < 0:
            return result.x.reshape(-1, 2), -result.fun
    except Exception:
        pass
    return points, _score(points)


def construct():
    """Construct 13 points to maximize minimum triangle area / convex hull area."""
    deadline = time.time() + 140  # Safety margin before 150s limit
    best_score = 0.0
    best_points = None
    
    def try_update(points):
        nonlocal best_score, best_points
        if points is None or len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    # Phase 1: Regular 13-gon as initial guess
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    try_update(init_points)
    
    # Phase 2: Ring configurations (inner + outer)
    if time.time() < deadline - 60:
        for r_in, r_out, n_in, n_out in [
            (0.3, 1.0, 5, 8),
            (0.4, 1.0, 6, 7),
            (0.5, 1.0, 5, 8),
            (0.4, 0.9, 6, 7),
            (0.3, 0.95, 5, 8),
        ]:
            if time.time() > deadline - 60:
                break
            ring_pts = _make_ring_config(r_in, r_out, n_in, n_out)
            # Polish
            polished, s = _local_polish(ring_pts, max_iter=3000)
            try_update(polished)
    
    # Phase 3: Differential evolution with smooth objective
    if time.time() < deadline - 40:
        bounds = [(-1.5, 1.5)] * (2 * N)
        try:
            de_result = differential_evolution(
                lambda p: _smooth_objective(p, temperature=0.5),
                bounds,
                maxiter=100,
                popsize=12,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.8,
                seed=42,
                polish=False,
                init='sobol'
            )
            if de_result.fun < 0:
                de_pts = de_result.x.reshape(-1, 2)
                # Polish with hard objective
                polished, s = _local_polish(de_pts, max_iter=5000)
                try_update(polished)
        except Exception:
            pass
    
    # Phase 4: Support-line representation
    if time.time() < deadline - 30:
        # Try various angle configurations
        for angle_seed in [0, 1, 2]:
            if time.time() > deadline - 30:
                break
            rng = np.random.RandomState(angle_seed)
            # Start with roughly evenly spaced angles
            base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            noise = rng.normal(0, 0.2, N)
            angles_init = base_angles + noise
            offsets_init = np.ones(N) * 1.0
            
            params_init = np.concatenate([angles_init, offsets_init])
            
            bounds_sl = [(-np.pi, np.pi)] * N + [(0.1, 2.0)] * N
            
            try:
                sl_result = minimize(
                    _support_line_objective,
                    params_init,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if sl_result.fun < 0:
                    angles_opt = sl_result.x[:N]
                    offsets_opt = sl_result.x[N:]
                    normals = np.column_stack([np.cos(angles_opt), np.sin(angles_opt)])
                    polygon = _half_plane_intersection(normals, offsets_opt)
                    if polygon is not None and len(polygon) >= 3:
                        if len(polygon) == N:
                            try_update(polygon)
                        else:
                            # Subsample or use as-is
                            if len(polygon) > N:
                                idx = np.linspace(0, len(polygon) - 1, N, dtype=int)
                                try_update(polygon[idx])
                            else:
                                try_update(polygon)
            except Exception:
                pass
    
    # Phase 5: Final coordinate descent polish
    if best_points is not None and time.time() < deadline - 10:
        polished, s = _coordinate_descent_polish(best_points, max_iter=300)
        try_update(polished)
    
    # Phase 6: One more Nelder-Mead pass
    if best_points is not None and time.time() < deadline - 5:
        polished, s = _local_polish(best_points, max_iter=3000)
        try_update(polished)
    
    if best_points is None:
        best_points = init_points
    
    return best_points
# EVOLVE-BLOCK-END
