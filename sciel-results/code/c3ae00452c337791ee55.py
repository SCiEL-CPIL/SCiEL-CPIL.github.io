import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13

_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)


def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0], atol=1e-12):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _neg_score(points):
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _points_from_offsets(offsets, n_lines=N):
    """
    Given offsets for n_lines support lines with fixed normal angles,
    compute the vertices of the convex polygon.
    
    The support lines are: n_k · x = offset_k, where n_k = (cos(θ_k), sin(θ_k))
    with θ_k = 2πk/n_lines for k=0,...,n_lines-1.
    
    The polygon vertices are intersections of consecutive support lines.
    """
    n_lines = len(offsets)
    angles = 2 * np.pi * np.arange(n_lines) / n_lines
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Find vertices: intersection of line k and line (k+1) % n_lines
    # n_k · v = offset_k
    # n_{k+1} · v = offset_{k+1}
    
    vertices = np.zeros((n_lines, 2))
    for k in range(n_lines):
        k1 = (k + 1) % n_lines
        n0 = normals[k]
        n1 = normals[k1]
        o0 = offsets[k]
        o1 = offsets[k1]
        
        # Solve 2x2 system:
        # n0[0]*x + n0[1]*y = o0
        # n1[0]*x + n1[1]*y = o1
        det = n0[0] * n1[1] - n0[1] * n1[0]
        if abs(det) < 1e-12:
            vertices[k] = [0, 0]
            continue
        x = (o0 * n1[1] - o1 * n0[1]) / det
        y = (n0[0] * o1 - n1[0] * o0) / det
        vertices[k] = [x, y]
    
    return vertices


def _score_from_offsets(offsets):
    """Compute score directly from support line offsets."""
    pts = _points_from_offsets(offsets)
    return _score(pts)


def _offsets_from_points(points):
    """
    Given N points (assumed to be vertices of a convex polygon in CCW order),
    compute the support line offsets for N equally-spaced normal directions.
    
    For each normal direction, the offset is the maximum dot product with any point.
    """
    n_lines = len(points)
    angles = 2 * np.pi * np.arange(n_lines) / n_lines
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # For each normal, offset = max over all points of n·p
    dots = points @ normals.T  # shape (N, N)
    offsets = np.max(dots, axis=0)
    
    return offsets


def _sa_optimize(offsets, deadline, T0=1.0, T_end=0.01, n_iter=50000, 
                 step_scale=0.1, rng_seed=42):
    """
    Simulated annealing on support line offsets.
    """
    rng = np.random.RandomState(rng_seed)
    current_offsets = offsets.copy()
    current_score = _score_from_offsets(current_offsets)
    
    best_offsets = current_offsets.copy()
    best_score = current_score
    
    n_lines = len(offsets)
    
    for it in range(n_iter):
        if time.time() > deadline - 2:
            break
        
        # Temperature schedule (geometric cooling)
        alpha = T_end / T0
        T = T0 * (alpha ** (it / max(1, n_iter - 1)))
        
        # Propose a move: perturb one or two offsets
        n_perturb = rng.randint(1, 3)  # perturb 1 or 2 offsets
        idx = rng.randint(0, n_lines, size=n_perturb)
        delta = rng.normal(0, step_scale, size=n_perturb)
        
        new_offsets = current_offsets.copy()
        new_offsets[idx] += delta
        
        new_score = _score_from_offsets(new_offsets)
        
        # Accept if better, or with probability exp(-ΔE/T)
        if new_score >= current_score:
            current_offsets = new_offsets
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_offsets = current_offsets.copy()
        else:
            delta_E = current_score - new_score
            if delta_E < 1e-15:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta_E / max(T, 1e-15))
            if rng.random() < accept_prob:
                current_offsets = new_offsets
                current_score = new_score
    
    return best_offsets, best_score


def _seed_regular():
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _seed_quasi_regular(seed=0, ang_scale=0.05, rad_scale=0.05):
    rng = np.random.RandomState(seed)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    ang += rng.normal(0, ang_scale, N)
    rad = 1.0 + rng.normal(0, rad_scale, N)
    return np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])


def _seed_two_ring(n_out, r_inner, angle_offset):
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_in, endpoint=False)
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_inner * np.cos(a_in), r_inner * np.sin(a_in)])
    return np.vstack([outer, inner])


def _seed_random_convex(seed=0):
    """Generate a random convex polygon by sorting random angles and radii."""
    rng = np.random.RandomState(seed)
    angles = np.sort(rng.uniform(0, 2 * np.pi, N))
    radii = 1.0 + rng.uniform(-0.2, 0.2, N)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _polish_true_objective(points, deadline, max_iter=1000):
    """
    Final polish using the true non-smooth objective with Nelder-Mead.
    """
    best = points.copy()
    best_s = _score(best)
    
    if time.time() < deadline - 1:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    return best, best_s


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    np.random.seed(42)
    
    best_points = None
    best_score = 0.0
    
    def _update(pts):
        nonlocal best_points, best_score
        if pts is None or len(pts) != N:
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    # Phase 1: Simulated Annealing on support-line representation
    # Start with regular polygon offsets
    regular_pts = _seed_regular()
    regular_offsets = _offsets_from_points(regular_pts)
    
    # Run SA with multiple restarts and different parameters
    sa_configs = [
        {'T0': 1.0, 'T_end': 0.01, 'n_iter': 30000, 'step_scale': 0.1, 'seed': 0},
        {'T0': 0.5, 'T_end': 0.005, 'n_iter': 30000, 'step_scale': 0.05, 'seed': 1},
        {'T0': 2.0, 'T_end': 0.02, 'n_iter': 30000, 'step_scale': 0.15, 'seed': 2},
        {'T0': 1.0, 'T_end': 0.01, 'n_iter': 30000, 'step_scale': 0.08, 'seed': 3},
        {'T0': 0.3, 'T_end': 0.003, 'n_iter': 30000, 'step_scale': 0.03, 'seed': 4},
    ]
    
    # Also start from quasi-regular seeds
    for s in range(5):
        qpts = _seed_quasi_regular(seed=s, ang_scale=0.1, rad_scale=0.1)
        qoffsets = _offsets_from_points(qpts)
        sa_configs.append({
            'offsets': qoffsets,
            'T0': 1.0, 'T_end': 0.01, 'n_iter': 20000, 'step_scale': 0.1, 'seed': 10 + s
        })
    
    for cfg in sa_configs:
        if time.time() >= deadline - 20:
            break
        
        if 'offsets' in cfg:
            init_offsets = cfg['offsets']
        else:
            init_offsets = regular_offsets
        
        try:
            sa_offsets, sa_score = _sa_optimize(
                init_offsets,
                deadline,
                T0=cfg['T0'],
                T_end=cfg['T_end'],
                n_iter=cfg['n_iter'],
                step_scale=cfg['step_scale'],
                rng_seed=cfg['seed']
            )
            sa_pts = _points_from_offsets(sa_offsets)
            _update(sa_pts)
        except Exception:
            pass
    
    # Phase 2: Local optimization from best found so far
    if best_points is not None:
        # Convert to offsets and do more SA refinement
        best_offsets = _offsets_from_points(best_points)
        
        # Fine-tune SA
        for cfg in [
            {'T0': 0.1, 'T_end': 0.001, 'n_iter': 20000, 'step_scale': 0.02, 'seed': 100},
            {'T0': 0.05, 'T_end': 0.0005, 'n_iter': 20000, 'step_scale': 0.01, 'seed': 101},
        ]:
            if time.time() >= deadline - 10:
                break
            try:
                sa_offsets, sa_score = _sa_optimize(
                    best_offsets,
                    deadline,
                    T0=cfg['T0'],
                    T_end=cfg['T_end'],
                    n_iter=cfg['n_iter'],
                    step_scale=cfg['step_scale'],
                    rng_seed=cfg['seed']
                )
                sa_pts = _points_from_offsets(sa_offsets)
                _update(sa_pts)
            except Exception:
                pass
        
        # Phase 3: Nelder-Mead polish on Cartesian coordinates
        if time.time() < deadline - 5:
            try:
                pts, s = _polish_true_objective(best_points, deadline, max_iter=2000)
                _update(pts)
            except Exception:
                pass
        
        # Phase 4: One more SA from polished result
        if time.time() < deadline - 3 and best_points is not None:
            polished_offsets = _offsets_from_points(best_points)
            try:
                sa_offsets, sa_score = _sa_optimize(
                    polished_offsets,
                    deadline,
                    T0=0.05, T_end=0.001, n_iter=15000, step_scale=0.01, seed=200
                )
                sa_pts = _points_from_offsets(sa_offsets)
                _update(sa_pts)
            except Exception:
                pass
    
    if best_points is None:
        best_points = _seed_regular()
        
    return best_points
