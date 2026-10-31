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
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
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

def _soft_min_score(points, beta):
    """
    Score using soft-min for the triangle areas.
    This provides a smooth surrogate for the min function.
    """
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    # Soft min calculation:
    # min(areas) approx -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, we shift by the minimum area.
    a_min = np.min(areas)
    shifted = -beta * (areas - a_min)
    # max shifted is 0, so exp is <= 1
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_area = a_min - log_sum / beta
    
    if soft_min_area < 1e-15:
        return 0.0
    return soft_min_area / hull_area

def _objective_soft(x_flat, beta):
    points = x_flat.reshape(N, 2)
    score = _soft_min_score(points, beta)
    if score <= 0:
        return 1e10
    return -score

def _objective_hard(x_flat):
    points = x_flat.reshape(N, 2)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _make_ring_points(n_out, n_in, r_in, delta):
    angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    angles_in = np.linspace(delta, 2 * np.pi + delta, n_in, endpoint=False)
    
    pts_out = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    pts_in = np.column_stack([r_in * np.cos(angles_in), r_in * np.sin(angles_in)])
    
    return np.vstack([pts_out, pts_in])

def _generate_initial_configs():
    configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(pts)
    
    # 2. Ring configurations (7 outer, 6 inner)
    for r_in in [0.5, 0.6, 0.7, 0.8]:
        for delta in [0, np.pi/7, np.pi/6]:
            configs.append(_make_ring_points(7, 6, r_in, delta))
            
    # 3. Ring configurations (6 outer, 7 inner)
    for r_in in [0.5, 0.6, 0.7, 0.8]:
        for delta in [0, np.pi/6, np.pi/7]:
            configs.append(_make_ring_points(6, 7, r_in, delta))
            
    # 4. Center + Rings
    # 1 center, 6 inner, 6 outer
    center = np.array([[0.0, 0.0]])
    for r_in in [0.5, 0.7]:
        for r_out in [1.0, 1.2]:
            for delta_in in [0, np.pi/6]:
                for delta_out in [0, np.pi/6]:
                    pts_in = _make_ring_points(6, 0, r_in, delta_in)
                    pts_out = _make_ring_points(6, 0, r_out, delta_out)
                    configs.append(np.vstack([center, pts_in, pts_out]))
                    
    # 5. Random inits
    rng = np.random.RandomState(42)
    for _ in range(10):
        pts = rng.uniform(-1, 1, (N, 2))
        configs.append(pts)
        
    # 6. Random on circle
    for _ in range(5):
        angles = 2 * np.pi * np.arange(N) / N + rng.uniform(-0.1, 0.1, N)
        radii = 1.0 + rng.uniform(-0.2, 0.2, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        configs.append(pts)
        
    return configs

def _refine(points, deadline):
    """
    Refine a point configuration using soft-min objective with increasing beta.
    """
    t_start = time.time()
    best_score = _score(points)
    best_x = points.flatten().copy()
    
    # Phase 1: Coarse optimization with low beta (smooth landscape)
    beta = 50.0
    try:
        res = minimize(
            _objective_soft,
            best_x,
            args=(beta,),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5}
        )
        if res.x is not None:
            s = _score(res.x.reshape(N, 2))
            if s > best_score:
                best_score = s
                best_x = res.x.copy()
    except Exception:
        pass

    # Phase 2: Medium beta
    if time.time() < deadline - 10:
        beta = 200.0
        try:
            res = minimize(
                _objective_soft,
                best_x,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if res.x is not None:
                s = _score(res.x.reshape(N, 2))
                if s > best_score:
                    best_score = s
                    best_x = res.x.copy()
        except Exception:
            pass

    # Phase 3: High beta (sharper minimum)
    if time.time() < deadline - 5:
        beta = 1000.0
        try:
            res = minimize(
                _objective_soft,
                best_x,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.x is not None:
                s = _score(res.x.reshape(N, 2))
                if s > best_score:
                    best_score = s
                    best_x = res.x.copy()
        except Exception:
            pass
            
    # Phase 4: Nelder-Mead polish on hard objective if time permits
    if time.time() < deadline - 2:
        try:
            res = minimize(
                _objective_hard,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_score:
                best_score = s
                best_x = res.x.copy()
        except Exception:
            pass

    return best_x.reshape(N, 2), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    best_points = None
    best_score = 0.0

    def _update_best(points, score=None):
        nonlocal best_points, best_score
        if points is None or len(points) != N:
            return
        if score is None:
            score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

    configs = _generate_initial_configs()
    
    # Sort initial configs by hard score to prioritize good starting points
    scored_configs = []
    for cfg in configs:
        s = _score(cfg)
        scored_configs.append((s, cfg))
    scored_configs.sort(key=lambda x: x[0], reverse=True)
    
    # Refine top candidates
    # We have ~115s. Each refinement might take 10-30s depending on convergence.
    # Let's try to refine the top 5-10 candidates.
    top_k = 8
    for i in range(min(top_k, len(scored_configs))):
        if time.time() > deadline - 15:
            break
        init_score, init_pts = scored_configs[i]
        if init_score < best_score * 0.9 and best_score > 0:
            # If this config is significantly worse than current best, maybe skip?
            # But different basins... let's still try a few.
            if i > 4:
                continue
                
        refined_pts, refined_score = _refine(init_pts, deadline)
        _update_best(refined_pts, refined_score)
        print(f"Config {i} (init {init_score:.6f}) -> {refined_score:.6f}", flush=True)
        
        # If we've found a very good score, we might want to spend more time polishing it
        # instead of trying new inits.
        if best_score > 0.05: # Arbitrary threshold, but helps focus
            if time.time() > deadline - 30:
                 # Spend remaining time polishing the best
                 final_pts, final_score = _refine(best_points, deadline)
                 _update_best(final_pts, final_score)
                 break

    # Final polish on the absolute best
    if best_points is not None and time.time() < deadline - 2:
        final_pts, final_score = _refine(best_points, deadline)
        _update_best(final_pts, final_score)

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points
