import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

# Pre-compute triangle indices for speed
def _get_triangle_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]
_N_TRI = len(_TRI_INDICES)

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
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        mta = _min_triangle_area(points)
        return mta / ha
    except Exception:
        return 0.0

def _normalize(points):
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _two_heptagons(r1, r2, rot1, rot2):
    """Generate 14 points as two regular heptagons with given radii and rotations."""
    pts1 = _regular_polygon(7, r1, rot1)
    pts2 = _regular_polygon(7, r2, rot2)
    return np.vstack([pts1, pts2])

def _three_ring(r1, r2, r3, rot1, rot2, rot3, n1=5, n2=5, n3=4):
    """Generate 14 points on three concentric regular polygons."""
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    pts3 = _regular_polygon(n3, r3, rot3)
    return np.vstack([pts1, pts2, pts3])

def _four_ring(r1, r2, r3, r4, rot1, rot2, rot3, rot4, n1=4, n2=4, n3=3, n4=3):
    """Generate 14 points on four concentric regular polygons."""
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    pts3 = _regular_polygon(n3, r3, rot3)
    pts4 = _regular_polygon(n4, r4, rot4)
    return np.vstack([pts1, pts2, pts3, pts4])

def _neg_two_heptagons_score(params):
    """Negative score for two-heptagon configuration."""
    r1, r2, rot1, rot2 = params
    if r1 < 0.01 or r2 < 0.01:
        return 1e6
    pts = _two_heptagons(r1, r2, rot1, rot2)
    sc = _score(_normalize(pts))
    return -sc

def _neg_three_ring_score(params):
    """Negative score for three-ring configuration."""
    r1, r2, r3, rot1, rot2, rot3 = params
    if r1 < 0.01 or r2 < 0.01 or r3 < 0.01:
        return 1e6
    pts = _three_ring(r1, r2, r3, rot1, rot2, rot3)
    sc = _score(_normalize(pts))
    return -sc

def _neg_four_ring_score(params):
    """Negative score for four-ring configuration."""
    r1, r2, r3, r4, rot1, rot2, rot3, rot4 = params
    if r1 < 0.01 or r2 < 0.01 or r3 < 0.01 or r4 < 0.01:
        return 1e6
    pts = _four_ring(r1, r2, r3, r4, rot1, rot2, rot3, rot4)
    sc = _score(_normalize(pts))
    return -sc

def _neg_exact_score(x_flat):
    """Negative exact score for optimization."""
    points = x_flat.reshape(-1, 2)
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)

    def _update_best(points):
        nonlocal best_score, best_points
        pts = _normalize(points)
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
        return sc

    def _try_optimize_exact(init_points, maxiter=500):
        """Optimize using Nelder-Mead on the exact score."""
        if time.time() > deadline - 5:
            return
        init_norm = _normalize(init_points)
        x0 = init_norm.flatten()
        
        try:
            res = scipy.optimize.minimize(
                _neg_exact_score,
                x0,
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            _update_best(res.x.reshape(-1, 2))
        except Exception:
            pass

    def _coordinate_descent_polish(pts, max_rounds=3):
        """Polish using coordinate descent on the exact score."""
        if time.time() > deadline - 10:
            return pts
        for iteration in range(max_rounds):
            if time.time() > deadline - 8:
                break
            for i in range(N):
                if time.time() > deadline - 8:
                    break
                def _neg_score_one_point(params):
                    new_pts = pts.copy()
                    new_pts[i, 0] = params[0]
                    new_pts[i, 1] = params[1]
                    return -_score(new_pts)
                
                x0 = np.array([pts[i, 0], pts[i, 1]])
                try:
                    res = scipy.optimize.minimize(
                        _neg_score_one_point,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 50, 'xatol': 1e-11, 'fatol': 1e-13}
                    )
                    pts[i, 0] = res.x[0]
                    pts[i, 1] = res.x[1]
                except Exception:
                    pass
        return pts

    # === PHASE 1: Two-heptagon differential evolution ===
    bounds_h = [(0.1, 1.5), (0.1, 1.5), (-np.pi/7, np.pi/7), (-np.pi/7, np.pi/7)]
    
    try:
        de_res = scipy.optimize.differential_evolution(
            _neg_two_heptagons_score,
            bounds_h,
            maxiter=200,
            tol=1e-8,
            popsize=15,
            seed=42,
            polish=True,
            init='sobol'
        )
        best_params_de = de_res.x
        best_pts_de = _two_heptagons(*best_params_de)
        sc_de = _update_best(best_pts_de)
        
        # Refine the best DE result
        if time.time() < deadline - 20:
            _try_optimize_exact(best_pts_de, maxiter=300)
    except Exception:
        pass

    # === PHASE 2: Three-ring differential evolution ===
    bounds_3r = [(0.1, 1.5), (0.1, 1.2), (0.05, 0.9),
                 (-np.pi/5, np.pi/5), (-np.pi/5, np.pi/5), (-np.pi/4, np.pi/4)]
    
    if time.time() < deadline - 30:
        try:
            de_res_3r = scipy.optimize.differential_evolution(
                _neg_three_ring_score,
                bounds_3r,
                maxiter=150,
                tol=1e-8,
                popsize=15,
                seed=42,
                polish=True,
                init='sobol'
            )
            best_params_3r = de_res_3r.x
            pts_3r = _three_ring(*best_params_3r)
            sc_3r = _update_best(pts_3r)
            
            if time.time() < deadline - 20:
                _try_optimize_exact(pts_3r, maxiter=300)
        except Exception:
            pass

    # === PHASE 3: Four-ring differential evolution ===
    bounds_4r = [(0.1, 1.5), (0.1, 1.2), (0.1, 0.9), (0.1, 0.6),
                 (-np.pi/4, np.pi/4), (-np.pi/4, np.pi/4), (-np.pi/3, np.pi/3), (-np.pi/3, np.pi/3)]
    
    if time.time() < deadline - 30:
        try:
            de_res_4r = scipy.optimize.differential_evolution(
                _neg_four_ring_score,
                bounds_4r,
                maxiter=150,
                tol=1e-8,
                popsize=15,
                seed=42,
                polish=True,
                init='sobol'
            )
            best_params_4r = de_res_4r.x
            pts_4r = _four_ring(*best_params_4r)
            sc_4r = _update_best(pts_4r)
            
            if time.time() < deadline - 20:
                _try_optimize_exact(pts_4r, maxiter=300)
        except Exception:
            pass

    # === PHASE 4: Local refinement from multiple starting points ===
    if time.time() < deadline - 20:
        symmetric_starts_2h = [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, np.pi/7],
            [1.0, 0.5, 0.0, 0.0],
            [1.0, 0.5, 0.0, np.pi/7],
            [1.0, 0.3, 0.0, np.pi/14],
            [1.0, 0.7, 0.0, np.pi/7],
            [0.8, 0.8, 0.0, np.pi/7],
            [1.0, 0.4, np.pi/14, 0.0],
        ]
        
        for x0 in symmetric_starts_2h:
            if time.time() > deadline - 15:
                break
            try:
                res = scipy.optimize.minimize(
                    _neg_two_heptagons_score,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                pts = _two_heptagons(*res.x)
                sc = _update_best(pts)
                
                if sc > best_score * 0.95 and time.time() < deadline - 15:
                    _try_optimize_exact(pts, maxiter=200)
            except Exception:
                pass

        # Three-ring starts
        symmetric_starts_3r = [
            [1.0, 0.6, 0.3, 0.0, 0.0, 0.0],
            [1.0, 0.6, 0.3, 0.0, np.pi/5, 0.0],
            [1.0, 0.7, 0.4, 0.0, 0.0, np.pi/4],
            [1.0, 0.5, 0.25, np.pi/5, 0.0, 0.0],
        ]
        
        for x0 in symmetric_starts_3r:
            if time.time() > deadline - 15:
                break
            try:
                res = scipy.optimize.minimize(
                    _neg_three_ring_score,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                pts = _three_ring(*res.x)
                sc = _update_best(pts)
                
                if sc > best_score * 0.95 and time.time() < deadline - 15:
                    _try_optimize_exact(pts, maxiter=200)
            except Exception:
                pass

        # Four-ring starts
        symmetric_starts_4r = [
            [1.0, 0.7, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.7, 0.4, 0.1, 0.0, np.pi/4, 0.0, np.pi/3],
        ]
        
        for x0 in symmetric_starts_4r:
            if time.time() > deadline - 15:
                break
            try:
                res = scipy.optimize.minimize(
                    _neg_four_ring_score,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                pts = _four_ring(*res.x)
                sc = _update_best(pts)
                
                if sc > best_score * 0.95 and time.time() < deadline - 15:
                    _try_optimize_exact(pts, maxiter=200)
            except Exception:
                pass

    # === PHASE 5: Random restarts with noise ===
    for i in range(8):
        if time.time() > deadline - 15:
            break
        r1 = rng.uniform(0.5, 1.5)
        r2 = rng.uniform(0.5, 1.5)
        rot1 = rng.uniform(-np.pi/7, np.pi/7)
        rot2 = rng.uniform(-np.pi/7, np.pi/7)
        init = _two_heptagons(r1, r2, rot1, rot2)
        
        noise = rng.normal(0, 0.05, (N, 2))
        init = init + noise
        
        if time.time() < deadline - 15:
            _try_optimize_exact(init, maxiter=300)

    # === PHASE 6: Coordinate descent polish ===
    if best_points is not None and time.time() < deadline - 10:
        pts = best_points.copy()
        pts = _coordinate_descent_polish(pts, max_rounds=3)
        _update_best(pts)

    # === PHASE 7: Final Nelder-Mead polish ===
    if best_points is not None and time.time() < deadline - 5:
        _try_optimize_exact(best_points, maxiter=200)

    # === PHASE 8: Final coordinate descent ===
    if best_points is not None and time.time() < deadline - 2:
        pts = best_points.copy()
        pts = _coordinate_descent_polish(pts, max_rounds=2)
        _update_best(pts)

    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0))
    else:
        best_points = _normalize(best_points)
        
    final_score = _score(best_points)
    if final_score < 1e-10:
        best_points = _normalize(_regular_polygon(N, 1.0))
        
    return best_points
