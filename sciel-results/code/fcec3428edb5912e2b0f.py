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

def _generate_two_ring_config(n1, n2, r1, r2, rot1, rot2):
    """
    Generate points on two concentric circles.
    n1, n2: number of points on outer and inner ring.
    r1, r2: radii.
    rot1, rot2: rotations.
    """
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    return np.vstack([pts1, pts2])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    best_score = 0.0
    best_points = None

    def _update_best(points):
        nonlocal best_score, best_points
        if points is None:
            return 0.0
        pts = _normalize(points)
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
        return sc

    def _neg_score(points_flat):
        points = points_flat.reshape(-1, 2)
        return -_score(points)

    def _try_optimize_points(init_points, maxiter=3000):
        """Optimize the 2N coordinates directly."""
        if time.time() > deadline - 5:
            return
        init_norm = _normalize(init_points)
        try:
            res = scipy.optimize.minimize(
                _neg_score,
                init_norm.flatten(),
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            _update_best(res.x.reshape(-1, 2))
        except Exception:
            pass

    # 1. Explore the two-ring family analytically
    # Possible splits for N=14: (1,13), (2,12), (3,11), (4,10), (5,9), (6,8), (7,7)
    # (7,7) is symmetric and a very strong candidate.
    # (6,8) and (8,6) are also good.
    
    splits = [(7, 7), (6, 8), (8, 6), (5, 9), (9, 5), (4, 10), (10, 4), (3, 11), (11, 3), (2, 12), (12, 2)]
    
    # We will collect candidates from the two-ring search and then refine the top ones with coordinate optimization.
    two_ring_candidates = []
    
    for n1, n2 in splits:
        if time.time() > deadline - 20:
            break
            
        # Optimize the parameters of the two-ring configuration
        # Params: r_ratio (r2/r1), rot1, rot2
        # We fix r1=1 for optimization and scale at the end.
        
        def _two_ring_score(params, n1_val=n1, n2_val=n2):
            r_ratio = params[0]
            rot1 = params[1]
            rot2 = params[2]
            
            r1 = 1.0
            r2 = r_ratio * r1
            
            # Ensure r2 > 0 and points are distinct enough
            if r2 <= 0.01 or r2 >= 0.99:
                return 1e6
                
            pts = _generate_two_ring_config(n1_val, n2_val, r1, r2, rot1, rot2)
            # Normalize and score
            sc = _score(_normalize(pts))
            return -sc

        # Initial guesses for r_ratio, rot1, rot2
        # r_ratio typically between 0.4 and 0.8
        # rot1 = 0 is WLOG due to rotational symmetry of the whole set? No, relative rotation matters.
        # We can fix rot1=0 and optimize rot2 relative to it? 
        # Actually, the score is invariant under global rotation. So we can fix rot1=0.
        # But the optimizer might struggle if we don't allow it. Let's just optimize all 3.
        
        # Try a few different starting points for the ratio and rotations
        found_local_best = False
        best_local_score = -1e10
        best_local_pts = None
        
        for r_init in [0.4, 0.5, 0.6, 0.7]:
            if time.time() > deadline - 10:
                break
            for rot_init in [0.0, np.pi/n1, np.pi/n2]:
                if time.time() > deadline - 5:
                    break
                x0 = np.array([r_init, 0.0, rot_init])
                try:
                    res = scipy.optimize.minimize(
                        _two_ring_score,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    r_ratio, rot1, rot2 = res.x
                    pts = _generate_two_ring_config(n1, n2, 1.0, r_ratio, rot1, rot2)
                    sc = _score(_normalize(pts))
                    
                    if sc > best_local_score:
                        best_local_score = sc
                        best_local_pts = pts.copy()
                        found_local_best = True
                        
                except Exception:
                    pass
        
        if found_local_best and best_local_score > 0:
            two_ring_candidates.append((best_local_score, best_local_pts))

    # Sort candidates by score
    two_ring_candidates.sort(key=lambda x: x[0], reverse=True)

    # 2. Refine the top candidates with coordinate optimization
    # This is the "refinement machinery" from inspiration 1.
    # We take the best few two-ring configurations and run Nelder-Mead on all 28 coordinates.
    
    top_k = 5
    for sc, pts in two_ring_candidates[:top_k]:
        if time.time() > deadline - 15:
            break
        # Only refine if it's competitive
        if sc > best_score * 0.9 or best_score == 0:
            _try_optimize_points(pts, maxiter=2000)

    # 3. Explore regular polygon and slight perturbations
    # Regular 14-gon
    reg = _regular_polygon(N, 1.0)
    _update_best(reg)
    
    # Perturb regular polygon
    rng = np.random.RandomState(42)
    for i in range(10):
        if time.time() > deadline - 10:
            break
        noise = rng.uniform(-0.1, 0.1, (N, 2))
        init = _regular_polygon(N, 1.0) + noise
        _try_optimize_points(init, maxiter=1000)

    # 4. Random restarts with optimization
    # Generate random points and optimize
    for i in range(5):
        if time.time() > deadline - 10:
            break
        # Generate points on a disk, then optimize
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        init = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        _try_optimize_points(init, maxiter=500)

    # Final check
    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points
