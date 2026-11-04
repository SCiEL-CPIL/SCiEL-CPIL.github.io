import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations
import time

N = 13
TRIS = list(combinations(range(N), 3))
NUM_TRIS = len(TRIS)

def _triangle_areas(pts):
    """Compute areas of all C(N,3) triangles as a 1D array."""
    areas = np.empty(NUM_TRIS)
    for idx, (i, j, k) in enumerate(TRIS):
        p1 = pts[i]
        p2 = pts[j]
        p3 = pts[k]
        areas[idx] = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
    return areas

def _convex_hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10

def _score(pts):
    min_area = np.min(_triangle_areas(pts))
    h_area = _convex_hull_area(pts)
    if h_area < 1e-10:
        return 0.0
    return min_area / h_area

def _soft_min_objective(x, beta=50.0):
    """
    Smooth surrogate for -min_triangle_area / hull_area.
    x is a vector of length 2*N (flattened points).
    """
    pts = x.reshape(N, 2)
    areas = _triangle_areas(pts)
    areas = np.maximum(areas, 1e-15)
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min_area = -log_sum_exp / beta
    
    h_area = _convex_hull_area(pts)
    if h_area < 1e-10:
        return 100.0
    return -soft_min_area / h_area

def _exact_objective(x):
    """
    Exact objective for minimization: -score.
    Penalizes non-convex configurations heavily.
    """
    pts = x.reshape(N, 2)
    try:
        hull = ConvexHull(pts)
        if len(hull.vertices) < N:
            return 100.0
    except Exception:
        return 100.0
    return -_score(pts)

def _halton_sequence(n, base):
    """Generate the first n points of the Halton sequence in base."""
    q = 1.0
    out = 0.0
    for _ in range(n):
        q /= base
        out += q * (int(1.0 / q) % base)
    return out

def _halton_points(n, dim=2, offset=0.0, scale=1.0):
    """
    Generate points using Halton sequences for low-discrepancy sampling.
    """
    pts = np.zeros((n, dim))
    for d in range(dim):
        base = d + 2
        for i in range(n):
            pts[i, d] = _halton_sequence(i + 1, base) * scale + offset
    return pts

def _center_ring(n, radius=1.0):
    """Generate points on a circle of given radius."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _regular_polygon(n, radius=1.0):
    return _center_ring(n, radius)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s/150s limit
    
    best_score = 0.0
    best_points = None
    
    def update_best(pts, score):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # 1. Initial Candidates
    # Regular polygon is a strong baseline
    base_pts = _regular_polygon(N, 1.0)
    update_best(base_pts, _score(base_pts))
    
    # Halton sequence points (low discrepancy)
    # Scale to fit in unit area roughly? Halton points are in [0,1]^2.
    # Area of convex hull of Halton points is not 1. We will normalize later or just use as is.
    # The score is invariant to scaling/translation if we normalize by hull area.
    # So we can just use the raw Halton points.
    halton_pts = _halton_points(N, dim=2)
    # Center them
    halton_pts -= np.mean(halton_pts, axis=0)
    try:
        if len(ConvexHull(halton_pts).vertices) == N:
            update_best(halton_pts, _score(halton_pts))
    except:
        pass

    # Random perturbations of regular polygon
    rng = np.random.default_rng(42)
    for i in range(10):
        noise = rng.normal(0, 0.1, (N, 2))
        pts = base_pts + noise
        try:
            if len(ConvexHull(pts).vertices) == N:
                update_best(pts, _score(pts))
        except:
            pass

    if best_points is None:
        best_points = base_pts
        best_score = _score(best_points)

    # 2. Differential Evolution on the full 2N parameter space
    # This is a high-dimensional search (26 dims), but DE is robust.
    # We restrict bounds to a box around the origin.
    # Since the problem is scale-invariant (score is ratio), we can fix the scale.
    # However, DE works better with bounded variables.
    # Let's bound x, y in [-2, 2].
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    # To help DE, we can provide an initial population or just let it sample.
    # Using a seed for reproducibility.
    
    try:
        # Use a smaller population and fewer iterations to stay within time
        # 26 dimensions * 15 popsize * 100 maxiter = ~39000 evaluations
        # Each evaluation is O(N^2) or O(N log N) for hull + O(N^3) for triangles?
        # Triangle area calc is O(1) per tri, 286 tris. Fast.
        # Hull is O(N log N). Fast.
        
        res_de = differential_evolution(
            _exact_objective,
            bounds=bounds,
            seed=42,
            maxiter=50,
            popsize=20,
            tol=1e-5,
            polish=False,
            init='sobol'  # Better initial spread
        )
        
        if time.time() < deadline - 10:
            pts_de = res_de.x.reshape(N, 2)
            try:
                if len(ConvexHull(pts_de).vertices) == N:
                    s_de = _score(pts_de)
                    update_best(pts_de, s_de)
            except:
                pass
                
    except Exception as e:
        pass

    # 3. Local Refinement with L-BFGS-B on Soft-Min Surrogate
    # Use the best points found so far.
    # Progressive sharpening of the soft-min (increasing beta).
    if best_points is not None:
        current_best = best_points.copy()
        betas = [20.0, 50.0, 100.0, 200.0, 500.0]
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
            
            try:
                res_lbfgs = minimize(
                    _soft_min_objective,
                    current_best.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                res_pts = res_lbfgs.x.reshape(N, 2)
                try:
                    if len(ConvexHull(res_pts).vertices) == N:
                        s_res = _score(res_pts)
                        update_best(res_pts, s_res)
                        if s_res > _score(current_best):
                            current_best = res_pts.copy()
                except:
                    pass
                    
            except Exception:
                pass

    # 4. Final Polish with Nelder-Mead on Exact Score
    # NM is derivative-free and can handle the kinks in the min function better than L-BFGS on the exact objective.
    if best_points is not None and time.time() < deadline - 2:
        try:
            res_nm = minimize(
                _exact_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 2000, 'adaptive': True}
            )
            
            res_pts = res_nm.x.reshape(N, 2)
            try:
                if len(ConvexHull(res_pts).vertices) == N:
                    s_nm = _score(res_pts)
                    update_best(res_pts, s_nm)
            except:
                pass
                
        except Exception:
            pass

    return best_points
