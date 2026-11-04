"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

# Precompute triangle indices
TRI_INDICES = list(combinations(range(N), 3))
NUM_TRIS = len(TRI_INDICES)  # 286

# Precompute index arrays for vectorized area computation
_I = np.array([t[0] for t in TRI_INDICES])
_J = np.array([t[1] for t in TRI_INDICES])
_K = np.array([t[2] for t in TRI_INDICES])

def _tri_areas(pts):
    """Vectorized triangle areas for all 286 triples."""
    pts = np.asarray(pts, dtype=np.float64)
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    return 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])

def _hull_area(pts):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return 0.0
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    min_area = np.min(_tri_areas(pts))
    return min_area / ha

def _soft_min_ratio(pts, temperature):
    """
    Smooth surrogate: soft-min of triangle areas / hull area.
    Uses log-sum-exp for numerical stability.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return -1e9
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return -1e9
    except Exception:
        return -1e9
    
    areas = _tri_areas(pts)
    areas = np.maximum(areas, 1e-15)
    
    # soft_min = -T * log(sum(exp(-a/T)))
    scaled = -areas / temperature
    scaled_max = np.max(scaled)
    log_sum = np.log(np.sum(np.exp(scaled - scaled_max))) + scaled_max
    soft_min_area = -log_sum * temperature
    
    return soft_min_area / hull_area

# === NEW: Support function (convex polygon) parametrization ===
# Fixed normal angles for 13 tangent lines
_NORMAL_ANGLES = np.linspace(0, 2*np.pi, N, endpoint=False)
_NORMAL_DIRS = np.column_stack([np.cos(_NORMAL_ANGLES), np.sin(_NORMAL_ANGLES)])

def _support_to_points(offsets):
    """
    Convert 13 support function values (offsets along fixed normals)
    to the 13 vertices of the resulting convex polygon.
    
    The polygon is the intersection of half-planes: n_i · x <= h_i
    The vertices are intersections of consecutive edge lines.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    n = len(offsets)
    points = np.zeros((n, 2))
    
    for i in range(n):
        # Line i: n_i · x = h_i
        # Line i+1: n_{i+1} · x = h_{i+1}
        ni = _NORMAL_DIRS[i]
        ni1 = _NORMAL_DIRS[(i + 1) % n]
        hi = offsets[i]
        hi1 = offsets[(i + 1) % n]
        
        # Solve: ni[0]*x + ni[1]*y = hi
        #         ni1[0]*x + ni1[1]*y = hi1
        det = ni[0] * ni1[1] - ni[1] * ni1[0]
        if abs(det) < 1e-15:
            # Degenerate: use midpoint of normals as fallback
            points[i] = 0.5 * (ni + ni1) * hi
        else:
            x = (hi * ni1[1] - hi1 * ni[1]) / det
            y = (ni[0] * hi1 - ni1[0] * hi) / det
            points[i] = np.array([x, y])
    
    return points

def _score_from_offsets(offsets):
    """Score computed from support function offsets."""
    pts = _support_to_points(offsets)
    return _score(pts)

def _neg_soft_min_from_offsets(offsets, temperature):
    """Negative soft-min ratio for minimization from offsets."""
    pts = _support_to_points(offsets)
    val = _soft_min_ratio(pts, temperature)
    if val < -1e8:
        return 1e8
    return -val

def _neg_score_from_offsets(offsets):
    """Negative exact score for minimization from offsets."""
    return -_score_from_offsets(offsets)

# === Helper functions for diverse initializations ===

def _regular_polygon(n, radius=1.0, center=(0,0)):
    """Generate regular polygon vertices."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius*np.cos(angles), 
                            center[1] + radius*np.sin(angles)])

def _pts_to_offsets(pts):
    """
    Convert a set of points to support function offsets.
    For each fixed normal direction, compute the maximum projection.
    This gives a valid (possibly non-tight) support function.
    """
    pts = np.asarray(pts, dtype=np.float64)
    offsets = np.array([np.max(pts @ nd) for nd in _NORMAL_DIRS])
    return offsets

def _generate_diverse_offset_starts(rng):
    """Generate diverse initial offset vectors."""
    starts = []
    
    # 1. Regular polygon (all equal offsets)
    starts.append(np.ones(N) * 1.0)
    
    # 2. Slightly perturbed regular
    for scale in [0.8, 1.2]:
        starts.append(np.ones(N) * scale)
    
    # 3. Random offsets in [0.5, 1.5]
    for _ in range(5):
        starts.append(rng.uniform(0.5, 1.5, N))
    
    # 4. Oscillating patterns
    for freq in [2, 3, 4]:
        t = np.linspace(0, 2*np.pi, N, endpoint=False)
        starts.append(1.0 + 0.15 * np.cos(freq * t))
    
    # 5. From a regular polygon with random jitter
    reg = _regular_polygon(N, radius=1.0)
    for _ in range(3):
        jittered = reg + rng.normal(0, 0.1, (N, 2))
        starts.append(_pts_to_offsets(jittered))
    
    # 6. Concentric-like: alternate high/low
    alt = np.ones(N)
    alt[::2] = 1.2
    alt[1::2] = 0.8
    starts.append(alt)
    
    return starts

def _differential_evolution_search(offsets_init, time_budget, rng):
    """
    Run differential evolution on the 13 offset parameters.
    Uses the exact score as the objective.
    """
    start_time = time.time()
    best_offsets = offsets_init.copy()
    best_score = _score_from_offsets(offsets_init)
    
    # Bounds: offsets should be positive and bounded
    lower = 0.1 * np.ones(N)
    upper = 3.0 * np.ones(N)
    
    # Run DE with time limit
    max_iter = 200
    popsize = 15
    
    def callback(xk, convergence):
        nonlocal best_offsets, best_score
        s = _score_from_offsets(xk)
        if s > best_score:
            best_score = s
            best_offsets = xk.copy()
        return time.time() - start_time > time_budget - 5.0
    
    try:
        result = differential_evolution(
            _neg_score_from_offsets,
            bounds=list(zip(lower, upper)),
            x0=offsets_init,
            seed=rng.randint(0, 2**31),
            maxiter=max_iter,
            popsize=popsize,
            tol=1e-9,
            mutation=(0.4, 1.0),
            recombination=0.8,
            polish=False,  # We'll do our own polish
            callback=callback,
            init='sobol'
        )
        if result.success or result.fun < _neg_score_from_offsets(best_offsets):
            cand_score = _score_from_offsets(result.x)
            if cand_score > best_score:
                best_score = cand_score
                best_offsets = result.x.copy()
    except Exception:
        pass
    
    return best_offsets, best_score

def _local_polish_offsets(offsets, time_budget):
    """
    Local polish using L-BFGS-B on the soft-min surrogate in offset space.
    """
    start_time = time.time()
    best_offsets = offsets.copy()
    best_score = _score_from_offsets(offsets)
    
    # Multi-temperature L-BFGS
    for temp in [0.01, 0.003, 0.001]:
        if time.time() - start_time > time_budget - 3:
            break
        
        try:
            res = minimize(
                _neg_soft_min_from_offsets,
                best_offsets,
                args=(temp,),
                method='L-BFGS-B',
                bounds=[(0.1, 3.0)] * N,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            cand_score = _score_from_offsets(res.x)
            if cand_score > best_score:
                best_score = cand_score
                best_offsets = res.x.copy()
        except Exception:
            pass
    
    # Final Nelder-Mead on exact objective
    if time.time() - start_time < time_budget - 2:
        try:
            res = minimize(
                _neg_score_from_offsets,
                best_offsets,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12,
                         'initial_simplex': 0.01}
            )
            cand_score = _score_from_offsets(res.x)
            if cand_score > best_score:
                best_score = cand_score
                best_offsets = res.x.copy()
        except Exception:
            pass
    
    return best_offsets, best_score

def _cartesian_polish(pts, time_budget):
    """
    Final polish in Cartesian space using L-BFGS on soft-min.
    """
    start_time = time.time()
    best = pts.copy()
    best_s = _score(pts)
    
    for temp in [0.005, 0.001]:
        if time.time() - start_time > time_budget - 2:
            break
        
        def obj(x):
            p = x.reshape(N, 2)
            val = _soft_min_ratio(p, temp)
            if val < -1e8:
                return 1e8
            return -val
        
        try:
            res = minimize(obj, best.flatten(), method='L-BFGS-B',
                           options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-9})
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    return best, best_s

def construct():
    """Main entry point: find optimal 13-point configuration."""
    start_time = time.time()
    total_budget = 145.0  # Safety margin for 150s limit
    
    rng = np.random.RandomState(12345)
    
    best_points = _regular_polygon(N, radius=1.0)
    best_ratio = _score(best_points)
    
    # Generate diverse offset initializations
    offset_starts = _generate_diverse_offset_starts(rng)
    
    # Evaluate initial offsets
    scored_offsets = []
    for off in offset_starts:
        try:
            s = _score_from_offsets(off)
            scored_offsets.append((s, off))
        except Exception:
            scored_offsets.append((0.0, off))
    scored_offsets.sort(key=lambda x: -x[0])
    
    # Phase 1: Differential evolution from best starts
    n_de_runs = 4
    de_time_each = min(25.0, (total_budget - 30.0) / n_de_runs)
    
    for i in range(n_de_runs):
        if time.time() - start_time > total_budget - 15:
            break
        
        remaining = total_budget - (time.time() - start_time) - 15.0
        alloc = min(de_time_each, remaining / max(1, n_de_runs - i))
        if alloc < 5.0:
            break
        
        init_off = scored_offsets[i % len(scored_offsets)][1]
        
        try:
            opt_off, opt_score = _differential_evolution_search(init_off, alloc, rng)
            opt_pts = _support_to_points(opt_off)
            if opt_score > best_ratio:
                best_ratio = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Phase 2: Local polish on best offsets
    if time.time() - start_time < total_budget - 10:
        polish_time = min(20.0, total_budget - (time.time() - start_time) - 10.0)
        if polish_time > 5.0:
            try:
                # Convert best points back to offsets for polish
                best_off = _pts_to_offsets(best_points)
                opt_off, opt_score = _local_polish_offsets(best_off, polish_time)
                opt_pts = _support_to_points(opt_off)
                if opt_score > best_ratio:
                    best_ratio = opt_score
                    best_points = opt_pts.copy()
            except Exception:
                pass
    
    # Phase 3: Cartesian polish (allows escaping the convex polygon constraint)
    if time.time() - start_time < total_budget - 5:
        cart_time = min(15.0, total_budget - (time.time() - start_time) - 5.0)
        if cart_time > 5.0:
            try:
                final_pts, final_score = _cartesian_polish(best_points, cart_time)
                if final_score > best_ratio:
                    best_ratio = final_score
                    best_points = final_pts.copy()
            except Exception:
                pass
    
    # Phase 4: One more DE run with remaining time if available
    if time.time() - start_time < total_budget - 8:
        remaining = total_budget - (time.time() - start_time) - 8.0
        if remaining > 10.0:
            try:
                best_off = _pts_to_offsets(best_points)
                opt_off, opt_score = _differential_evolution_search(best_off, remaining, rng)
                opt_pts = _support_to_points(opt_off)
                if opt_score > best_ratio:
                    best_ratio = opt_score
                    best_points = opt_pts.copy()
            except Exception:
                pass
    
    # Final safety check
    if time.time() - start_time > total_budget - 2:
        # Return immediately
        pass
    
    return best_points
# EVOLVE-BLOCK-END
