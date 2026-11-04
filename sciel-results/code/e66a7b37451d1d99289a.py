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

def _smooth_min(areas, temperature):
    """Smooth minimum using log-sum-exp."""
    areas = np.maximum(areas, 1e-15)
    scaled = -areas / temperature
    scaled_max = np.max(scaled)
    log_sum = np.log(np.sum(np.exp(scaled - scaled_max))) + scaled_max
    return -log_sum * temperature

def _smooth_objective(x, temperature=0.005):
    """
    Smooth objective: negative of soft-min triangle area / hull area.
    x is a flat array of shape (N*2,).
    """
    pts = x.reshape(N, 2)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e8
    areas = _tri_areas(pts)
    soft_min_area = _smooth_min(areas, temperature)
    return -(soft_min_area / ha)

def _exact_objective(x):
    """Exact objective: negative of min triangle area / hull area."""
    pts = x.reshape(N, 2)
    return -_score(pts)

def _nelder_mead_polish(pts, time_budget):
    """Polish using Nelder-Mead on the exact objective."""
    t0 = time.time()
    best = pts.copy()
    best_s = _score(pts)
    
    try:
        result = minimize(
            _exact_objective, best.flatten(), method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
                     'initial_simplex': 1e-3}
        )
        cand = result.x.reshape(N, 2)
        s = _score(cand)
        if s > best_s:
            best_s = s
            best = cand.copy()
    except Exception:
        pass
    
    # Second polish with smaller simplex if time permits
    if time.time() - t0 < time_budget - 1:
        try:
            result = minimize(
                _exact_objective, best.flatten(), method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-11, 'fatol': 1e-13,
                         'initial_simplex': 1e-4}
            )
            cand = result.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    return best, best_s

def _l_bfgs_polish(pts, time_budget):
    """Polish using L-BFGS-B on smooth surrogate."""
    t0 = time.time()
    best = pts.copy()
    best_s = _score(pts)
    
    for temp in [0.01, 0.005, 0.002, 0.001]:
        if time.time() - t0 > time_budget - 1:
            break
        try:
            result = minimize(
                lambda x: _smooth_objective(x, temp),
                best.flatten(), method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            cand = result.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    return best, best_s

def _differential_evolution_search(time_budget, rng):
    """
    Global search using differential evolution.
    Search in bounded box [-1, 1]^2 for all 13 points.
    """
    t0 = time.time()
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Use a moderate temperature for the smooth objective
    temp = 0.01
    
    def de_obj(x):
        if time.time() - t0 > time_budget - 1:
            return 1e8
        return _smooth_objective(x, temp)
    
    try:
        result = differential_evolution(
            de_obj, bounds,
            seed=rng.randint(2**31 - 1),
            maxiter=100,
            tol=1e-6,
            popsize=15,
            mutation=(0.3, 1.0),
            recombination=0.8,
            polish=True,
            init='halton',
            x0=None,
            updating='immediate',
            workers=1
        )
        cand = result.x.reshape(N, 2)
        s = _score(cand)
        return cand, s
    except Exception:
        return None, 0.0

def _local_search_cartesian(pts, time_budget, rng):
    """
    Coordinate-wise local search in Cartesian space.
    """
    t0 = time.time()
    best = pts.copy()
    best_s = _score(pts)
    step = 0.01
    
    while time.time() - t0 < time_budget - 1:
        improved = False
        for i in range(N):
            for dim in range(2):
                for delta in [-step, step]:
                    trial = best.copy()
                    trial[i, dim] += delta
                    s = _score(trial)
                    if s > best_s:
                        best = trial.copy()
                        best_s = s
                        improved = True
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
    
    return best, best_s

def construct():
    """Main entry point: find optimal 13-point configuration."""
    start_time = time.time()
    DEADLINE = start_time + 148.0  # 148s safety margin for 150s limit
    
    def _time_left():
        return DEADLINE - time.time()
    
    rng = np.random.RandomState(42)
    
    best_points = None
    best_ratio = -1e9
    
    # Generate diverse starting points
    starts = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    starts.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2-4: Halton-like sequences
    for base1, base2 in [(2, 3), (3, 5), (5, 7)]:
        result = []
        f1, f2 = 1.0, 1.0
        for i in range(1, N + 1):
            x = 0.0
            f = 1.0
            val = i
            while val > 0:
                val, d = divmod(val, base1)
                x += d * f
                f /= base1
            y = 0.0
            f = 1.0
            val = i
            while val > 0:
                val, d = divmod(val, base2)
                y += d * f
                f /= base2
            r = np.sqrt(x)
            theta = 2 * np.pi * y
            result.append([r * np.cos(theta), r * np.sin(theta)])
        starts.append(np.array(result))
    
    # 5. Concentric rings
    pts = [[0.0, 0.0]]
    remaining = N - 1
    for ring in range(3):
        if remaining <= 0:
            break
        n_ring = min(remaining, max(3, (N-1)//3))
        if ring == 2:
            n_ring = remaining
        radius = (ring + 1) / 3
        angles = np.linspace(0, 2*np.pi, n_ring, endpoint=False) + ring * np.pi/6
        ring_pts = np.column_stack([radius*np.cos(angles), radius*np.sin(angles)])
        pts.append(ring_pts)
        remaining -= n_ring
    if remaining > 0:
        angles = np.linspace(0, 2*np.pi, remaining, endpoint=False)
        pts.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    starts.append(np.vstack(pts)[:N])
    
    # 6-12: Jittered and random starts
    for seed in range(7):
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2*np.pi, N)
        starts.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
    
    # Evaluate and sort starts
    scored_starts = []
    for start in starts:
        if _time_left() < 5:
            break
        try:
            s = _score(start)
            if s > best_ratio:
                best_ratio = s
                best_points = start.copy()
            scored_starts.append((s, start))
        except Exception:
            scored_starts.append((0.0, start))
    
    scored_starts.sort(key=lambda x: -x[0])
    
    # Phase 1: Differential evolution global search (use ~60% of budget)
    de_time = min(60.0, _time_left() * 0.6)
    if de_time > 10:
        try:
            de_pts, de_score = _differential_evolution_search(de_time, rng)
            if de_pts is not None and de_score > best_ratio:
                best_ratio = de_score
                best_points = de_pts.copy()
        except Exception:
            pass
    
    # Phase 2: Polish the best point found so far
    if best_points is not None and _time_left() > 15:
        polish_time = min(_time_left() * 0.3, 45.0)
        try:
            polished, pol_score = _l_bfgs_polish(best_points, polish_time * 0.5)
            if pol_score > best_ratio:
                best_ratio = pol_score
                best_points = polished.copy()
            
            if _time_left() > 5:
                nm_time = min(_time_left() - 2, 30.0)
                polished2, nm_score = _nelder_mead_polish(best_points, nm_time)
                if nm_score > best_ratio:
                    best_ratio = nm_score
                    best_points = polished2.copy()
        except Exception:
            pass
    
    # Phase 3: Try polishing from other good starts if time remains
    for init_score, start in scored_starts[:5]:
        if _time_left() < 10:
            break
        if init_score < best_ratio * 0.9:
            continue
        try:
            alloc = min(_time_left() - 5, 20.0)
            if alloc < 5:
                continue
            polished, pol_score = _l_bfgs_polish(start, alloc * 0.4)
            if pol_score > best_ratio:
                best_ratio = pol_score
                best_points = polished.copy()
            
            if _time_left() > 5 and pol_score > best_ratio * 0.95:
                nm_time = min(_time_left() - 3, 15.0)
                polished2, nm_score = _nelder_mead_polish(best_points, nm_time)
                if nm_score > best_ratio:
                    best_ratio = nm_score
                    best_points = polished2.copy()
        except Exception:
            pass
    
    # Phase 4: Final Nelder-Mead polish on best
    if best_points is not None and _time_left() > 8:
        try:
            final_pts, final_score = _nelder_mead_polish(best_points, _time_left() - 2)
            if final_score > best_ratio:
                best_ratio = final_score
                best_points = final_pts.copy()
        except Exception:
            pass
    
    # Phase 5: Local search polish
    if best_points is not None and _time_left() > 5:
        try:
            ls_pts, ls_score = _local_search_cartesian(best_points, _time_left() - 2, rng)
            if ls_score > best_ratio:
                best_ratio = ls_score
                best_points = ls_pts.copy()
        except Exception:
            pass
    
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
