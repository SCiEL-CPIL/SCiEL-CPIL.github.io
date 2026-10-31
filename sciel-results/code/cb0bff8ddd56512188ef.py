import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
H = np.sqrt(3) / 2.0

# Pre-compute triangle indices
_tri_i = []
_tri_j = []
_tri_k = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _tri_i.append(i)
            _tri_j.append(j)
            _tri_k.append(k)
_tri_i = np.array(_tri_i)
_tri_j = np.array(_tri_j)
_tri_k = np.array(_tri_k)
N_TRI = len(_tri_i)

def _all_areas(pts):
    """Vectorized computation of all triangle areas."""
    pi = pts[_tri_i]
    pj = pts[_tri_j]
    pk = pts[_tri_k]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    return 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)

def _min_area(pts):
    """Minimum area of any triangle formed by three points."""
    return float(np.min(_all_areas(pts)))

def _soft_min_area(pts, beta=100.0):
    """Smooth approximation of min area."""
    areas = _all_areas(pts)
    m = np.min(areas)
    shifted = -beta * (areas - m)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return m - log_sum / beta

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle using barycentric coords."""
    x, y = p
    c = 2 * y / np.sqrt(3)
    b = x - 0.5 * c
    a = 1 - b - c
    eps = 1e-9
    a = max(eps, min(1 - 2 * eps, a))
    b = max(eps, min(1 - 2 * eps, b))
    c = max(eps, min(1 - 2 * eps, c))
    s = a + b + c
    a /= s
    b /= s
    c /= s
    return a * V0 + b * V1 + c * V2

def _clamp_all(pts):
    """Clamp all points to triangle."""
    return np.array([_clamp_to_triangle(p) for p in pts])

def _objective_smooth(flat_pts, beta=100.0):
    """Objective to minimize: negative soft-min area."""
    pts = flat_pts.reshape(-1, 2)
    pts = _clamp_all(pts)
    return -_soft_min_area(pts, beta)

def _objective_hard(flat_pts):
    """Objective to minimize: negative min area."""
    pts = flat_pts.reshape(-1, 2)
    pts = _clamp_all(pts)
    return -_min_area(pts)

def _initial_placement_hexagonal():
    """Hexagonal lattice initial placement with 3-fold symmetry."""
    pts = []
    # Center
    cx, cy = 1/3, H/3
    pts.append([cx, cy])
    
    # Ring 1: 6 points
    r1 = 0.15
    for angle in np.linspace(0, 2*np.pi, 7)[:-1]:
        px = cx + r1 * np.cos(angle)
        py = cy + r1 * np.sin(angle)
        pts.append([px, py])
    
    # Ring 2: 4 points (to make 11 total)
    r2 = 0.28
    for angle in [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]:
        px = cx + r2 * np.cos(angle)
        py = cy + r2 * np.sin(angle)
        pts.append([px, py])
    
    return _clamp_all(np.array(pts))

def _initial_placement_structured():
    """Structured initial placement with 3-fold symmetry."""
    pts = []
    m = 0.10
    pts.append(m * V0 + (1 - m) / 2 * V1 + (1 - m) / 2 * V2)
    pts.append((1 - m) / 2 * V0 + m * V1 + (1 - m) / 2 * V2)
    pts.append((1 - m) / 2 * V0 + (1 - m) / 2 * V1 + m * V2)
    m2 = 0.12
    pts.append(0.5 * (1 - m2) * V0 + 0.5 * m2 * V1 + 0.5 * (1 - m2) * V2)
    pts.append(m2 * V0 + 0.5 * (1 - m2) * V1 + 0.5 * (1 - m2) * V2)
    pts.append(0.5 * (1 - m2) * V0 + 0.5 * (1 - m2) * V1 + m2 * V2)
    pts.append((V0 + V1 + V2) / 3)
    cx, cy = 1 / 3, H / 3
    r = 0.14
    for angle in [np.pi / 6, 5 * np.pi / 6, 7 * np.pi / 6, 11 * np.pi / 6]:
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        pts.append([px, py])
    return _clamp_all(np.array(pts))

def _initial_placement_grid():
    """Triangular grid initial placement."""
    pts = []
    y0 = 0.06
    for i in range(4):
        x = 0.06 + i * (0.88 / 3)
        pts.append([x, y0])
    y1 = 0.34
    for i in range(4):
        x = 0.14 + i * (0.72 / 3)
        pts.append([x, y1])
    y2 = 0.60
    for i in range(3):
        x = 0.24 + i * (0.52 / 2)
        pts.append([x, y2])
    return _clamp_all(np.array(pts))

def _initial_placement_random(seed):
    """Random initial placement inside triangle."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(N):
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        p = r1 * V1 + r2 * V2
        pts.append(p)
    return _clamp_all(np.array(pts))

def _optimize_from_start(start_pts, beta, maxiter):
    """Run L-BFGS-B optimization from a starting point."""
    try:
        res = minimize(
            lambda x, b=beta: _objective_smooth(x, b),
            start_pts.flatten(),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        opt_pts = _clamp_all(res.x.reshape(-1, 2))
        return opt_pts, _min_area(opt_pts)
    except Exception:
        return None, -1.0

def _nelder_mead_polish(pts):
    """Nelder-Mead polish with hard objective."""
    try:
        res = minimize(
            _objective_hard,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
        )
        nm_pts = _clamp_all(res.x.reshape(-1, 2))
        return nm_pts, _min_area(nm_pts)
    except Exception:
        return pts, _min_area(pts)

def _coordinate_descent_polish(pts, n_iters=500):
    """Coordinate descent polish: optimize each point one at a time."""
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    current = pts.copy()
    current_score = _min_area(current)
    
    for step in step_sizes:
        for _ in range(n_iters // len(step_sizes)):
            improved = False
            for i in range(N):
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        trial = current.copy()
                        trial[i] = _clamp_to_triangle(current[i] + np.array([dx, dy]))
                        trial_score = _min_area(trial)
                        if trial_score > current_score + 1e-15:
                            current = trial
                            current_score = trial_score
                            improved = True
            if not improved:
                break
    return current, current_score

def construct():
    start_time = time.time()
    deadline = 145.0

    def time_left():
        return deadline - (time.time() - start_time)

    # Generate diverse initializations
    init_candidates = []
    init_candidates.append(_initial_placement_hexagonal())
    init_candidates.append(_initial_placement_structured())
    init_candidates.append(_initial_placement_grid())
    for seed in [1, 7, 13, 29, 42, 100, 200, 300, 400, 500, 600, 700, 800, 900]:
        init_candidates.append(_initial_placement_random(seed))

    best_pts = None
    best_score = -1.0

    # Phase 1: Multi-start smooth optimization with varying beta
    for init in init_candidates:
        if time_left() < 10.0:
            break
        for beta in [50.0, 100.0, 200.0]:
            if time_left() < 6.0:
                break
            opt_pts, opt_score = _optimize_from_start(init, beta, 1200)
            if opt_pts is not None and opt_score > best_score:
                best_pts = opt_pts.copy()
                best_score = opt_score

    if best_pts is None:
        best_pts = _initial_placement_hexagonal()
        best_score = _min_area(best_pts)

    # Phase 2: Perturbation-based multi-start from best
    rng = np.random.default_rng(12345)
    n_perturb = 20
    for trial in range(n_perturb):
        if time_left() < 8.0:
            break
        scale = 0.012 + 0.006 * trial
        trial_pts = best_pts + rng.normal(0, scale, size=(N, 2))
        trial_pts = _clamp_all(trial_pts)
        
        for beta in [80.0, 160.0]:
            if time_left() < 5.0:
                break
            opt_pts, opt_score = _optimize_from_start(trial_pts, beta, 800)
            if opt_pts is not None and opt_score > best_score:
                best_pts = opt_pts.copy()
                best_score = opt_score

    # Phase 3: Polish the best candidate with Nelder-Mead
    if time_left() > 5.0:
        polished_pts, polished_score = _nelder_mead_polish(best_pts)
        if polished_score > best_score:
            best_pts = polished_pts
            best_score = polished_score

    # Phase 4: Coordinate descent for fine-tuning
    if time_left() > 2.0:
        cd_pts, cd_score = _coordinate_descent_polish(best_pts, n_iters=200)
        if cd_score > best_score:
            best_pts = cd_pts
            best_score = cd_score

    # Final safety check: ensure points are inside
    if not np.all(_clamp_all(best_pts) == best_pts):
        best_pts = _clamp_all(best_pts)

    return best_pts
