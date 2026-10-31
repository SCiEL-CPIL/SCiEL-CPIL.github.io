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

def _penalty_objective(x, mu):
    """
    Objective: minimize -t / hull_area + mu * sum(max(0, t - area_i)^2)
    x = [t, x1, y1, ..., xN, yN]
    """
    t = x[0]
    pts = x[1:].reshape(N, 2)
    
    # Ensure points are roughly centered to help numerical stability? 
    # Actually, the score is invariant to translation, but the penalty depends on absolute areas.
    # Centering might help L-BFGS-B. Let's center inside the objective? No, that breaks gradient.
    # We will rely on the optimizer to find a good position.
    
    areas = _triangle_areas_vectorized(pts)
    hull_a = _hull_area(pts)
    
    if hull_a < 1e-12:
        return 1e10
        
    # If t is negative, we don't really care about the penalty, but we want to maximize t.
    # The term -t/hull_a drives t up.
    # Penalty drives areas up to t.
    
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations ** 2)
    
    return -t / hull_a + mu * penalty

def _penalty_optimize(points, deadline, mu_start=10.0, mu_end=1e6, n_mu_steps=8,
                      max_iter_per_mu=200, time_budget=None):
    """
    Penalty-based interior-point optimization.
    """
    best = points.copy()
    best_s = _score(best)
    
    # Initialize t to current min area
    areas = _triangle_areas_vectorized(best)
    t_init = float(np.min(areas)) * 0.9
    
    x0 = np.zeros(1 + 2 * N)
    x0[0] = t_init
    x0[1:] = best.flatten()
    
    mus = np.geomspace(mu_start, mu_end, n_mu_steps)
    
    for mu in mus:
        if time.time() > deadline - 2:
            break
        try:
            # Use a tight time check inside the loop if possible, but L-BFGS-B doesn't have a callback for time easily.
            # We rely on maxiter and the outer loop check.
            res = minimize(
                lambda x: _penalty_objective(x, mu),
                x0,
                method='L-BFGS-B',
                options={'maxiter': max_iter_per_mu, 'ftol': 1e-14, 'gtol': 1e-12},
            )
            x0 = res.x
            cand_pts = x0[1:].reshape(N, 2)
            s = _score(cand_pts)
            if s > best_s:
                best_s = s
                best = cand_pts.copy()
        except Exception:
            pass
    
    return best, best_s

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
    # Add some structure to radii to avoid degeneracy
    radii = 1.0 + rng.uniform(-0.2, 0.2, N)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

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
            
    # 1. Generate diverse seeds
    seeds = []
    
    # Regular
    seeds.append(_seed_regular())
    
    # Quasi-regular variations
    for s in range(10):
        seeds.append(_seed_quasi_regular(seed=s, ang_scale=0.1, rad_scale=0.1))
        
    # Two-ring variations
    for n_out in [5, 6, 7]:
        for r_in in [0.5, 0.7]:
            seeds.append(_seed_two_ring(n_out, r_in, np.pi / n_out))
            
    # Random convex
    for s in range(10):
        seeds.append(_seed_random_convex(seed=s))
        
    # Evaluate seeds
    for seed in seeds:
        _update(seed)
        
    if best_points is None:
        best_points = _seed_regular()
        best_score = _score(best_points)
        
    # 2. Local optimization from top seeds
    # Sort seeds by score
    seed_scores = [(seed, _score(seed)) for seed in seeds]
    seed_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Optimize top 5 seeds with penalty method
    # Time budget per seed: ~20s total for this phase, so 4s each?
    # Let's do 10 seeds, 10s each? No, L-BFGS-B is fast.
    # Let's allocate 25s for this phase.
    phase_deadline = start_time + 25.0
    
    for i in range(min(10, len(seed_scores))):
        if time.time() >= phase_deadline:
            break
        seed_pts, seed_s = seed_scores[i]
        try:
            # Use a tighter time budget for individual optimizations
            # We can't easily pass a time limit to minimize, but we can limit iterations.
            # 200 iterations per mu step, 8 mu steps = 1600 function evals.
            # Each eval is fast (vectorized).
            pts, s = _penalty_optimize(seed_pts, deadline, 
                                       mu_start=10.0, mu_end=1e5, n_mu_steps=6,
                                       max_iter_per_mu=150)
            _update(pts)
        except Exception:
            pass
            
    # 3. Refine the best candidate found so far with higher precision
    if best_points is not None and time.time() < deadline - 30:
        try:
            # Higher mu_end and more steps
            pts, s = _penalty_optimize(best_points, deadline,
                                       mu_start=100.0, mu_end=1e8, n_mu_steps=10,
                                       max_iter_per_mu=300)
            _update(pts)
        except Exception:
            pass
            
    # 4. Final polish with Nelder-Mead on true objective
    if best_points is not None and time.time() < deadline - 5:
        try:
            pts, s = _polish_true_objective(best_points, deadline, max_iter=2000)
            _update(pts)
        except Exception:
            pass
            
    # 5. One last check with a different local search strategy if time permits
    # Try a simple coordinate descent on the true objective?
    # Or just another Nelder-Mead restart?
    if best_points is not None and time.time() < deadline - 2:
        try:
             # Perturb slightly and polish again to escape potential local minima
            rng = np.random.RandomState(123)
            perturbed = best_points + rng.normal(0, 0.01, (N, 2))
            pts, s = _polish_true_objective(perturbed, deadline, max_iter=500)
            _update(pts)
        except Exception:
            pass

    if best_points is None:
        best_points = _seed_regular()
        
    return best_points
