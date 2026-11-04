"""Heilbronn convex regions: maximize min triangle area / hull area for 13 points."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 13

# Precompute all triangle index combinations
_TRI_IDX = np.array(list(itertools.combinations(range(N), 3)))

def _hull_area(pts):
    """Calculate the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _all_triangle_areas(pts):
    """Compute areas of all C(N,3) triangles."""
    i, j, k = _TRI_IDX[:, 0], _TRI_IDX[:, 1], _TRI_IDX[:, 2]
    pi = pts[i]
    pj = pts[j]
    pk = pts[k]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _min_tri_area(pts):
    """Minimum triangle area."""
    return np.min(_all_triangle_areas(pts))

def _score(pts):
    """Score: min triangle area / hull area."""
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 0.0
    mta = _min_tri_area(pts)
    return mta / ha

def _soft_min_score(pts, temperature=0.01):
    """
    Smooth surrogate: soft-min of triangle areas divided by hull area.
    Uses log-sum-exp for numerical stability.
    """
    ha = _hull_area(pts)
    if ha < 1e-8:
        return -1e6
    
    areas = _all_triangle_areas(pts)
    # Soft min: -T * log(sum(exp(-a/T)))
    # For small T, this approximates the true minimum
    min_val = np.min(areas)
    shifted = areas - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / temperature))) + min_val / temperature
    soft_min = min_val - temperature * log_sum
    
    return soft_min / ha

def _neg_soft_score(pts, temperature=0.01):
    """Negative soft score for minimization."""
    return -_soft_min_score(pts, temperature)

def _neg_score(pts):
    """Negative exact score for minimization."""
    return -_score(pts)

def _regular_polygon():
    """Regular 13-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _halton_sequence(n, dim=2):
    """Generate Halton sequence points in [0,1]^dim."""
    def halton_one_index(i, base):
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    pts = np.zeros((n, dim))
    for i in range(1, n + 1):
        for d in range(dim):
            pts[i-1, d] = halton_one_index(i, d + 2)
    return pts

def _halton_points():
    """13 Halton sequence points."""
    return _halton_sequence(N)

def _centered_ring():
    """Points on a ring with slight radial variation."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = 1.0 + 0.05 * np.sin(3 * angles)
    return np.column_stack((r * np.cos(angles), r * np.sin(angles)))

def _jittered_grid():
    """Grid-like points with jitter."""
    rng = np.random.default_rng(42)
    x = np.linspace(-1, 1, 4) + rng.uniform(-0.1, 0.1, 4)
    y = np.linspace(-1, 1, 4) + rng.uniform(-0.1, 0.1, 4)
    grid = np.array([[xi, yi] for xi in x for yi in y])
    return grid[:13]

def _random_in_circle(seed):
    """Random points in unit circle."""
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.random(N))
    theta = rng.random(N) * 2 * np.pi
    return np.column_stack((r * np.cos(theta), r * np.sin(theta)))

def _polish_l_bfgs_b(pts, temperature, max_iter=500, deadline=None):
    """Polish using L-BFGS-B on the soft-min surrogate."""
    def obj(flat):
        p = flat.reshape(N, 2)
        return _neg_soft_score(p, temperature)
    
    def jac(flat):
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(flat)
        for i in range(len(flat)):
            flat_plus = flat.copy()
            flat_plus[i] += eps
            f_plus = obj(flat_plus)
            grad[i] = (f_plus - obj(flat)) / eps
        return grad
    
    try:
        res = minimize(
            obj,
            pts.flatten(),
            method='L-BFGS-B',
            jac=jac,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return pts

def _polish_nelder_mead(pts, max_iter=300):
    """Polish using Nelder-Mead on the exact score."""
    try:
        res = minimize(
            _neg_score,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_score = -np.inf
    best_pts = None
    
    def update_best(pts):
        nonlocal best_score, best_pts
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts
        return s
    
    def time_left():
        return deadline - time.time()
    
    # Generate diverse initial guesses
    guesses = [
        _regular_polygon(),
        _halton_points(),
        _centered_ring(),
        _jittered_grid(),
        _random_in_circle(1),
        _random_in_circle(2),
        _random_in_circle(3),
        _random_in_circle(4),
        _random_in_circle(5),
    ]
    
    # Also add some more structured guesses
    # Points at vertices of a regular polygon plus center
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ring = np.column_stack((np.cos(angles), np.sin(angles)))
    center = np.array([[0.0, 0.0]])
    guesses.append(np.vstack([ring, center]))
    
    # Two concentric rings
    angles1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    ring1 = np.column_stack((0.5 * np.cos(angles1), 0.5 * np.sin(angles1)))
    ring2 = np.column_stack((1.0 * np.cos(angles2), 1.0 * np.sin(angles2)))
    guesses.append(np.vstack([ring1, ring2]))
    
    # Optimize each guess with multiple strategies
    for guess in guesses:
        if time_left() < 10:
            break
        
        # Strategy 1: L-BFGS-B with soft-min (fast, good for smooth landscapes)
        if time_left() > 15:
            opt_pts = _polish_l_bfgs_b(guess, temperature=0.01, max_iter=200)
            update_best(opt_pts)
        
        # Strategy 2: Nelder-Mead on exact score (more robust to non-smoothness)
        if time_left() > 10:
            opt_pts = _polish_nelder_mead(opt_pts, max_iter=200)
            update_best(opt_pts)
        
        # Strategy 3: Second L-BFGS-B pass with smaller temperature for refinement
        if time_left() > 10:
            opt_pts = _polish_l_bfgs_b(opt_pts, temperature=0.005, max_iter=200)
            update_best(opt_pts)
        
        # Strategy 4: Final Nelder-Mead polish
        if time_left() > 5:
            opt_pts = _polish_nelder_mead(opt_pts, max_iter=200)
            update_best(opt_pts)
    
    # If we still have time, do a few more random restarts
    while time_left() > 15:
        seed = int(time.time() * 1000) % 100000
        guess = _random_in_circle(seed)
        
        opt_pts = _polish_l_bfgs_b(guess, temperature=0.01, max_iter=100)
        update_best(opt_pts)
        
        if time_left() > 10:
            opt_pts = _polish_nelder_mead(opt_pts, max_iter=150)
            update_best(opt_pts)
    
    # Fallback
    if best_pts is None:
        best_pts = _regular_polygon()
    
    return best_pts
