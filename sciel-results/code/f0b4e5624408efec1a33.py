"""Rewritten program for Heilbronn convex regions n=14."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area(pts):
    """Compute minimum triangle area over all triples."""
    n = len(pts)
    min_a = float('inf')
    # Vectorized: for each pair (i,j), compute cross product with all other points
    for i in range(n):
        for j in range(i + 1, n):
            v = pts[j] - pts[i]
            # cross product magnitude with all other points
            others = np.delete(pts, [i, j], axis=0)
            cross = np.abs(np.cross(np.tile(v, (len(others), 1)), others))
            a = 0.5 * cross.min()
            if a < min_a:
                min_a = a
                if min_a < 1e-12:
                    return 0.0
    return min_a


def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        h = ConvexHull(pts)
        return h.volume
    except Exception:
        # Fallback: shoelace on all points (not correct for non-convex, but safe)
        # Actually, let's use a simpler approach
        return 1.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    cha = _convex_hull_area(pts)
    if cha < 1e-12:
        return 0.0
    mta = _min_triangle_area(pts)
    return mta / cha


def _heptagon_with_offsets(params):
    """
    Generate 14 points from a regular heptagon structure.
    params: 7 radial offsets (one per vertex pair)
    Each vertex of the heptagon is replaced by two points separated radially.
    """
    n_hept = 7
    angles = 2 * np.pi * np.arange(n_hept) / n_hept
    # Base radius = 1
    base_r = 1.0
    pts = np.zeros((2 * n_hept, 2))
    for k in range(n_hept):
        a = angles[k]
        offset = params[k]
        # Two points: one at radius base_r + offset, one at base_r - offset
        r1 = base_r + offset
        r2 = base_r - offset
        # Small angular separation to avoid overlap
        da = 0.05  # small angular offset
        a1 = a - da
        a2 = a + da
        pts[2 * k] = [r1 * np.cos(a1), r1 * np.sin(a1)]
        pts[2 * k + 1] = [r2 * np.cos(a2), r2 * np.sin(a2)]
    return pts


def _affine_transform(pts, a, b, c, d, tx, ty):
    """Apply affine transformation: [x,y] -> [a*x+b*y+tx, c*x+d*y+ty]"""
    M = np.array([[a, b], [c, d]])
    t = np.array([tx, ty])
    return pts @ M.T + t


def _objective_full(x):
    """
    Full objective: 7 radial offsets + 6 affine params (a,b,c,d,tx,ty)
    Maximize min_triangle_area / convex_hull_area
    """
    offsets = x[:7]
    a, b, c, d, tx, ty = x[7:13]
    pts = _heptagon_with_offsets(offsets)
    pts = _affine_transform(pts, a, b, c, d, tx, ty)
    s = _score(pts)
    return -s  # minimize negative score


def _objective_offsets_only(x):
    """Objective with fixed affine (identity), only optimize offsets."""
    pts = _heptagon_with_offsets(x)
    s = _score(pts)
    return -s


def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # 110s safety margin before 120s limit

    best_pts = None
    best_score = -float('inf')

    # Initial: regular heptagon with zero offsets
    offsets0 = np.zeros(7)
    pts0 = _heptagon_with_offsets(offsets0)
    s0 = _score(pts0)
    if s0 > best_score:
        best_score = s0
        best_pts = pts0.copy()

    # Phase 1: Optimize radial offsets with identity affine
    from scipy.optimize import differential_evolution

    bounds_off = [(-0.3, 0.3)] * 7

    def elapsed_ok():
        return time.time() < deadline - 20.0  # leave 20s for final phases

    # Try differential evolution on offsets
    if elapsed_ok():
        try:
            res = differential_evolution(
                _objective_offsets_only,
                bounds=bounds_off,
                seed=42,
                maxiter=50,
                tol=1e-6,
                polish=True,
                workers=1
            )
            if res.success or res.fun < _objective_offsets_only(offsets0):
                offsets_opt = res.x
                pts_opt = _heptagon_with_offsets(offsets_opt)
                s_opt = _score(pts_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_pts = pts_opt.copy()
        except Exception:
            pass

    # Phase 2: Optimize full parameters (offsets + affine)
    if elapsed_ok():
        # Start from best offsets + identity affine
        if best_pts is not None:
            # Recover offsets from best_pts (approximate)
            # Actually, let's just use the optimized offsets
            pass

        full_bounds = [(-0.3, 0.3)] * 7 + [(-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2)]

        # Use L-BFGS-B for local optimization from multiple starts
        starts = [
            np.concatenate([np.zeros(7), [1, 0, 0, 1, 0, 0]]),  # identity
            np.concatenate([np.zeros(7), [1, 0, 0, 1, 0.5, 0.5]]),
            np.concatenate([np.zeros(7), [0.8, 0.1, -0.1, 0.9, 0, 0]]),
            np.concatenate([np.zeros(7), [1.2, -0.1, 0.1, 0.8, 0, 0]]),
        ]

        for start in starts:
            if not elapsed_ok():
                break
            try:
                res = minimize(
                    _objective_full,
                    start,
                    method='L-BFGS-B',
                    bounds=full_bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.fun < -best_score:
                    x_opt = res.x
                    pts_opt = _heptagon_with_offsets(x_opt[:7])
                    pts_opt = _affine_transform(pts_opt, x_opt[7], x_opt[8], x_opt[9], x_opt[10], x_opt[11], x_opt[12])
                    s_opt = _score(pts_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_pts = pts_opt.copy()
            except Exception:
                pass

    # Phase 3: Nelder-Mead polish on full parameters
    if elapsed_ok() and best_pts is not None:
        # Reconstruct parameter vector from best_pts
        # This is tricky; instead, use the last known good x
        # Let's just do a local Nelder-Mead from the best full param
        # We need to track the best x_full
        # For now, try a few random perturbations around identity
        for trial in range(5):
            if not elapsed_ok():
                break
            x_try = np.concatenate([
                np.random.uniform(-0.1, 0.1, 7),
                np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]) + np.random.uniform(-0.2, 0.2, 6)
            ])
            try:
                res = minimize(
                    _objective_full,
                    x_try,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res.fun < -best_score:
                    x_opt = res.x
                    pts_opt = _heptagon_with_offsets(x_opt[:7])
                    pts_opt = _affine_transform(pts_opt, x_opt[7], x_opt[8], x_opt[9], x_opt[10], x_opt[11], x_opt[12])
                    s_opt = _score(pts_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_pts = pts_opt.copy()
            except Exception:
                pass

    # Phase 4: Final high-precision polish
    if elapsed_ok() and best_pts is not None:
        # Try to refine with tighter tolerances
        # Use the best x_full if we tracked it, otherwise reconstruct
        # For simplicity, do a local search around current best
        pass

    if best_pts is None:
        # Fallback: return the initial heptagon
        return _heptagon_with_offsets(np.zeros(7))

    return best_pts

# EVOLVE-BLOCK-END
