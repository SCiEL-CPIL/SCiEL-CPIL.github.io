# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_TOL = 1e-7


def _is_feasible(centers, sides, tol=_TOL):
    """Check all constraints: inside unit square, positive sides, no overlap."""
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
            return False
        if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True


def _penalty(centers, sides, beta):
    """Smooth penalty for constraint violations (overlap + boundary)."""
    n = len(sides)
    p = 0.0
    # Boundary penalties (soft)
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        v_left = max(0.0, s / 2 - cx)
        v_right = max(0.0, cx + s / 2 - 1.0)
        v_bottom = max(0.0, s / 2 - cy)
        v_top = max(0.0, cy + s / 2 - 1.0)
        p += beta * (v_left**2 + v_right**2 + v_bottom**2 + v_top**2)
    # Overlap penalties (smooth)
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - _TOL
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            ox = max(0.0, t - dx)
            oy = max(0.0, t - dy)
            p += beta * (ox * oy / max(t, 1e-12)) ** 2
    return p


def _objective(x, beta):
    n = 10
    centers = x[:2 * n].reshape(n, 2)
    sides = x[2 * n:]
    side_pen = beta * np.sum(np.maximum(0, -sides) ** 2)
    pen = _penalty(centers, sides, beta)
    return -np.sum(sides) + pen + side_pen


def _make_init_1():
    """Layout: 5 bottom + 5 top with varying sizes."""
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    # Bottom row: 5 squares of side 0.2
    for i in range(5):
        centers[i] = [0.2 * i + 0.1, 0.1]
        sides[i] = 0.2
    # Top region
    centers[5] = [0.2, 0.4]
    sides[5] = 0.4
    centers[6] = [0.6, 0.4]
    sides[6] = 0.4
    centers[7] = [0.9, 0.4]
    sides[7] = 0.2
    centers[8] = [0.2, 0.8]
    sides[8] = 0.4
    centers[9] = [0.6, 0.8]
    sides[9] = 0.4
    return centers, sides


def _make_init_2():
    """Layout: asymmetric with one large square."""
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    # Large square bottom-left: 0.5
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    # Right of it: [0.5,1]x[0,0.5]
    # Two 0.25 squares stacked
    centers[1] = [0.75, 0.125]
    sides[1] = 0.25
    centers[2] = [0.75, 0.375]
    sides[2] = 0.25
    # Top-left: [0,0.5]x[0.5,1]
    # Two 0.25 squares stacked
    centers[3] = [0.125, 0.625]
    sides[3] = 0.25
    centers[4] = [0.375, 0.625]
    sides[4] = 0.25
    # Top-right: [0.5,1]x[0.5,1]
    # One 0.5 square
    centers[5] = [0.75, 0.75]
    sides[5] = 0.5
    # Fill remaining with small squares
    centers[6] = [0.125, 0.875]
    sides[6] = 0.25
    centers[7] = [0.375, 0.875]
    sides[7] = 0.25
    centers[8] = [0.625, 0.875]
    sides[8] = 0.25
    centers[9] = [0.875, 0.875]
    sides[9] = 0.25
    return centers, sides


def _make_init_3():
    """Layout: 3x3 grid plus one extra."""
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    # 3x3 grid of 1/3 squares, but we need 10
    # Use 3 columns, with varying row heights
    # Column 0: 4 squares
    for i in range(4):
        centers[i] = [1/6, 1/8 + i * 1/4]
        sides[i] = 0.25
    # Column 1: 3 squares
    for i in range(3):
        centers[4+i] = [1/2, 1/6 + i * 1/3]
        sides[4+i] = 1/3
    # Column 2: 3 squares
    for i in range(3):
        centers[7+i] = [5/6, 1/6 + i * 1/3]
        sides[7+i] = 1/3
    return centers, sides


def _make_init_4():
    """Layout: 2 large + 8 small."""
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    # Two 0.4 squares
    centers[0] = [0.2, 0.2]
    sides[0] = 0.4
    centers[1] = [0.8, 0.8]
    sides[1] = 0.4
    # 8 small squares of side 0.2 filling the rest
    positions = [
        [0.6, 0.2], [0.2, 0.6], [0.6, 0.6], [0.2, 0.8],
        [0.8, 0.2], [0.4, 0.4], [0.4, 0.8], [0.8, 0.4]
    ]
    for i, pos in enumerate(positions):
        centers[2+i] = pos
        sides[2+i] = 0.2
    return centers, sides


def _make_init_5():
    """Layout: based on known competitive packing."""
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    # Try a layout inspired by optimal packings
    # One 0.35, two 0.3, rest smaller
    centers[0] = [0.175, 0.175]
    sides[0] = 0.35
    centers[1] = [0.525, 0.175]
    sides[1] = 0.3
    centers[2] = [0.85, 0.175]
    sides[2] = 0.3
    centers[3] = [0.175, 0.525]
    sides[3] = 0.35
    centers[4] = [0.525, 0.525]
    sides[4] = 0.3
    centers[5] = [0.85, 0.525]
    sides[5] = 0.3
    centers[6] = [0.175, 0.85]
    sides[6] = 0.3
    centers[7] = [0.525, 0.85]
    sides[7] = 0.3
    centers[8] = [0.85, 0.85]
    sides[8] = 0.3
    centers[9] = [0.5, 0.35]
    sides[9] = 0.2
    return centers, sides


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem for n=10.
    Uses smooth penalty optimization with multiple structural initializations.
    """
    deadline = time.time() + 580  # safety margin
    
    inits = [
        _make_init_1(),
        _make_init_2(),
        _make_init_3(),
        _make_init_4(),
        _make_init_5(),
    ]
    
    best_centers = None
    best_sides = None
    best_score = 0.0
    
    for centers0, sides0 in inits:
        if time.time() > deadline:
            break
        
        x0 = np.concatenate([centers0.ravel(), sides0])
        
        # Multi-restart with increasing beta
        for beta in [1.0, 5.0, 20.0, 100.0, 500.0]:
            if time.time() > deadline:
                break
            try:
                res = minimize(
                    _objective, x0, args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                x_opt = res.x
                centers_opt = x_opt[:20].reshape(10, 2)
                sides_opt = x_opt[20:]
                
                # Clip to valid range
                centers_opt = np.clip(centers_opt, 0, 1)
                sides_opt = np.clip(sides_opt, 1e-6, 1.0)
                
                # Check feasibility
                if _is_feasible(centers_opt, sides_opt):
                    score = np.sum(sides_opt)
                    if score > best_score:
                        best_score = score
                        best_centers = centers_opt.copy()
                        best_sides = sides_opt.copy()
                else:
                    # Try to repair: reduce sides slightly
                    sides_try = sides_opt * 0.95
                    if _is_feasible(centers_opt, sides_try):
                        score = np.sum(sides_try)
                        if score > best_score:
                            best_score = score
                            best_centers = centers_opt.copy()
                            best_sides = sides_try.copy()
                
                x0 = x_opt  # warm start for next beta
            except Exception:
                pass
        
        # Final polish from best found so far
        if best_centers is not None and time.time() < deadline:
            for _ in range(3):
                if time.time() > deadline:
                    break
                x0 = np.concatenate([best_centers.ravel(), best_sides])
                try:
                    res = minimize(
                        _objective, x0, args=(1000.0,),
                        method='L-BFGS-B',
                        options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
                    )
                    x_opt = res.x
                    centers_opt = np.clip(x_opt[:20].reshape(10, 2), 0, 1)
                    sides_opt = np.clip(x_opt[20:], 1e-6, 1.0)
                    if _is_feasible(centers_opt, sides_opt):
                        score = np.sum(sides_opt)
                        if score > best_score:
                            best_score = score
                            best_centers = centers_opt.copy()
                            best_sides = sides_opt.copy()
                except Exception:
                    break
    
    # Ensure final result is valid
    if best_centers is None or not _is_feasible(best_centers, best_sides):
        # Fallback to grid
        k = 4
        side = 1.0 / k
        idx = np.arange(10)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(10, side, dtype=np.float64)
    
    return best_centers.astype(np.float64), best_sides.astype(np.float64)


# EVOLVE-BLOCK-END
