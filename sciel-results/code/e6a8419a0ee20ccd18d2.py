"""Circle packing in unit square: pack 26 disjoint circles, maximize sum of radii.

Combines a force-based physics optimizer with diverse restarts and a final L-BFGS-B
polish pass for refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
_START = time.time()
DEADLINE = _START + 116.0  # safety margin before 150s limit
RNG = np.random.default_rng(20240517)

# --- force parameters ---
K_WALL = 500.0
K_SEP = 40.0
K_R = 10.0
EPS = 1e-9
STEP = 0.03
R_MIN = 1e-6
R_MAX = 0.5

# Smoother separation force parameters
SOFT_MIN_TEMP = 0.01  # Temperature for soft-min in overlap penalty


def _is_valid(c, tol=1e-6):
    """Return True if packing is inside [0,1]^2 and circles are disjoint."""
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    if r.min() < R_MIN:
        return False
    if (x.min() < -tol) or (x.max() > 1 + tol) or (y.min() < -tol) or (y.max() > 1 + tol):
        return False
    if np.any(x + r > 1 + tol) or np.any(x - r < -tol):
        return False
    if np.any(y + r > 1 + tol) or np.any(y - r < -tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - tol:
                return False
    return True


def _score(c):
    return float(np.sum(c[:, 2])) if _is_valid(c) else -np.inf


def _clamp(c):
    """Clamp positions/radii to a feasible box (keeps circles inside square)."""
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    r = np.clip(r, R_MIN, R_MAX)
    x = np.clip(x, r, 1.0 - r)
    y = np.clip(y, r, 1.0 - r)
    return np.column_stack([x, y, r])


def _forces(c):
    """Compute forces on each circle: wall repulsion, separation, radius growth.
    
    Uses smooth soft-min penalty for overlaps to avoid discontinuities.
    """
    x = c[:, 0].copy(); y = c[:, 1].copy(); r = c[:, 2].copy()
    fx = np.zeros(N); fy = np.zeros(N); fr = np.zeros(N)

    # wall forces: smooth penalty when circle touches wall
    # Use smooth min: -T * log(1 + exp(-a * (x - r))) approximates max(0, r - x) smoothly
    a_wall = 50.0
    left_pen = -SOFT_MIN_TEMP * np.log1p(np.exp(-a_wall * (x - r)))
    right_pen = -SOFT_MIN_TEMP * np.log1p(np.exp(-a_wall * (1 - x - r)))
    bottom_pen = -SOFT_MIN_TEMP * np.log1p(np.exp(-a_wall * (y - r)))
    top_pen = -SOFT_MIN_TEMP * np.log1p(np.exp(-a_wall * (1 - y - r)))
    
    # Derivatives for smooth forces
    fx -= K_WALL * a_wall * SOFT_MIN_TEMP * np.exp(-a_wall * (x - r)) / (1 + np.exp(-a_wall * (x - r)))
    fx += K_WALL * a_wall * SOFT_MIN_TEMP * np.exp(-a_wall * (1 - x - r)) / (1 + np.exp(-a_wall * (1 - x - r)))
    fy -= K_WALL * a_wall * SOFT_MIN_TEMP * np.exp(-a_wall * (y - r)) / (1 + np.exp(-a_wall * (y - r)))
    fy += K_WALL * a_wall * SOFT_MIN_TEMP * np.exp(-a_wall * (1 - y - r)) / (1 + np.exp(-a_wall * (1 - y - r)))
    fr += K_WALL * (left_pen + right_pen + bottom_pen + top_pen)

    # separation forces: smooth soft-min penalty for overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]; dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            s = r[i] + r[j]
            if d < s * 1.05:  # Only compute for near/overlapping pairs
                if d < 1e-9:
                    dx = RNG.standard_normal(); dy = RNG.standard_normal()
                    d = np.hypot(dx, dy)
                ux = dx / d; uy = dy / d
                # Smooth overlap penalty: -T * log(1 + exp(-a * (d - s)))
                a_sep = 30.0
                overlap = s - d
                if overlap > 0:
                    f = K_SEP * a_sep * SOFT_MIN_TEMP * np.exp(-a_sep * overlap) / (1 + np.exp(-a_sep * overlap))
                    fx[i] += f * ux; fy[i] += f * uy
                    fx[j] -= f * ux; fy[j] -= f * uy
                    fr[i] += f * 0.5; fr[j] += f * 0.5

    # maximize radii
    fr += K_R
    return fx, fy, fr


def _pbd_step(c):
    """One force-based step with adaptive step size and clamping."""
    c = _clamp(c)
    fx, fy, fr = _forces(c)

    # adaptive step: smaller when forces are large
    fmag = np.hypot(fx, fy)
    step = STEP * 0.5 / (1.0 + 0.02 * fmag.max())

    x = c[:, 0] + step * fx
    y = c[:, 1] + step * fy
    r = c[:, 2] + step * fr

    return np.column_stack([x, y, r])


def _optimize(c0):
    """Force-based optimizer with adaptive step size and time-aware stopping."""
    c = c0.copy()
    best = _score(c)
    best_c = c.copy()
    step = STEP
    it = 0
    no_improve_count = 0
    max_no_improve = 200  # Stop if no improvement for 200 iterations
    
    while it < 8000:  # Increased max iterations, but time-limited
        c = _pbd_step(c)
        val = _score(c)
        if val > best:
            best = val
            best_c = c.copy()
            step = min(step * 1.12, 0.15)
            no_improve_count = 0
        else:
            step *= 0.85
            no_improve_count += 1
            if it > 10:
                c = best_c.copy()
            if step < 1e-5 or no_improve_count > max_no_improve:
                break
        it += 1
        if time.time() > DEADLINE - 2.0:
            break
    return best_c, best


def _seed_grid():
    """Grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])


def _seed_random():
    """Random initialization with small radii."""
    x = RNG.uniform(0.08, 0.92, N)
    y = RNG.uniform(0.08, 0.92, N)
    r = RNG.uniform(0.02, 0.08, N)
    return np.column_stack([x, y, r])


def _seed_symmetric():
    """Symmetric corner initialization."""
    # Place circles near corners and center
    positions = []
    radii = []
    # Corners
    for cx, cy in [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]:
        positions.append((cx, cy))
        radii.append(0.12)
    # Center
    positions.append((0.5, 0.5))
    radii.append(0.15)
    # Fill remaining with random
    while len(positions) < N:
        x = RNG.uniform(0.1, 0.9)
        y = RNG.uniform(0.1, 0.9)
        r = RNG.uniform(0.05, 0.1)
        positions.append((x, y))
        radii.append(r)
    
    x = np.array([p[0] for p in positions[:N]])
    y = np.array([p[1] for p in positions[:N]])
    r = np.array(radii[:N])
    return np.column_stack([x, y, r])


def _lbfgs_polish(c0):
    """Final L-BFGS-B polish pass with smooth soft-min penalties and tighter tolerances."""
    if time.time() > DEADLINE - 8.0:
        return c0, _score(c0)
    
    # Convert to optimization parameters
    x0 = c0.flatten()
    
    def _objective(params):
        c = params.reshape(-1, 3)
        x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
        
        # Smooth penalties using soft-min
        T = 0.005  # Small temperature for near-exact behavior
        a = 100.0  # Sharpness parameter
        
        penalty = 0.0
        # Wall violations (smooth)
        penalty += np.sum(-T * np.log1p(np.exp(-a * (x - r))))
        penalty += np.sum(-T * np.log1p(np.exp(-a * (1 - x - r))))
        penalty += np.sum(-T * np.log1p(np.exp(-a * (y - r))))
        penalty += np.sum(-T * np.log1p(np.exp(-a * (1 - y - r))))
        
        # Overlap violations (smooth)
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]; dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                s = r[i] + r[j]
                if d < s * 1.1:
                    penalty += -T * np.log1p(np.exp(-a * (d - s)))
        
        # Negative sum of radii (minimize) + large penalty weight
        return -np.sum(r) + 100.0 * penalty
    
    try:
        result = minimize(
            _objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 20000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        c_polished = result.x.reshape(-1, 3)
        c_polished = _clamp(c_polished)
        
        # Only accept if valid
        if _is_valid(c_polished, tol=1e-6):
            score = _score(c_polished)
            if score > _score(c0):
                return c_polished, score
    except Exception:
        pass
    
    return c0, _score(c0)


def construct():
    t0 = time.time()
    
    # Start with grid initialization
    c_grid = _seed_grid()
    best_c, best_val = _optimize(c_grid)
    
    # Try diverse random starts - reduced count to allow more polish time
    n_random = 20
    for i in range(n_random):
        if time.time() > DEADLINE - 15.0:
            break
        
        if i % 8 == 0:
            c0 = _seed_symmetric()
        else:
            c0 = _seed_random()
        
        cc, val = _optimize(c0)
        if val > best_val:
            best_val, best_c = val, cc
    
    # Multiple polish passes for refinement
    # First polish
    if time.time() < DEADLINE - 10.0:
        cc, val = _lbfgs_polish(best_c)
        if val > best_val:
            best_val, best_c = val, cc
    
    # Second polish pass on the improved result
    if time.time() < DEADLINE - 5.0:
        cc, val = _lbfgs_polish(best_c)
        if val > best_val:
            best_val, best_c = val, cc
    
    # Fallback: seed grid is always valid
    if best_val <= 0:
        return c_grid
    
    return best_c
