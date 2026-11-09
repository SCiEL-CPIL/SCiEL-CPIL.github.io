"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
SIDE = 1.0
HALF = 0.5
DEADLINE = 115.0  # safety margin before 120s kill

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def _score_from_centres(c):
    """Compute the radius score for given centres (N,2)."""
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check inside square
    if np.any(c < -HALF + 1e-12) or np.any(c > HALF - 1e-12):
        return 0.0
    # Pairwise distances
    d = pdist(c)
    min_pair = d.min() if d.size > 0 else 0.0
    # Distance to walls
    dist_walls = np.min(np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ]), axis=0)
    min_wall = dist_walls.min()
    r = min(min_pair / 2.0, min_wall)
    return r


def _negative_score(c_flat):
    """Objective to minimise (negative radius score)."""
    c = c_flat.reshape(N, 2)
    # Clamp to box for feasibility during optimisation
    c = np.clip(c, -HALF + 1e-9, HALF - 1e-9)
    r = _score_from_centres(c)
    return -r


def _grid_init():
    """Initial grid layout."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _random_init(rng):
    """Random initial layout."""
    return rng.uniform(-HALF, HALF, size=(N, 2))


def _hex_init():
    """Hexagonal lattice initialization for 50 points in unit square."""
    # Hexagonal packing: rows offset by half spacing
    # For 50 points, try a 7x8 or similar arrangement
    # Actually, let's use a proper hexagonal grid clipped to the square
    
    # Generate hexagonal points
    # Spacing for hexagonal: dx = 2*r, dy = r*sqrt(3)
    # But we don't know r yet, so let's just generate a dense hex grid
    # and take the first 50 that fit in [-0.5, 0.5]^2
    
    points = []
    # Hexagonal lattice: 
    # Row 0: x = 0, dx, 2*dx, ...
    # Row 1: x = dx/2, 3*dx/2, ...
    # dy = dx * sqrt(3)/2
    
    # Start with a reasonable spacing
    dx = 0.12
    dy = dx * np.sqrt(3) / 2
    
    row = 0
    while len(points) < N:
        offset = (dx / 2) if (row % 2 == 1) else 0
        x = -HALF + 0.05 + offset
        while x < HALF - 0.05:
            y = -HALF + 0.05 + row * dy
            if -HALF <= x <= HALF and -HALF <= y <= HALF:
                points.append([x, y])
            x += dx
        row += 1
    
    if len(points) >= N:
        return np.array(points[:N])
    else:
        # Fallback to grid
        return _grid_init()


def construct():
    t0 = time.time()
    best_centres = _grid_init()
    best_score = _score_from_centres(best_centres)

    rng = np.random.default_rng(42)

    # Try hexagonal initialization too
    hex_centres = _hex_init()
    hex_score = _score_from_centres(hex_centres)
    if hex_score > best_score:
        best_score = hex_score
        best_centres = hex_centres

    # Strategy 1: Local optimisation from multiple starts
    n_starts = 15
    for i in range(n_starts):
        if time.time() - t0 > DEADLINE - 35:
            break
        
        if i == 0:
            x0 = best_centres.ravel()
        elif i == 1:
            x0 = _hex_init().ravel()
        else:
            x0 = _random_init(rng).ravel()
            x0 = x0 + rng.normal(0, 0.01, size=x0.shape)

        try:
            res = minimize(
                _negative_score, x0,
                method='L-BFGS-B',
                bounds=[(-HALF + 1e-9, HALF - 1e-9)] * (2 * N),
                options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            c = np.clip(res.x.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
            s = _score_from_centres(c)
            if s > best_score:
                best_score = s
                best_centres = c
        except Exception:
            pass

    # Strategy 2: SLSQP refinement from best
    if time.time() - t0 < DEADLINE - 10:
        try:
            res = minimize(
                _negative_score, best_centres.ravel(),
                method='SLSQP',
                bounds=[(-HALF + 1e-9, HALF - 1e-9)] * (2 * N),
                options={'maxiter': 1500, 'ftol': 1e-16}
            )
            c = np.clip(res.x.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
            s = _score_from_centres(c)
            if s > best_score:
                best_score = s
                best_centres = c
        except Exception:
            pass

    # Strategy 3: Final L-BFGS-B polish
    if time.time() - t0 < DEADLINE - 3:
        try:
            res = minimize(
                _negative_score, best_centres.ravel(),
                method='L-BFGS-B',
                bounds=[(-HALF + 1e-9, HALF - 1e-9)] * (2 * N),
                options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            c = np.clip(res.x.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
            s = _score_from_centres(c)
            if s > best_score:
                best_score = s
                best_centres = c
        except Exception:
            pass

    # Ensure valid output
    best_centres = np.clip(best_centres, -HALF + 1e-9, HALF - 1e-9)
    return best_centres
# EVOLVE-BLOCK-END
