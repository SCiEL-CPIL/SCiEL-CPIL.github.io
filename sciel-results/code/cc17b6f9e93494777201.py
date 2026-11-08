"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def _optimize_radii(circles, w, h, max_iter=200, tol=1e-10):
    """Iteratively maximize radii given fixed positions."""
    circles = circles.copy()
    n = len(circles)
    cx = circles[:, 0]
    cy = circles[:, 1]
    
    for it in range(max_iter):
        improved = False
        for i in range(n):
            r = circles[i, 2]
            # Wall constraint
            r_wall = min(cx[i], w - cx[i], cy[i], h - cy[i])
            r_max = r_wall
            # Circle constraints
            for j in range(n):
                if i == j:
                    continue
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = np.sqrt(dx*dx + dy*dy)
                r_cand = dist - circles[j, 2]
                if r_cand < r_max:
                    r_max = r_cand
            if r_max > r + tol:
                circles[i, 2] = r_max
                improved = True
        if not improved:
            break
    return circles

def _grid_positions(ncols, nrows, w, h, n_circles):
    """Return positions for n_circles in a ncols x nrows grid."""
    positions = []
    for i in range(ncols):
        for j in range(nrows):
            if len(positions) >= n_circles:
                break
            cx = (i + 0.5) * w / ncols
            cy = (j + 0.5) * h / nrows
            positions.append((cx, cy))
    return positions

def _evaluate(circles, w, h):
    """Validate and return sum of radii if valid, else 0."""
    n = len(circles)
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    # Check inside
    if np.any(cx - r < -1e-7) or np.any(cx + r > w + 1e-7):
        return 0.0
    if np.any(cy - r < -1e-7) or np.any(cy + r > h + 1e-7):
        return 0.0
    # Check overlap
    for i in range(n):
        for j in range(i+1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if np.sqrt(dx*dx + dy*dy) < r[i] + r[j] - 1e-7:
                return 0.0
    return float(np.sum(r))

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    best_circles = None
    best_sum = 0.0
    
    # Try multiple aspect ratios and grid configs
    aspects = [1.0, 2.0, 0.5, 1.5, 0.667, 1.333, 0.75, 1.25, 0.8, 1.2]
    
    for aspect in aspects:
        if time.time() > deadline:
            break
        h = 2.0 / (1.0 + aspect)
        w = 2.0 * aspect / (1.0 + aspect)
        
        for ncols in range(1, 9):
            for nrows in range(1, 9):
                if ncols * nrows < N:
                    continue
                if ncols * nrows > N + 8:
                    continue
                positions = _grid_positions(ncols, nrows, w, h, N)
                if len(positions) < N:
                    continue
                circles = np.zeros((N, 3))
                for k, (px, py) in enumerate(positions):
                    circles[k, 0] = px
                    circles[k, 1] = py
                    circles[k, 2] = 0.01  # small initial radius
                circles = _optimize_radii(circles, w, h)
                s = _evaluate(circles, w, h)
                if s > best_sum:
                    best_sum = s
                    best_circles = circles.copy()
    
    # Fallback
    if best_circles is None or best_sum <= 0:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles
# EVOLVE-BLOCK-END

