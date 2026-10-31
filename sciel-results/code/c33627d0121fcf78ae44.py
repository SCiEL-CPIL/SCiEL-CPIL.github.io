"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _check_valid(cx, cy, r, W, H, tol=1e-8):
    """Check validity: circles inside [0,W]x[0,H], disjoint interiors."""
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    n = len(cx)
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _relax_positions(cx, cy, r, W, H, max_iter=800, dt=0.002):
    """Force-directed relaxation to fix overlaps and wall violations."""
    n = len(cx)
    cx = cx.astype(float).copy()
    cy = cy.astype(float).copy()
    for it in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)
        # Wall repulsion (strong)
        for i in range(n):
            if cx[i] - r[i] < 0:
                fx[i] += (cx[i] - r[i]) * 50
            if cy[i] - r[i] < 0:
                fy[i] += (cy[i] - r[i]) * 50
            if cx[i] + r[i] > W:
                fx[i] -= (cx[i] + r[i] - W) * 50
            if cy[i] + r[i] > H:
                fy[i] -= (cy[i] + r[i] - H) * 50
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d2 = dx * dx + dy * dy
                min_d = r[i] + r[j]
                if d2 < min_d * min_d:
                    d = np.sqrt(d2) if d2 > 1e-14 else 1e-7
                    overlap = min_d - d
                    if d > 1e-12:
                        fx[i] += (dx / d) * overlap * 10
                        fy[i] += (dy / d) * overlap * 10
                        fx[j] -= (dx / d) * overlap * 10
                        fy[j] -= (dy / d) * overlap * 10
        cx += dt * fx
        cy += dt * fy
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)
    return cx, cy


def _hex_init(W, H, n):
    """Hexagonal packing initialization."""
    # Try to fit hexagonal grid
    # For hex packing, rows are offset
    # Start with a uniform hex pattern
    cols = max(1, int(np.sqrt(n * W / H * 1.155)))
    rows = int(np.ceil(n / cols))
    # Hexagonal spacing
    spacing_x = W / (cols + 0.5)
    spacing_y = H / (rows + 0.5)
    r_base = min(spacing_x * 0.866, spacing_y * 0.95)
    
    cx = []
    cy = []
    for row in range(rows):
        for col in range(cols):
            if len(cx) >= n:
                break
            x = (col + 0.5 + (0.5 if row % 2 == 1 else 0)) * spacing_x
            y = (row + 0.5) * spacing_y
            cx.append(x)
            cy.append(y)
    # Pad if needed
    while len(cx) < n:
        cx.append(W / 2)
        cy.append(H / 2)
    cx = np.array(cx[:n])
    cy = np.array(cy[:n])
    r = np.full(n, r_base * 0.9)
    return cx, cy, r


def _random_init(W, H, n, seed=0):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    r_base = min(W, H) / (2 * np.sqrt(n) * 1.2)
    cx = rng.uniform(r_base, W - r_base, n)
    cy = rng.uniform(r_base, H - r_base, n)
    r = np.full(n, r_base * 0.85)
    return cx, cy, r


def _optimize_aspect(W, H, deadline, seed_offset=0):
    """Optimize for fixed aspect ratio using multiple strategies."""
    best_val = -float('inf')
    best_result = None
    t0 = time.time()
    
    # Strategy 1: Hex initialization + relaxation + radius growth
    for attempt in range(3):
        if time.time() > deadline:
            break
        cx, cy, r = _hex_init(W, H, N)
        # Add small perturbation
        rng = np.random.RandomState(seed_offset + attempt * 100)
        cx += rng.normal(0, 0.01 * W, N)
        cy += rng.normal(0, 0.01 * H, N)
        
        cx, cy = _relax_positions(cx, cy, r, W, H, max_iter=500)
        
        # Now try to grow radii via annealing
        for grow_iter in range(200):
            if time.time() > deadline:
                break
            # Try increasing all radii slightly
            scale = 1.0 + 0.005 * (1.0 - grow_iter / 200.0)
            r_new = r * scale
            cx2, cy2 = _relax_positions(cx, cy, r_new, W, H, max_iter=300)
            if _check_valid(cx2, cy2, r_new, W, H):
                val = np.sum(r_new)
                if val > best_val:
                    best_val = val
                    best_result = (cx2.copy(), cy2.copy(), r_new.copy(), W, H)
                cx, cy, r = cx2, cy2, r_new
            else:
                # Try smaller growth
                scale = 1.0 + 0.002
                r_new = r * scale
                cx2, cy2 = _relax_positions(cx, cy, r_new, W, H, max_iter=200)
                if _check_valid(cx2, cy2, r_new, W, H):
                    val = np.sum(r_new)
                    if val > best_val:
                        best_val = val
                        best_result = (cx2.copy(), cy2.copy(), r_new.copy(), W, H)
                    cx, cy, r = cx2, cy2, r_new
                else:
                    # Try decreasing
                    scale = 0.99
                    r_new = r * scale
                    cx2, cy2 = _relax_positions(cx, cy, r_new, W, H, max_iter=100)
                    if _check_valid(cx2, cy2, r_new, W, H):
                        cx, cy, r = cx2, cy2, r_new
                    break
        
        # Final relaxation
        if best_result is not None:
            cx_f, cy_f, r_f, W_f, H_f = best_result
            cx_f, cy_f = _relax_positions(cx_f, cy_f, r_f, W_f, H_f, max_iter=300)
            if _check_valid(cx_f, cy_f, r_f, W_f, H_f):
                val = np.sum(r_f)
                if val > best_val:
                    best_val = val
                    best_result = (cx_f.copy(), cy_f.copy(), r_f.copy(), W_f, H_f)
    
    # Strategy 2: Random initialization
    for attempt in range(2):
        if time.time() > deadline:
            break
        cx, cy, r = _random_init(W, H, N, seed=seed_offset + 1000 + attempt)
        cx, cy = _relax_positions(cx, cy, r, W, H, max_iter=500)
        
        for grow_iter in range(150):
            if time.time() > deadline:
                break
            scale = 1.0 + 0.004 * (1.0 - grow_iter / 150.0)
            r_new = r * scale
            cx2, cy2 = _relax_positions(cx, cy, r_new, W, H, max_iter=250)
            if _check_valid(cx2, cy2, r_new, W, H):
                val = np.sum(r_new)
                if val > best_val:
                    best_val = val
                    best_result = (cx2.copy(), cy2.copy(), r_new.copy(), W, H)
                cx, cy, r = cx2, cy2, r_new
            else:
                scale = 0.99
                r_new = r * scale
                cx2, cy2 = _relax_positions(cx, cy, r_new, W, H, max_iter=100)
                if _check_valid(cx2, cy2, r_new, W, H):
                    cx, cy, r = cx2, cy2, r_new
                break
    
    return best_result, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    best_val = -float('inf')
    best_result = None
    
    # Try various aspect ratios W (H = 2 - W)
    # Square is W=1, H=1. Try a range
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.7, 1.4, 0.65, 1.5, 0.6, 1.6, 0.55]
    
    for idx, W in enumerate(aspect_ratios):
        if time.time() > deadline:
            break
        H = 2.0 - W
        if H <= 0.1:
            continue
        result, val = _optimize_aspect(W, H, deadline, seed_offset=idx * 50)
        if val > best_val:
            best_val = val
            best_result = result
    
    # If no valid result, fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
        return np.array(cells[:N])
    
    cx, cy, r, W, H = best_result
    return np.column_stack([cx, cy, r])

# EVOLVE-BLOCK-END
