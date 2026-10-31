"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import differential_evolution, minimize

N = 21
PERIMETER = 4.0
START = time.time()
DEADLINE = START + 138.0  # safety margin before 150s kill


def _validity(circles, W, H):
    """Check disjoint interiors and bounds. Returns (ok, perimeter, sum_radii)."""
    n = len(circles)
    if n != N:
        return False, 0.0, 0.0
    minx = min(c[0] - c[2] for c in circles)
    miny = min(c[1] - c[2] for c in circles)
    maxx = max(c[0] + c[2] for c in circles)
    maxy = max(c[1] + c[2] for c in circles)
    bw = maxx - minx
    bh = maxy - miny
    per = 2.0 * (bw + bh)
    if per > PERIMETER + 1e-6:
        return False, per, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(circles[i][0] - circles[j][0], circles[i][1] - circles[j][1])
            if d < circles[i][2] + circles[j][2] - 1e-7:
                return False, per, 0.0
    s = sum(c[2] for c in circles)
    return True, per, s


def _scale_to_perimeter(circles, target_per=PERIMETER):
    """Scale the whole packing so bounding-box perimeter equals target_per."""
    minx = min(c[0] - c[2] for c in circles)
    miny = min(c[1] - c[2] for c in circles)
    maxx = max(c[0] + c[2] for c in circles)
    maxy = max(c[1] + c[2] for c in circles)
    bw = maxx - minx
    bh = maxy - miny
    per = 2.0 * (bw + bh)
    if per <= 0:
        return circles
    s = target_per / per
    out = []
    for (x, y, r) in circles:
        nx = (x - minx) * s
        ny = (y - miny) * s
        nr = r * s
        out.append((nx, ny, nr))
    return out


def _force_directed_relax(cx, cy, r, W, H, max_iter=500, dt=0.003):
    """Force-directed relaxation to fix overlaps and wall violations."""
    n = len(cx)
    cx = cx.copy().astype(float)
    cy = cy.copy().astype(float)
    for it in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)
        # Wall forces
        for i in range(n):
            if cx[i] - r[i] < 0:
                fx[i] += (cx[i] - r[i]) * 20
            if cy[i] - r[i] < 0:
                fy[i] += (cy[i] - r[i]) * 20
            if cx[i] + r[i] > W:
                fx[i] -= (cx[i] + r[i] - W) * 20
            if cy[i] + r[i] > H:
                fy[i] -= (cy[i] + r[i] - H) * 20
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d:
                    if d < 1e-12:
                        dx, dy = 1.0, 0.0
                        d = 1.0
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 8
                    fy[i] += (dy / d) * overlap * 8
                    fx[j] -= (dx / d) * overlap * 8
                    fy[j] -= (dy / d) * overlap * 8
        cx += dt * fx
        cy += dt * fy
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)
    return cx, cy


def _objective(x, W, H, mu=500.0):
    """Penalty objective: minimize -sum(r) + penalties for violations.
    x = [cx(21), cy(21), r(21)]"""
    cx = x[:N]
    cy = x[N:2*N]
    r = x[2*N:3*N]
    
    val = -np.sum(r)
    
    # Wall violations
    wall_pen = 0.0
    for i in range(N):
        w1 = max(0, r[i] - cx[i])
        w2 = max(0, r[i] - (W - cx[i]))
        w3 = max(0, r[i] - cy[i])
        w4 = max(0, r[i] - (H - cy[i]))
        wall_pen += w1**2 + w2**2 + w3**2 + w4**2
    val += mu * wall_pen
    
    # Overlap violations
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.sqrt(dx * dx + dy * dy)
            min_d = r[i] + r[j]
            if d < min_d:
                overlap_pen += (min_d - d) ** 2
    val += mu * overlap_pen
    
    return val


def _hex_init(W, H, n_cols):
    """Generate hexagonal lattice positions for N circles in [0,W]x[0,H]."""
    positions = []
    n_rows = int(np.ceil(N / n_cols))
    dx = W / n_cols
    dy = H / n_rows
    r_est = min(dx, dy) * 0.48
    
    for row in range(n_rows):
        for col in range(n_cols):
            if len(positions) >= N:
                break
            x = (col + 0.5) * dx
            y = (row + 0.5) * dy
            if row % 2 == 1:
                x += dx * 0.5
            # Clamp to bounds
            x = min(max(x, r_est), W - r_est)
            y = min(max(y, r_est), H - r_est)
            positions.append((x, y, r_est))
    return positions[:N]


def _optimize_for_aspect(W, H, n_cols, time_budget=10.0):
    """Optimize using differential evolution from hexagonal init, then local refine."""
    t_start = time.time()
    
    # Hexagonal initialization
    circles_init = _hex_init(W, H, n_cols)
    cx0 = np.array([c[0] for c in circles_init])
    cy0 = np.array([c[1] for c in circles_init])
    r0 = np.array([c[2] for c in circles_init])
    
    # Force-directed to get a feasible starting point
    cx0, cy0 = _force_directed_relax(cx0, cy0, r0, W, H, max_iter=200, dt=0.003)
    
    x0 = np.concatenate([cx0, cy0, r0])
    
    # Bounds for DE
    r_max = min(W, H) / 2 * 0.95
    bounds = []
    for i in range(N):
        bounds.append((r0[i] * 0.1, W - r0[i] * 0.1))  # cx: keep some margin
    for i in range(N):
        bounds.append((r0[i] * 0.1, H - r0[i] * 0.1))  # cy
    for i in range(N):
        bounds.append((1e-4, r_max))  # r
    
    # Differential evolution
    try:
        de_result = differential_evolution(
            _objective, bounds, args=(W, H, 500.0),
            seed=42, maxiter=300, popsize=15, tol=1e-6,
            mutation=(0.3, 1.0), recombination=0.8,
            polish=True,
            init='latinhypercube',
            constraints=[],
            callback=None,
            update_interval=1,
            disp=False,
            time_budget=time_budget
        )
        
        if de_result.success or de_result.fun < _objective(x0, W, H, 500.0):
            cx_opt = de_result.x[:N]
            cy_opt = de_result.x[N:2*N]
            r_opt = de_result.x[2*N:3*N]
            
            # Clip to valid ranges
            cx_opt = np.clip(cx_opt, 0, W)
            cy_opt = np.clip(cy_opt, 0, H)
            r_opt = np.clip(r_opt, 1e-4, r_max)
            
            circles = list(zip(cx_opt, cy_opt, r_opt))
            ok, per, s = _validity(circles, W, H)
            if ok:
                return circles, s
    except Exception:
        pass
    
    # Fallback: try SLSQP from the hex init
    if time.time() - t_start < time_budget:
        try:
            res = minimize(_objective, x0, args=(W, H, 500.0), method='SLSQP',
                           bounds=bounds, options={'maxiter': 300, 'ftol': 1e-8})
            if res.fun < _objective(x0, W, H, 500.0):
                cx_opt = np.clip(res.x[:N], 0, W)
                cy_opt = np.clip(res.x[N:2*N], 0, H)
                r_opt = np.clip(res.x[2*N:3*N], 1e-4, r_max)
                circles = list(zip(cx_opt, cy_opt, r_opt))
                ok, per, s = _validity(circles, W, H)
                if ok:
                    return circles, s
        except:
            pass
    
    return None, -float('inf')


def _refine_positions(circles, W, H, steps=2000, lr0=0.03):
    """Local search: nudge each circle to increase its radius while keeping constraints."""
    circles = [list(c) for c in circles]
    n = len(circles)
    rng = random.Random(12345)
    
    for step in range(steps):
        lr = lr0 * (1.0 - step / steps)
        i = rng.randrange(n)
        x, y, r = circles[i]
        
        # Try multiple random directions
        best_move = None
        best_nr = r
        
        for _ in range(8):
            dx = rng.uniform(-1, 1)
            dy = rng.uniform(-1, 1)
            norm = np.hypot(dx, dy)
            if norm < 1e-9:
                continue
            dx, dy = dx / norm, dy / norm
            nx = x + dx * lr
            ny = y + dy * lr
            nx = min(max(nx, 0.0), W)
            ny = min(max(ny, 0.0), H)
            
            # Max radius at this position
            nr = min(nx, W - nx, ny, H - ny)
            for j in range(n):
                if j == i:
                    continue
                px, py, pr = circles[j]
                d = np.hypot(nx - px, ny - py)
                cand = d - pr
                if cand < nr:
                    nr = cand
            nr = max(nr, 0.0)
            
            if nr > best_nr:
                best_nr = nr
                best_move = (nx, ny, nr)
        
        if best_move is not None and best_nr > r + 1e-12:
            circles[i] = list(best_move)
    
    return circles


def construct():
    global DEADLINE
    t0 = time.time()
    DEADLINE = t0 + 135.0  # extra safety margin
    
    best = None
    best_val = -1.0
    
    # Aspect ratio candidates: W + H = 2
    # Focus on more square-like and moderately elongated ratios
    aspect_choices = [1.0, 0.85, 1.18, 0.75, 1.3, 0.65, 1.5, 0.55, 1.7, 0.5, 0.9, 1.1]
    rng = random.Random(42)
    rng.shuffle(aspect_choices)
    
    for W in aspect_choices:
        if time.time() > DEADLINE:
            break
        H = 2.0 - W
        if H <= 0.15:
            continue
        
        # Time remaining for this aspect ratio
        time_left = DEADLINE - time.time()
        if time_left < 3.0:
            break
        
        # Try multiple column configurations
        for n_cols in [4, 5, 6, 3]:
            if time.time() > DEADLINE:
                break
            
            # Allocate time budget based on remaining time
            tb = min(8.0, max(2.0, (DEADLINE - time.time()) * 0.3))
            
            circles, val = _optimize_for_aspect(W, H, n_cols, time_budget=tb)
            if circles is not None and val > best_val:
                # Refine with local search
                if time.time() < DEADLINE - 2.0:
                    circles = _refine_positions(circles, W, H, steps=1500, lr0=0.02)
                
                # Scale to exact perimeter 4
                circles = _scale_to_perimeter(circles, PERIMETER)
                ok, per, s = _validity(circles, 2.0, 2.0)
                if ok and s > best_val:
                    best_val = s
                    best = circles
    
    if best is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best = cells[:N]
        best = _scale_to_perimeter(best, PERIMETER)
    
    return np.array(best)


if __name__ == "__main__":
    res = construct()
    print(res.shape)
    print(res[:3])
