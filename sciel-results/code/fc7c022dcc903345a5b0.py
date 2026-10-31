"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize, differential_evolution

N = 21
PERIMETER = 4.0
START = time.time()
DEADLINE = START + 138.0


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


def _force_directed_relax(cx, cy, r, W, H, max_iter=500, dt=0.005):
    """Force-directed relaxation to fix overlaps and wall violations."""
    n = len(cx)
    cx = cx.copy()
    cy = cy.copy()
    for _ in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)
        for i in range(n):
            if cx[i] - r[i] < 0:
                fx[i] += (cx[i] - r[i]) * 10
            if cy[i] - r[i] < 0:
                fy[i] += (cy[i] - r[i]) * 10
            if cx[i] + r[i] > W:
                fx[i] -= (cx[i] + r[i] - W) * 10
            if cy[i] + r[i] > H:
                fy[i] -= (cy[i] + r[i] - H) * 10
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 5
                    fy[i] += (dy / d) * overlap * 5
                    fx[j] -= (dx / d) * overlap * 5
                    fy[j] -= (dy / d) * overlap * 5
        cx += dt * fx
        cy += dt * fy
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)
    return cx, cy


def _penalty_objective(x, W, H, mu=50.0):
    """Penalty objective for optimization. x = [cx, cy, r] flattened."""
    cx = x[:N]
    cy = x[N:2*N]
    r = x[2*N:3*N]
    
    val = -np.sum(r)
    
    # Wall violations
    for i in range(N):
        if cx[i] - r[i] < 0:
            val += mu * (cx[i] - r[i]) ** 2
        if cy[i] - r[i] < 0:
            val += mu * (cy[i] - r[i]) ** 2
        if cx[i] + r[i] > W:
            val += mu * (cx[i] + r[i] - W) ** 2
        if cy[i] + r[i] > H:
            val += mu * (cy[i] + r[i] - H) ** 2
    
    # Overlap violations
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.sqrt(dx * dx + dy * dy)
            min_d = r[i] + r[j]
            if d < min_d:
                val += mu * (min_d - d) ** 2
    
    return val


def _refine_positions(circles, W, H, steps=2000, lr0=0.02):
    """Local search: nudge each circle to increase its radius while keeping constraints."""
    circles = [list(c) for c in circles]
    n = len(circles)
    rng = random.Random(42)
    
    for step in range(steps):
        if time.time() > DEADLINE:
            break
        lr = lr0 * max(0.1, 1.0 - step / steps)
        i = rng.randrange(n)
        x, y, r = circles[i]
        
        best_nr = r
        best_nx, best_ny = x, y
        
        for _ in range(5):
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
            
            # Check if new position allows larger radius
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
                best_nx, best_ny = nx, ny
        
        if best_nr > r + 1e-12:
            circles[i] = [best_nx, best_ny, best_nr]
    
    return circles


def _optimize_aspect(W, H, seed=0):
    """Optimize circle positions and radii for a fixed aspect ratio W:H."""
    rng = random.Random(seed)
    best_circles = None
    best_val = -float('inf')
    
    # Multiple initial configurations
    for attempt in range(3):
        if time.time() > DEADLINE:
            break
        
        if attempt == 0:
            # Grid layout
            cols = 5
            rows = int(np.ceil(N / cols))
            W_cell = W / cols
            H_cell = H / rows
            cx = np.array([(i + 0.5) * W_cell for i in range(cols) for _ in range(rows)][:N])
            cy = np.array([(j + 0.5) * H_cell for _ in range(cols) for j in range(rows)][:N])
            r = np.full(N, min(W_cell, H_cell) / 2 * 0.85)
        elif attempt == 1:
            # Random layout
            cx = np.array([rng.uniform(r, W - r) for r in np.full(N, 0.1)])
            cy = np.array([rng.uniform(r, H - r) for r in np.full(N, 0.1)])
            r = np.full(N, 0.1)
        else:
            # Clustered layout
            cx = np.array([rng.uniform(0, W) for _ in range(N)])
            cy = np.array([rng.uniform(0, H) for _ in range(N)])
            r = np.full(N, 0.15)
        
        # Force-directed to fix initial overlaps
        cx, cy = _force_directed_relax(cx, cy, r, W, H, max_iter=400)
        
        x0 = np.concatenate([cx, cy, r])
        bounds = []
        for i in range(N):
            bounds.append((0, W))
        for i in range(N):
            bounds.append((0, H))
        for i in range(N):
            bounds.append((1e-4, min(W, H) / 2))
        
        try:
            res = minimize(_penalty_objective, x0, args=(W, H, 50.0), method='SLSQP',
                           bounds=bounds, options={'maxiter': 500, 'ftol': 1e-9})
            if res.success or res.fun < _penalty_objective(x0, W, H, 50.0):
                cx_opt = res.x[:N]
                cy_opt = res.x[N:2*N]
                r_opt = res.x[2*N:3*N]
                circles = list(zip(cx_opt, cy_opt, r_opt))
                ok, per, s = _validity(circles, W, H)
                if ok and s > best_val:
                    best_val = s
                    best_circles = circles
        except:
            pass
    
    return best_circles, best_val


def construct():
    global DEADLINE
    t0 = time.time()
    DEADLINE = t0 + 138.0
    
    best = None
    best_val = -1.0
    
    # Aspect ratio candidates: W + H = 2
    aspect_choices = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 1.1, 0.9]
    random.shuffle(aspect_choices)
    
    for W in aspect_choices:
        if time.time() > DEADLINE:
            break
        H = 2.0 - W
        if H <= 0.1:
            continue
        
        circles, val = _optimize_aspect(W, H, seed=int(W * 100))
        if circles is not None and val > best_val:
            # Refine
            circles = _refine_positions(circles, W, H, steps=1500, lr0=0.015)
            # Scale to exact perimeter 4
            circles = _scale_to_perimeter(circles, PERIMETER)
            ok, per, s = _validity(circles, 2.0, 2.0)
            if ok and s > best_val:
                best_val = s
                best = circles
    
    if best is None:
        # Fallback to grid
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
