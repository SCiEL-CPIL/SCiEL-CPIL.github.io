import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
PERIMETER = 4.0
START = time.time()
DEADLINE_FRACTION = 0.85  # Safety margin

def construct():
    t0 = time.time()
    deadline = t0 + 120.0 * DEADLINE_FRACTION
    
    best_val = -1.0
    best_circles = None
    
    # Aspect ratios to try: W + H = 2. W in [0.5, 1.5]
    # The optimal is likely near 1.0 (square) but we will scan.
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45]
    random.shuffle(aspect_ratios)
    
    for W in aspect_ratios:
        if time.time() - t0 > deadline:
            break
        H = 2.0 - W
        
        # Try multiple grid configurations
        for grid_n in [4, 5, 6]:
            if time.time() - t0 > deadline:
                break
            
            circles, val = _optimize_for_aspect(W, H, grid_n, t0, deadline)
            if circles is not None and val > best_val:
                # Refine
                circles = _refine_positions(circles, W, H, steps=500, lr0=0.01)
                # Scale to exact perimeter 4
                circles = _scale_to_perimeter(circles, PERIMETER)
                ok, per, s = _is_valid(circles, 2.0, 2.0)
                if ok and s > best_val:
                    best_val = s
                    best_circles = circles
                    
    if best_circles is None:
        # Fallback: simple grid
        side = 5
        pts = []
        for i in range(side):
            for j in range(side):
                if len(pts) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 0.01
                    pts.append((x, y, r))
        best_circles = pts
        # Scale
        minx = min(c[0] - c[2] for c in best_circles)
        miny = min(c[1] - c[2] for c in best_circles)
        maxx = max(c[0] + c[2] for c in best_circles)
        maxy = max(c[1] + c[2] for c in best_circles)
        bw = maxx - minx
        bh = maxy - miny
        per = 2.0 * (bw + bh)
        s = 4.0 / per
        best_circles = [((x-minx)*s, (y-miny)*s, r*s) for (x,y,r) in best_circles]
        
    return np.array(best_circles)

def _is_valid(circles, W, H, tol=1e-6):
    n = len(circles)
    if n != N:
        return False, 0.0, 0.0
    # Check bounds
    for x, y, r in circles:
        if x - r < -tol or x + r > W + tol or y - r < -tol or y + r > H + tol:
            return False, 0.0, 0.0
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            d = np.hypot(x1-x2, y1-y2)
            if d < r1 + r2 - tol:
                return False, 0.0, 0.0
    # Calculate perimeter and sum
    minx = min(c[0] - c[2] for c in circles)
    miny = min(c[1] - c[2] for c in circles)
    maxx = max(c[0] + c[2] for c in circles)
    maxy = max(c[1] + c[2] for c in circles)
    bw = maxx - minx
    bh = maxy - miny
    per = 2.0 * (bw + bh)
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

def _force_directed_relax(cx, cy, r, W, H, max_iter=300, dt=0.002):
    """Simple force-directed relaxation to fix overlaps and wall violations."""
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

def _objective(x, W, H, mu=100.0):
    """Penalty objective for SLSQP. x = [cx, cy, r] flattened."""
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

def _optimize_for_aspect(W, H, grid_n, t0, deadline, max_iter=400):
    """Optimize circle positions and radii for a fixed aspect ratio W:H."""
    cols = grid_n
    rows = int(np.ceil(N / cols))
    W_cell = W / cols
    H_cell = H / rows
    
    cx = np.array([(i + 0.5) * W_cell for i in range(cols) for _ in range(rows)][:N])
    cy = np.array([(j + 0.5) * H_cell for _ in range(cols) for j in range(rows)][:N])
    r = np.full(N, min(W_cell, H_cell) / 2 * 0.9)
    
    # Force-directed to fix initial overlaps
    cx, cy = _force_directed_relax(cx, cy, r, W, H, max_iter=300)
    
    x0 = np.concatenate([cx, cy, r])
    bounds = []
    for i in range(N):
        bounds.append((0, W))
    for i in range(N):
        bounds.append((0, H))
    for i in range(N):
        bounds.append((1e-4, min(W, H) / 2))
    
    try:
        res = minimize(_objective, x0, args=(W, H, 100.0), method='SLSQP',
                       bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-8})
        if res.success or res.fun < _objective(x0, W, H, 100.0):
            cx_opt = res.x[:N]
            cy_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            # Validate
            circles = list(zip(cx_opt, cy_opt, r_opt))
            ok, per, s = _is_valid(circles, W, H)
            if ok:
                return circles, s
    except:
        pass
    return None, -float('inf')

def _refine_positions(circles, W, H, steps=1000, lr0=0.02):
    """Local search: nudge each circle to increase its radius while keeping constraints."""
    circles = [list(c) for c in circles]
    n = len(circles)
    rng = random.Random(12345)
    
    for step in range(steps):
        lr = lr0 * (1.0 - step / steps)
        i = rng.randrange(n)
        x, y, r = circles[i]
        
        for _ in range(3):
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
            
            if nr > r + 1e-12:
                circles[i] = [nx, ny, nr]
                break
    
    return circles

if __name__ == "__main__":
    res = construct()
    print(res.shape)
    print(res[:3])
