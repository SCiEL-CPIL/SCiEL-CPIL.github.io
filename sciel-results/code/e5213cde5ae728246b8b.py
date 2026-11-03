import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 145.0  # 5s margin before 150s kill

def _is_valid(circles, tol=1e-9):
    """Check validity: disjoint interiors, bounding rectangle perimeter <= 4."""
    if len(circles) != N:
        return False
    if np.any(circles[:, 2] <= 0):
        return False
    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]
    xmin, xmax = np.min(xs - rs), np.max(xs + rs)
    ymin, ymax = np.min(ys - rs), np.max(ys + rs)
    w, h = xmax - xmin, ymax - ymin
    if 2 * (w + h) > 4.0 + 1e-6:
        return False
    # Check pairwise non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    return True

def _sum_radii(circles):
    return float(np.sum(circles[:, 2]))

def _separate_circles(circles, w, h, iters=200):
    """Separate circles that overlap or are out of bounds."""
    circles = circles.copy()
    for _ in range(iters):
        moved = False
        # Boundary
        for i in range(N):
            x, y, r = circles[i]
            if x - r < 0:
                circles[i, 0] = r
                moved = True
            if x + r > w:
                circles[i, 0] = w - r
                moved = True
            if y - r < 0:
                circles[i, 1] = r
                moved = True
            if y + r > h:
                circles[i, 1] = h - r
                moved = True
        # Pairwise
        for i in range(N):
            for j in range(i + 1, N):
                dx = circles[j, 0] - circles[i, 0]
                dy = circles[j, 1] - circles[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_d = circles[i, 2] + circles[j, 2]
                if dist < min_d and dist > 1e-12:
                    overlap = min_d - dist
                    nx, ny = dx / dist, dy / dist
                    circles[i, 0] -= nx * overlap * 0.5
                    circles[i, 1] -= ny * overlap * 0.5
                    circles[j, 0] += nx * overlap * 0.5
                    circles[j, 1] += ny * overlap * 0.5
                    moved = True
                elif dist <= 1e-12:
                    circles[i, 0] -= 0.005
                    circles[j, 0] += 0.005
                    moved = True
        if not moved:
            break
    return circles

def _objective_and_grad(params, w, h):
    """
    Objective: minimize negative sum of radii + penalty for overlaps and boundary violations.
    params: flat array of length 3*N: [x0,y0,r0, x1,y1,r1, ...]
    """
    circles = params.reshape(N, 3)
    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]
    
    # Negative sum of radii (we maximize sum)
    obj = -np.sum(rs)
    
    # Penalty for boundary violations
    pen = 0.0
    for i in range(N):
        pen += max(0, -(xs[i] - rs[i])) ** 2  # left
        pen += max(0, (xs[i] + rs[i]) - w) ** 2  # right
        pen += max(0, -(ys[i] - rs[i])) ** 2  # bottom
        pen += max(0, (ys[i] + rs[i]) - h) ** 2  # top
    
    # Penalty for overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_d = rs[i] + rs[j]
            if dist < min_d:
                pen += (min_d - dist) ** 2
    
    obj += 100.0 * pen
    
    # Gradient
    grad = np.zeros(3 * N)
    for i in range(N):
        grad[3*i + 2] = -1.0  # d(-sum r)/dr_i
    
    # Gradient for boundary penalties
    for i in range(N):
        # left
        if xs[i] - rs[i] < 0:
            g = 2 * 100.0 * (-(xs[i] - rs[i]))
            grad[3*i] += g
            grad[3*i + 2] += -g
        # right
        if xs[i] + rs[i] > w:
            g = 2 * 100.0 * ((xs[i] + rs[i]) - w)
            grad[3*i] += g
            grad[3*i + 2] += g
        # bottom
        if ys[i] - rs[i] < 0:
            g = 2 * 100.0 * (-(ys[i] - rs[i]))
            grad[3*i + 1] += g
            grad[3*i + 2] += -g
        # top
        if ys[i] + rs[i] > h:
            g = 2 * 100.0 * ((ys[i] + rs[i]) - h)
            grad[3*i + 1] += g
            grad[3*i + 2] += g
    
    # Gradient for overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_d = rs[i] + rs[j]
            if dist < min_d and dist > 1e-12:
                overlap = min_d - dist
                factor = 2 * 100.0 * overlap / dist
                grad[3*i] += factor * dx
                grad[3*i + 1] += factor * dy
                grad[3*i + 2] += 2 * 100.0 * overlap
                grad[3*j] -= factor * dx
                grad[3*j + 1] -= factor * dy
                grad[3*j + 2] += 2 * 100.0 * overlap
    
    return obj, grad

def _objective(params, w, h):
    return _objective_and_grad(params, w, h)[0]

def _local_optimize(circles, w, h, max_iters=500):
    """
    Local optimization: try to grow radii while maintaining feasibility.
    Uses a penalty-based approach with scipy minimize.
    """
    best = circles.copy()
    best_score = _sum_radii(best)
    
    # Phase 1: Use scipy to optimize positions and radii jointly
    x0 = circles.flatten()
    
    # Bounds: positions in [r, w-r], radii in [r_min, r_max]
    r_min = 0.01
    r_max = min(w, h) / 2.0 * 0.95
    
    bounds = []
    for i in range(N):
        bounds.append((r_min, w - r_min))  # x
        bounds.append((r_min, h - r_min))  # y
        bounds.append((r_min, r_max))      # r
    
    # Try multiple optimization attempts
    for attempt in range(3):
        if time.time() > DEADLINE:
            break
        
        res = minimize(
            _objective,
            x0,
            args=(w, h),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < _objective(x0, w, h):
            new_circles = res.x.reshape(N, 3)
            # Check validity
            if _is_valid(new_circles):
                new_score = _sum_radii(new_circles)
                if new_score > best_score:
                    best = new_circles.copy()
                    best_score = new_score
                    x0 = res.x
    
    return best, best_score

def _grid_initialization(w, h, m, n):
    """Create a grid of circles in a w x h rectangle with m columns and n rows."""
    if m * n < N:
        return None
    
    cells = []
    count = 0
    for i in range(m):
        for j in range(n):
            if count >= N:
                break
            x = (i + 0.5) * w / m
            y = (j + 0.5) * h / n
            r = min(w / m, h / n) / 2.0 * 0.98
            if r > 0.01:
                cells.append([x, y, r])
                count += 1
        if count >= N:
            break
    
    if len(cells) == N:
        return np.array(cells)
    return None

def _hex_lattice_positions(w, h):
    """
    Generate hexagonal lattice positions that fit in w x h rectangle.
    Returns list of (x, y) positions.
    """
    positions = []
    # Try various spacings
    for spacing in np.linspace(0.1, 1.0, 50):
        dx = spacing
        dy = spacing * np.sqrt(3) / 2.0
        
        col = 0
        x = 0.0
        while x < w:
            row = 0
            y_offset = 0.0 if col % 2 == 0 else dy / 2.0
            y = y_offset
            while y < h:
                if x >= 0 and x <= w and y >= 0 and y <= h:
                    positions.append((x, y))
                y += dy
                row += 1
            x += dx * 0.75
            col += 1
        
        if len(positions) >= N:
            break
    
    return positions[:N] if len(positions) >= N else None

def _random_initialization(w, h, seed):
    """Random initialization of N circles in w x h rectangle."""
    rng = np.random.RandomState(seed)
    r_max = min(w, h) / 2.0 * 0.9
    r_min = 0.05
    circles = np.zeros((N, 3))
    for i in range(N):
        r = rng.uniform(r_min, r_max * 0.5)
        x = rng.uniform(r, w - r)
        y = rng.uniform(r, h - r)
        circles[i] = [x, y, r]
    return circles

def _polish(circles, w, h, iters=50):
    """
    Polish: iteratively try to increase each circle's radius slightly,
    then separate, and keep if valid.
    """
    circles = circles.copy()
    for _ in range(iters):
        if time.time() > DEADLINE:
            break
        improved = False
        for i in range(N):
            old_r = circles[i, 2]
            # Try increasing radius
            for frac in [1.02, 1.01, 1.005]:
                new_r = old_r * frac
                # Check if it fits in bounds
                if new_r > w - circles[i, 0] or new_r > circles[i, 0]:
                    continue
                if new_r > h - circles[i, 1] or new_r > circles[i, 1]:
                    continue
                
                circles[i, 2] = new_r
                sep = _separate_circles(circles, w, h, iters=50)
                if _is_valid(sep):
                    circles = sep
                    improved = True
                    break
                else:
                    circles[i, 2] = old_r
            if not improved:
                # Try all circles at once
                break
        if not improved:
            break
    
    return circles

def _successive_convex_relaxation(circles, w, h, max_rounds=30):
    """
    SCR: fix positions, solve for optimal radii (linear program),
    then fix radii, optimize positions.
    """
    circles = circles.copy()
    
    for round_num in range(max_rounds):
        if time.time() > DEADLINE:
            break
        
        # Step 1: Fix positions, maximize sum of radii subject to constraints
        # For each circle i: r_i <= min(x_i, w-x_i, y_i, h-y_i, min_j (dist(i,j) - r_j))
        # This is a system of linear inequalities: r_i + r_j <= dist(i,j)
        # and r_i <= x_i, r_i <= w-x_i, r_i <= y_i, r_i <= h-y_i
        
        # Use scipy linprog
        from scipy.optimize import linprog
        
        n = N
        # Variables: r_0, ..., r_{n-1}
        # Maximize sum(r) => minimize -sum(r)
        c = -np.ones(n)
        
        # Constraints:
        # r_i + r_j <= dist(i,j) for all i < j
        # r_i <= x_i, r_i <= w - x_i, r_i <= y_i, r_i <= h - y_i
        A_ub = []
        b_ub = []
        
        xs, ys = circles[:, 0], circles[:, 1]
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                row = np.zeros(n)
                row[i] = 1
                row[j] = 1
                A_ub.append(row)
                b_ub.append(dist)
        
        for i in range(n):
            for bound_val in [xs[i], w - xs[i], ys[i], h - ys[i]]:
                row = np.zeros(n)
                row[i] = 1
                A_ub.append(row)
                b_ub.append(bound_val)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        bounds = [(0.001, None)] * n
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            new_rs = res.x
            # Check if this improves
            if np.sum(new_rs) > np.sum(circles[:, 2]) - 1e-9:
                circles[:, 2] = new_rs
        
        # Step 2: Fix radii, optimize positions using scipy
        r_fixed = circles[:, 2].copy()
        x0 = circles[:, :2].flatten()
        
        def pos_obj(params):
            pos = params.reshape(N, 2)
            pen = 0.0
            # Boundary
            for i in range(N):
                r = r_fixed[i]
                pen += max(0, -(pos[i, 0] - r)) ** 2
                pen += max(0, (pos[i, 0] + r) - w) ** 2
                pen += max(0, -(pos[i, 1] - r)) ** 2
                pen += max(0, (pos[i, 1] + r) - h) ** 2
            # Overlap
            for i in range(N):
                for j in range(i + 1, N):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_d = r_fixed[i] + r_fixed[j]
                    if dist < min_d:
                        pen += (min_d - dist) ** 2
            return 100.0 * pen
        
        res2 = minimize(pos_obj, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10})
        
        if res2.success:
            circles[:, :2] = res2.x.reshape(N, 2)
    
    return circles

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    best_result = None
    best_score = 0.0
    
    # Aspect ratios to try: w/h ratio, with 2(w+h)=4 => w+h=2
    # ar = w/h, so w = 2*ar/(1+ar), h = 2/(1+ar)
    aspect_ratios = [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.2]
    
    # Sort by expected quality: near-square and slightly elongated first
    aspect_ratios_sorted = sorted(aspect_ratios, key=lambda ar: abs(ar - 1.0))
    
    for ar in aspect_ratios_sorted:
        if time_left() < 10.0:
            break
        
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        # Multiple initializations
        inits = []
        
        # Grid initializations
        for m in range(2, 9):
            for n in range(2, 9):
                if m * n < N:
                    continue
                grid = _grid_initialization(w, h, m, n)
                if grid is not None and _is_valid(grid):
                    inits.append(grid)
        
        # Hex lattice
        hex_pos = _hex_lattice_positions(w, h)
        if hex_pos is not None:
            # Assign radii based on local spacing
            r_est = min(w / 5.0, h / 4.0) * 0.45
            hex_circles = np.array([[x, y, r_est] for x, y in hex_pos])
            hex_circles = _separate_circles(hex_circles, w, h, iters=100)
            if _is_valid(hex_circles):
                inits.append(hex_circles)
        
        # Random initializations
        for seed in range(5):
            if time_left() < 5.0:
                break
            rc = _random_initialization(w, h, seed + int(ar * 100))
            rc = _separate_circles(rc, w, h, iters=200)
            if _is_valid(rc):
                inits.append(rc)
        
        # Optimize each initialization
        for init in inits:
            if time_left() < 8.0:
                break
            
            # SCR
            scr_result = _successive_convex_relaxation(init, w, h, max_rounds=20)
            if _is_valid(scr_result):
                score = _sum_radii(scr_result)
                if score > best_score:
                    best_score = score
                    best_result = scr_result.copy()
            
            # Local optimize
            opt_result, opt_score = _local_optimize(scr_result if _is_valid(scr_result) else init, w, h)
            if opt_score > best_score:
                best_score = opt_score
                best_result = opt_result.copy()
            
            # Polish
            polished = _polish(best_result if best_result is not None else init, w, h, iters=30)
            if _is_valid(polished):
                pol_score = _sum_radii(polished)
                if pol_score > best_score:
                    best_score = pol_score
                    best_result = polished.copy()
    
    # Final polish on best result
    if best_result is not None:
        if time_left() > 5.0:
            w = 2.0 * aspect_ratios_sorted[0] / (1.0 + aspect_ratios_sorted[0])
            h = 2.0 / (1.0 + aspect_ratios_sorted[0])
            # Recompute w, h from bounding box of best_result
            xs, ys, rs = best_result[:, 0], best_result[:, 1], best_result[:, 2]
            w = np.max(xs + rs) - np.min(xs - rs)
            h = np.max(ys + rs) - np.min(ys - rs)
            # Shift to [0, w] x [0, h]
            xmin = np.min(xs - rs)
            ymin = np.min(ys - rs)
            best_result[:, 0] -= xmin
            best_result[:, 1] -= ymin
            
            polished = _polish(best_result, w, h, iters=50)
            if _is_valid(polished):
                pol_score = _sum_radii(polished)
                if pol_score > best_score:
                    best_score = pol_score
                    best_result = polished
    
    if best_result is not None and _is_valid(best_result):
        return best_result
    
    # Fallback: simple grid
    side = 5
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
