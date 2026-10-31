"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _penalty_objective(params, n, w, h):
    """Smooth penalty objective for L-BFGS-B.
    params: [x1, y1, r1, x2, y2, r2, ...]
    Returns (value, grad) for use with L-BFGS-B.
    """
    xs = params[0::3].copy()
    ys = params[1::3].copy()
    rs = params[2::3].copy()
    
    # Clip radii to small positive
    rs = np.maximum(rs, 1e-8)
    
    val = -np.sum(rs)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0  # d(-sum(r))/dr_i = -1
    
    penalty_weight = 50.0
    
    # Bounds constraints (soft)
    for i in range(n):
        # x_i >= r_i
        viol = rs[i] - xs[i]
        if viol > 0:
            val += penalty_weight * viol * viol
            grad[3*i] += -2.0 * penalty_weight * viol       # d/dx: -2*w*viol
            grad[3*i+2] += 2.0 * penalty_weight * viol      # d/dr: +2*w*viol
        
        # y_i >= r_i
        viol = rs[i] - ys[i]
        if viol > 0:
            val += penalty_weight * viol * viol
            grad[3*i+1] += -2.0 * penalty_weight * viol
            grad[3*i+2] += 2.0 * penalty_weight * viol
        
        # x_i + r_i <= w
        viol = xs[i] + rs[i] - w
        if viol > 0:
            val += penalty_weight * viol * viol
            grad[3*i] += 2.0 * penalty_weight * viol
            grad[3*i+2] += 2.0 * penalty_weight * viol
        
        # y_i + r_i <= h
        viol = ys[i] + rs[i] - h
        if viol > 0:
            val += penalty_weight * viol * viol
            grad[3*i+1] += 2.0 * penalty_weight * viol
            grad[3*i+2] += 2.0 * penalty_weight * viol
    
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx * dx + dy * dy
            min_dist = rs[i] + rs[j]
            if dist_sq < min_dist * min_dist:
                dist = np.sqrt(dist_sq)
                if dist < 1e-12:
                    # Degenerate: push apart in x
                    viol = min_dist
                    val += penalty_weight * viol * viol
                    grad[3*i] += 2.0 * penalty_weight * viol
                    grad[3*j] -= 2.0 * penalty_weight * viol
                    grad[3*i+2] += 2.0 * penalty_weight * viol
                    grad[3*j+2] += 2.0 * penalty_weight * viol
                else:
                    viol = min_dist - dist
                    if viol > 0:
                        val += penalty_weight * viol * viol
                        ux = dx / dist
                        uy = dy / dist
                        # d(viol)/dx_i = -ux, d(viol)/dy_i = -uy
                        # d(viol)/dx_j = +ux, d(viol)/dy_j = +uy
                        # d(viol)/dr_i = 1, d(viol)/dr_j = 1
                        factor = 2.0 * penalty_weight * viol
                        grad[3*i] += factor * ux
                        grad[3*i+1] += factor * uy
                        grad[3*j] -= factor * ux
                        grad[3*j+1] -= factor * uy
                        grad[3*i+2] += factor
                        grad[3*j+2] += factor
    
    return val, grad


def _validate(params, n, w, h, tol=1e-7):
    """Check if packing is valid."""
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    for i in range(n):
        if rs[i] < 1e-8:
            return False
        if xs[i] < rs[i] - tol:
            return False
        if ys[i] < rs[i] - tol:
            return False
        if xs[i] + rs[i] > w + tol:
            return False
        if ys[i] + rs[i] > h + tol:
            return False
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    
    return True


def _expansion_polish(params, n, w, h, time_limit=10.0):
    """Greedy expansion: try to increase each radius as much as possible."""
    xs = params[0::3].copy()
    ys = params[1::3].copy()
    rs = params[2::3].copy()
    
    start = time.time()
    for _ in range(50):
        if time.time() - start > time_limit:
            break
        improved = False
        for i in range(n):
            # Binary search for max radius for circle i
            lo = rs[i]
            hi = min(w - xs[i], h - ys[i], xs[i], ys[i])
            # Also limited by neighbors
            for j in range(n):
                if i != j:
                    d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    hi = min(hi, d - rs[j])
            hi = max(hi, lo)
            
            for _ in range(30):
                if time.time() - start > time_limit:
                    break
                mid = (lo + hi) / 2
                if mid <= lo:
                    break
                # Check if radius mid is feasible for circle i
                ok = True
                if xs[i] < mid or ys[i] < mid:
                    ok = False
                if xs[i] + mid > w or ys[i] + mid > h:
                    ok = False
                if ok:
                    for j in range(n):
                        if i == j:
                            continue
                        d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                        if d < mid + rs[j]:
                            ok = False
                            break
                if ok:
                    lo = mid
                    improved = True
                else:
                    hi = mid
            rs[i] = lo
        
        if not improved:
            break
    
    return np.concatenate([xs, ys, rs])


def _make_initial(n, w, h, seed=0):
    """Create initial positions using hexagonal grid."""
    rng = np.random.RandomState(seed)
    
    # Estimate radius from hex packing
    area = w * h
    r_est = np.sqrt(area / (n * 2 * np.sqrt(3)))
    dx = 2.2 * r_est
    dy = np.sqrt(3) * r_est
    
    pts = []
    row = 0
    while len(pts) < n + 20:
        y = dy * row
        if y > h:
            break
        x_offset = dx / 2 if row % 2 else 0
        col = 0
        while True:
            x = dx * col + x_offset
            if x > w:
                break
            if 0 <= x <= w and 0 <= y <= h:
                pts.append((x, y))
            col += 1
        row += 1
    
    if len(pts) < n:
        side = int(np.ceil(np.sqrt(n)))
        pts = [((i + 0.5) / side * w, (j + 0.5) / side * h) for i in range(side) for j in range(side)]
    
    rng.shuffle(pts)
    pts = np.array(pts[:n])
    
    # Small initial radius
    r_init = min(w, h) / (2 * np.ceil(np.sqrt(n))) * 0.5
    params = np.zeros(n * 3)
    for i in range(n):
        params[3*i] = pts[i, 0]
        params[3*i+1] = pts[i, 1]
        params[3*i+2] = r_init
    
    return params


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin for 150s limit
    
    best_score = -1.0
    best_result = None
    
    # Rectangle with perimeter 4: 2*(w+h) = 4, so w+h = 2
    # Focus on aspect ratios near 1.0 where packing density is typically best
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4, 0.5, 1.5]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 20.0:
            break
        
        w = ar
        h = 2.0 - ar
        if w <= 0 or h <= 0:
            continue
        
        for seed in range(3):
            if time.time() > deadline - 15.0:
                break
            
            params = _make_initial(N, w, h, seed)
            
            # L-BFGS-B optimization
            try:
                result = minimize(
                    _penalty_objective,
                    params,
                    args=(N, w, h),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 500}
                )
            except Exception:
                continue
            
            if result.success or result.fun < 0:
                params_opt = result.x
            else:
                params_opt = result.x
            
            # Clamp to valid region
            xs = params_opt[0::3]
            ys = params_opt[1::3]
            rs = params_opt[2::3]
            
            # Clamp positions and radii
            for i in range(N):
                rs[i] = max(rs[i], 1e-8)
                rs[i] = min(rs[i], min(xs[i], w - xs[i], ys[i], h - ys[i]))
                xs[i] = np.clip(xs[i], 0, w)
                ys[i] = np.clip(ys[i], 0, h)
            
            params_opt = np.concatenate([xs, ys, rs])
            
            # Check validity
            if _validate(params_opt, N, w, h):
                score = np.sum(rs)
                if score > best_score:
                    best_score = score
                    best_result = params_opt.copy()
            
            # Expansion polish
            if time.time() < deadline - 10.0:
                params_exp = _expansion_polish(params_opt, N, w, h, time_limit=5.0)
                if _validate(params_exp, N, w, h):
                    score = np.sum(params_exp[2::3])
                    if score > best_score:
                        best_score = score
                        best_result = params_exp.copy()
            
            # Second L-BFGS-B pass from expanded state
            if time.time() < deadline - 10.0 and best_result is not None:
                try:
                    result2 = minimize(
                        _penalty_objective,
                        params_exp,
                        args=(N, w, h),
                        method='L-BFGS-B',
                        jac=True,
                        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 300}
                    )
                    params2 = result2.x
                    xs2 = params2[0::3]
                    ys2 = params2[1::3]
                    rs2 = params2[2::3]
                    for i in range(N):
                        rs2[i] = max(rs2[i], 1e-8)
                        rs2[i] = min(rs2[i], min(xs2[i], w - xs2[i], ys2[i], h - ys2[i]))
                        xs2[i] = np.clip(xs2[i], 0, w)
                        ys2[i] = np.clip(ys2[i], 0, h)
                    params2 = np.concatenate([xs2, ys2, rs2])
                    
                    if _validate(params2, N, w, h):
                        score2 = np.sum(rs2)
                        if score2 > best_score:
                            best_score = score2
                            best_result = params2.copy()
                except Exception:
                    pass
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return np.column_stack([best_result[0::3], best_result[1::3], best_result[2::3]])

# EVOLVE-BLOCK-END
