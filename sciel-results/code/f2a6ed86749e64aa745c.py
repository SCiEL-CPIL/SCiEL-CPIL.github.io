import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 145.0  # Safety margin before 150s kill

def _is_valid(x, y, r, w, h, tol=1e-6):
    if np.any(r < -tol):
        return False
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _smooth_penalty(x, y, r, w, h, beta=200.0):
    """Smooth penalty using soft-min for wall distances and soft-min for pairwise."""
    # Wall soft-min: for each circle, soft min of 4 wall distances
    # softmin(a,b) = -log(exp(-beta*a)+exp(-beta*b))/beta
    def softmin2(a, b):
        m = np.minimum(a, b)
        return m - np.logaddexp(-beta*(a - m), -beta*(b - m)) / beta

    dx1 = x - r          # left
    dx2 = w - x - r      # right
    dy1 = y - r          # bottom
    dy2 = h - y - r      # top
    wall = softmin2(softmin2(dx1, dx2), softmin2(dy1, dy2))
    wall_pen = np.maximum(0.0, -wall) ** 2

    # Pairwise: soft min of (dist - r_i - r_j) over all pairs
    # Compute all pair gaps
    gaps = np.zeros((N*(N-1)//2,))
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            gaps[idx] = d - r[i] - r[j]
            idx += 1
    # soft-min over all gaps
    m = np.min(gaps)
    soft_min_gap = m - np.log(np.sum(np.exp(-beta*(gaps - m)))) / beta
    pair_pen = np.maximum(0.0, -soft_min_gap) ** 2

    return np.sum(wall_pen) + pair_pen

def _force_relax(x, y, r, w, h, max_iter=150):
    x = x.copy(); y = y.copy()
    dt = 0.02
    for _ in range(max_iter):
        fx = np.zeros(N); fy = np.zeros(N)
        for i in range(N):
            if x[i] < r[i]: fx[i] += (r[i] - x[i]) * 10
            if x[i] > w - r[i]: fx[i] -= (x[i] - (w - r[i])) * 10
            if y[i] < r[i]: fy[i] += (r[i] - y[i]) * 10
            if y[i] > h - r[i]: fy[i] -= (y[i] - (h - r[i])) * 10
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]; dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-12:
                    dx = np.random.uniform(-0.01, 0.01); dy = np.random.uniform(-0.01, 0.01)
                    dist = np.sqrt(dx*dx + dy*dy)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    f = overlap * 5.0
                    fx[i] += (dx/dist)*f; fy[i] += (dy/dist)*f
                    fx[j] -= (dx/dist)*f; fy[j] -= (dy/dist)*f
        x += fx * dt; y += fy * dt
        for i in range(N):
            x[i] = np.clip(x[i], max(0.0, r[i]), w - max(0.0, r[i]))
            y[i] = np.clip(y[i], max(0.0, r[i]), h - max(0.0, r[i]))
    return x, y

def _hard_constraints(w, h):
    def cons_inside(v):
        x = v[0::3]; y = v[1::3]; r = v[2::3]
        return np.concatenate([x - r, w - x - r, y - r, h - y - r])
    def cons_non_overlap(v):
        x = v[0::3]; y = v[1::3]; r = v[2::3]
        cons = np.zeros((N*(N-1)//2,))
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                cons[idx] = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2) - r[i] - r[j]
                idx += 1
        return cons
    def cons_pos_r(v):
        return v[2::3]
    return [
        {'type': 'ineq', 'fun': cons_inside},
        {'type': 'ineq', 'fun': cons_non_overlap},
        {'type': 'ineq', 'fun': cons_pos_r}
    ]

def _optimize_aspect(w, h, start_time, max_restarts=4):
    best_score = -np.inf
    best_vars = None
    aspect_deadline = start_time + 30.0  # ~30s per aspect ratio

    for restart in range(max_restarts):
        if time.time() > aspect_deadline or time.time() > DEADLINE - 10:
            break

        # Grid initialization
        cols = max(2, int(np.sqrt(N * w / h)))
        rows = max(1, int(np.ceil(N / cols)))
        while cols * rows < N:
            if cols <= rows: cols += 1
            else: rows += 1

        positions = np.zeros((N, 2))
        idx = 0
        for r_i in range(rows):
            for c_i in range(cols):
                if idx >= N: break
                x = (c_i + 0.5) * w / cols
                y = (r_i + 0.5) * h / rows
                if restart > 0:
                    jitter = 0.15 * min(w/cols, h/rows)
                    x += np.random.uniform(-jitter, jitter)
                    y += np.random.uniform(-jitter, jitter)
                x = np.clip(x, 0.02, w - 0.02)
                y = np.clip(y, 0.02, h - 0.02)
                positions[idx] = [x, y]
                idx += 1

        # Initial radii from min neighbor distance
        min_dist = np.inf
        for i in range(N):
            min_dist = min(min_dist, positions[i,0], w-positions[i,0], positions[i,1], h-positions[i,1])
            for j in range(i+1, N):
                d = np.sqrt((positions[i,0]-positions[j,0])**2 + (positions[i,1]-positions[j,1])**2)
                min_dist = min(min_dist, d)
        r0 = 0.45 * min_dist
        radii = np.full(N, r0)

        x_rel, y_rel = _force_relax(positions[:,0], positions[:,1], radii, w, h)

        v0 = np.zeros(3*N)
        for i in range(N):
            v0[3*i] = x_rel[i]; v0[3*i+1] = y_rel[i]; v0[3*i+2] = radii[i]

        # Phase 1: Smooth penalty optimization (aggressive radius growth)
        def obj_smooth(v):
            x = v[0::3]; y = v[1::3]; r = v[2::3]
            return -np.sum(r) + 50.0 * _smooth_penalty(x, y, r, w, h, beta=200.0)

        try:
            res1 = minimize(obj_smooth, v0, method='L-BFGS-B',
                            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6})
            v1 = res1.x
            # Clamp radii to positive
            v1[2::3] = np.maximum(v1[2::3], 0.01)
        except Exception:
            v1 = v0.copy()

        # Phase 2: Hard-constraint SLSQP polish (multiple passes, decreasing tol)
        constraints = _hard_constraints(w, h)
        def obj_hard(v):
            return -np.sum(v[2::3])

        current = v1.copy()
        for polish_pass in range(3):
            if time.time() > aspect_deadline or time.time() > DEADLINE - 10:
                break
            ftol = [1e-9, 1e-11, 1e-13][polish_pass]
            maxiter = [200, 300, 500][polish_pass]
            try:
                res = minimize(obj_hard, current, method='SLSQP',
                               constraints=constraints,
                               options={'maxiter': maxiter, 'ftol': ftol})
                if res.success or res.fun < -np.sum(current[2::3]) - 1e-12:
                    # Verify validity
                    xr = res.x[0::3]; yr = res.x[1::3]; rr = res.x[2::3]
                    if _is_valid(xr, yr, rr, w, h):
                        current = res.x
            except Exception:
                pass

        # Final validity check and score
        xr = current[0::3]; yr = current[1::3]; rr = current[2::3]
        if _is_valid(xr, yr, rr, w, h):
            score = np.sum(rr)
            if score > best_score:
                best_score = score
                best_vars = current.copy()

    return best_score, best_vars

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None

    np.random.seed(42)

    aspect_ratios = [0.85, 0.95, 1.0, 1.1, 1.25, 1.4, 1.6]

    for ar in aspect_ratios:
        if time.time() > DEADLINE - 15:
            break
        w = 2.0 / (1.0 + ar)
        h = 2.0 - w
        if h <= 0.15:
            continue

        score, vars_ = _optimize_aspect(w, h, start_time, max_restarts=4)
        if score > best_score and vars_ is not None:
            best_score = score
            best_result = np.column_stack((vars_[0::3], vars_[1::3], vars_[2::3])).copy()

    # Fallback: safe 5x5 grid in unit square
    if best_result is None or best_score < 0:
        w = 1.0; h = 1.0; r = 0.09
        positions = []; radii = []; count = 0
        for i in range(5):
            for j in range(5):
                if count >= N: break
                positions.append([(i+0.5)/5, (j+0.5)/5])
                radii.append(r)
                count += 1
        best_result = np.column_stack((np.array(positions), np.array(radii)))
        best_score = np.sum(radii)

    return best_result
