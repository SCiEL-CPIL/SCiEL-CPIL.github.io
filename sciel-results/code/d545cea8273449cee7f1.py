"""Circle packing in perimeter-4 rectangle: vectorized SLSQP + expansion polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
TIME_LIMIT = 145.0


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if packing is valid."""
    if W <= 0.01 or H <= 0.01:
        return False
    if np.any(r <= 0):
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    # Vectorized pairwise check
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d2 = dx * dx + dy * dy
    rs = r[:, None] + r[None, :]
    # Only check upper triangle
    iu = np.triu_indices(N, 1)
    if np.any(d2[iu] < rs[iu] ** 2 - tol):
        return False
    return True


def _compute_violation(cx, cy, r, W, H):
    """Compute total violation amount (0 if valid)."""
    viol = 0.0
    if W <= 0.01 or H <= 0.01:
        return 1e10
    if np.any(r <= 0):
        return 1e10
    viol += np.max(np.maximum(0, r - cx))
    viol += np.max(np.maximum(0, r - cy))
    viol += np.max(np.maximum(0, cx + r - W))
    viol += np.max(np.maximum(0, cy + r - H))
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d2 = dx * dx + dy * dy
    rs = r[:, None] + r[None, :]
    iu = np.triu_indices(N, 1)
    overlap = np.maximum(0, rs[iu] - np.sqrt(np.maximum(0, d2[iu])))
    viol += np.sum(overlap)
    return viol


def _max_radius_for_circle(cx, cy, r, W, H, i):
    """Max radius for circle i given others fixed."""
    r_wall = min(cx[i], W - cx[i], cy[i], H - cy[i])
    r_max = r_wall
    for j in range(N):
        if i == j:
            continue
        dx = cx[i] - cx[j]
        dy = cy[i] - cy[j]
        d = np.sqrt(dx * dx + dy * dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)


def _max_radius_vectorized(cx, cy, r, W, H, i):
    """Vectorized max radius for circle i."""
    r_wall = np.min(np.array([cx[i], W - cx[i], cy[i], H - cy[i]]))
    others = np.arange(N) != i
    dx = cx[i] - cx[others]
    dy = cy[i] - cy[others]
    d = np.sqrt(dx * dx + dy * dy)
    r_allowed = d - r[others]
    r_max = min(r_wall, np.min(r_allowed))
    return max(0.0, r_max)


def _is_valid_move(cx, cy, r, W, H, i, new_x, new_y):
    """Check if moving circle i to (new_x, new_y) is valid."""
    if new_x < r[i] or new_x > W - r[i]:
        return False
    if new_y < r[i] or new_y > H - r[i]:
        return False
    for j in range(N):
        if i == j:
            continue
        dx = new_x - cx[j]
        dy = new_y - cy[j]
        d2 = dx * dx + dy * dy
        r_sum = r[i] + r[j]
        if d2 < r_sum * r_sum - 1e-10:
            return False
    return True


def _force_relax(cx, cy, r, W, H, max_iter=2000, time_budget=None):
    """Force-directed relaxation to resolve overlaps and push to boundaries."""
    n = N
    dt = 0.003
    for it in range(max_iter):
        if time_budget is not None and time.time() > time_budget:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall forces
        for i in range(n):
            if cx[i] < r[i]:
                fx[i] += (r[i] - cx[i]) * 50
            if cx[i] > W - r[i]:
                fx[i] -= (cx[i] - (W - r[i])) * 50
            if cy[i] < r[i]:
                fy[i] += (r[i] - cy[i]) * 50
            if cy[i] > H - r[i]:
                fy[i] -= (cy[i] - (H - r[i])) * 50

        # Pairwise repulsion
        max_overlap = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_sum = r[i] + r[j]
                if d < r_sum:
                    overlap = r_sum - d
                    max_overlap = max(max_overlap, overlap)
                    if d > 1e-10:
                        ux = dx / d
                        uy = dy / d
                        f = overlap * 20.0
                        fx[i] += ux * f
                        fy[i] += uy * f
                        fx[j] -= ux * f
                        fy[j] -= uy * f

        if max_overlap < 1e-9:
            break

        cx = cx + dt * fx
        cy = cy + dt * fy
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)

    return cx, cy


def _simulated_annealing_polish(cx, cy, r, W, H, time_budget, n_iter=5000):
    """Simulated annealing to escape local minima."""
    n = N
    best_cx, best_cy, best_r = cx.copy(), cy.copy(), r.copy()
    best_val = np.sum(r)
    current_val = np.sum(r)
    
    T = 0.01
    T_min = 1e-6
    cooling = 0.995
    
    for it in range(n_iter):
        if time.time() > time_budget:
            break
        T = max(T_min, T * cooling)
        
        # Random perturbation
        i = np.random.randint(n)
        delta_r = np.random.uniform(-0.005, 0.005)
        delta_x = np.random.uniform(-0.02, 0.02)
        delta_y = np.random.uniform(-0.02, 0.02)
        
        new_r = r[i] + delta_r
        if new_r <= 0:
            continue
        new_x = cx[i] + delta_x
        new_y = cy[i] + delta_y
        
        # Check bounds
        if new_x < new_r or new_x > W - new_r or new_y < new_r or new_y > H - new_r:
            continue
        
        # Check overlaps with others
        valid = True
        for j in range(n):
            if i == j:
                continue
            dx = new_x - cx[j]
            dy = new_y - cy[j]
            d2 = dx * dx + dy * dy
            r_sum = new_r + r[j]
            if d2 < r_sum * r_sum - 1e-10:
                valid = False
                break
        
        if not valid:
            continue
        
        # Compute new value
        new_val = current_val - r[i] + new_r
        delta = new_val - current_val
        
        # Accept if better or with probability exp(-delta/T)
        if delta > 0 or np.random.rand() < np.exp(-delta / T):
            cx[i] = new_x
            cy[i] = new_y
            r[i] = new_r
            current_val = new_val
            
            if current_val > best_val:
                best_val = current_val
                best_cx, best_cy, best_r = cx.copy(), cy.copy(), r.copy()
    
    return best_cx, best_cy, best_r


def _expand_all_radii(cx, cy, r, W, H, time_budget):
    """Iteratively expand each circle's radius to its maximum possible."""
    n = N
    improved = True
    max_rounds = 50
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            r_max = _max_radius_for_circle(cx, cy, r, W, H, i)
            if r_max > r[i] + 1e-8:
                lo, hi = r[i], r_max
                for _ in range(30):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(cx, cy, r_trial, W, H):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-9:
                    r[i] = lo
                    improved = True
        if not improved:
            break
    return cx, cy, r


def _expansion_polish(cx, cy, r, W, H, time_budget):
    """Iteratively expand radii and shift positions to create space."""
    n = N
    for rnd in range(40):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            r_max = _max_radius_for_circle(cx, cy, r, W, H, i)
            if r_max > r[i] + 1e-7:
                lo, hi = r[i], r_max
                for _ in range(25):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(cx, cy, r_trial, W, H):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True

            if r_max <= r[i] + 1e-7:
                min_gap = 1e9
                move_dx, move_dy = 0.0, 0.0
                gaps = [
                    (cx[i] - r[i], 1, 0),
                    (W - cx[i] - r[i], -1, 0),
                    (cy[i] - r[i], 0, 1),
                    (H - cy[i] - r[i], 0, -1),
                ]
                for gap, dx_dir, dy_dir in gaps:
                    if gap < min_gap:
                        min_gap = gap
                        move_dx, move_dy = dx_dir, dy_dir
                for j in range(n):
                    if i == j:
                        continue
                    dx = cx[i] - cx[j]
                    dy = cy[i] - cy[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    gap = d - (r[i] + r[j])
                    if gap < min_gap:
                        min_gap = gap
                        if d > 1e-10:
                            move_dx = dx / d
                            move_dy = dy / d

                if min_gap < 0.03:
                    for step in [0.04, 0.02, 0.01, 0.005, 0.002]:
                        new_x = cx[i] + move_dx * step
                        new_y = cy[i] + move_dy * step
                        if _is_valid_move(cx, cy, r, W, H, i, new_x, new_y):
                            cx[i] = new_x
                            cy[i] = new_y
                            improved = True
                            break
                    else:
                        perp_dx, perp_dy = -move_dy, move_dx
                        for step in [0.02, 0.01, 0.005]:
                            for s in [1.0, -1.0]:
                                new_x = cx[i] + s * perp_dx * step
                                new_y = cy[i] + s * perp_dy * step
                                if _is_valid_move(cx, cy, r, W, H, i, new_x, new_y):
                                    cx[i] = new_x
                                    cy[i] = new_y
                                    improved = True
                                    break
                            if improved:
                                break
        if not improved:
            break

    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0005
        if _is_valid(cx, cy, r_trial, W, H):
            r = r_trial
        else:
            break

    return cx, cy, r


def _slsqp_optimize(cx, cy, r, W, H, time_budget, maxiter=500):
    """SLSQP with explicit constraints to maximize sum of radii."""
    n = N
    z0 = np.concatenate([cx, cy, r, [W]])
    h0 = H

    def objective(zv):
        r_v = zv[2*n:3*n]
        return -np.sum(r_v)

    constraints = []
    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3*i] - p[3*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*n+i+1]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3*i+1] - p[3*n+i+1]})

    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (
                    (p[3*i] - p[3*j])**2 +
                    (p[3*i+1] - p[3*j+1])**2 -
                    (p[3*n+i] + p[3*n+j])**2
                )
            })

    try:
        res = minimize(objective, z0, method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})
        z_opt = res.x
        cx_new = z_opt[0:n]
        cy_new = z_opt[n:2*n]
        r_new = z_opt[2*n:3*n]
        W_new = z_opt[-1]
        H_new = 2.0 - W_new
        if _is_valid(cx_new, cy_new, r_new, W_new, H_new):
            return cx_new, cy_new, r_new, W_new, H_new
    except Exception:
        pass
    return None


def _high_precision_polish(cx, cy, r, W, H, time_budget):
    """High-precision polish with small perturbations and radius expansion."""
    n = N
    best_cx, best_cy, best_r = cx.copy(), cy.copy(), r.copy()
    best_val = np.sum(r)
    
    # Multiple passes with decreasing step sizes
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for step in step_sizes:
        if time.time() > time_budget:
            break
        
        improved = True
        max_rounds = 20
        for rnd in range(max_rounds):
            if time.time() > time_budget:
                break
            improved = False
            
            for i in range(n):
                # Try to expand radius
                r_max = _max_radius_vectorized(cx, cy, r, W, H, i)
                if r_max > r[i] + 1e-8:
                    lo, hi = r[i], r_max
                    for _ in range(30):
                        mid = (lo + hi) / 2.0
                        r_trial = r.copy()
                        r_trial[i] = mid
                        if _is_valid(cx, cy, r_trial, W, H):
                            lo = mid
                        else:
                            hi = mid
                    if lo > r[i] + 1e-9:
                        r[i] = lo
                        improved = True
                        if np.sum(r) > best_val:
                            best_val = np.sum(r)
                            best_cx, best_cy, best_r = cx.copy(), cy.copy(), r.copy()
                
                # Try small position perturbations in 8 directions
                if r_max <= r[i] + 1e-7:
                    directions = [
                        (1, 0), (-1, 0), (0, 1), (0, -1),
                        (1, 1), (1, -1), (-1, 1), (-1, -1)
                    ]
                    for dx_dir, dy_dir in directions:
                        for s in [step, step * 0.5, step * 0.25]:
                            new_x = cx[i] + dx_dir * s
                            new_y = cy[i] + dy_dir * s
                            
                            if new_x < r[i] or new_x > W - r[i] or new_y < r[i] or new_y > H - r[i]:
                                continue
                            
                            valid = True
                            for j in range(n):
                                if i == j:
                                    continue
                                dx = new_x - cx[j]
                                dy = new_y - cy[j]
                                d2 = dx * dx + dy * dy
                                r_sum = r[i] + r[j]
                                if d2 < r_sum * r_sum - 1e-10:
                                    valid = False
                                    break
                            
                            if not valid:
                                continue
                            
                            # Check if this position allows larger radius
                            new_r_max = _max_radius_vectorized(cx, cy, r, W, H, i)
                            # Temporarily move
                            old_x, old_y = cx[i], cy[i]
                            cx[i], cy[i] = new_x, new_y
                            r_max_new = _max_radius_vectorized(cx, cy, r, W, H, i)
                            cx[i], cy[i] = old_x, old_y
                            
                            if r_max_new > r[i] + 1e-7:
                                cx[i], cy[i] = new_x, new_y
                                improved = True
                                if np.sum(r) > best_val:
                                    best_val = np.sum(r)
                                    best_cx, best_cy, best_r = cx.copy(), cy.copy(), r.copy()
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
    
    # Final uniform expansion
    for _ in range(500):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0001
        if _is_valid(cx, cy, r_trial, W, H):
            r = r_trial
        else:
            break
    
    if np.sum(r) > best_val:
        best_cx, best_cy, best_r = cx.copy(), cy.copy(), r.copy()
    
    return best_cx, best_cy, best_r


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_value = 0.0
    best_cx = best_cy = best_r = None
    best_W = best_H = None

    # Start from 5x5 grid in unit square
    side = 5
    W0, H0 = 1.0, 1.0
    r0 = 0.5 / side * 0.95

    cx = np.zeros(N)
    cy = np.zeros(N)
    r = np.zeros(N)
    for i in range(N):
        row, col = divmod(i, side)
        cx[i] = (col + 0.5) / side * W0
        cy[i] = (row + 0.5) / side * H0
        r[i] = r0

    cx, cy = _force_relax(cx, cy, r, W0, H0, max_iter=500)
    cx, cy, r = _expand_all_radii(cx, cy, r, W0, H0, deadline - 30)
    val = np.sum(r)
    if _is_valid(cx, cy, r, W0, H0):
        best_value = val
        best_cx, best_cy, best_r = cx.copy(), cy.copy(), r.copy()
        best_W, best_H = W0, H0

    # Try multiple aspect ratios with SLSQP
    if time.time() < deadline - 60:
        aspect_ratios = [1.0, 0.85, 1.15, 0.75, 1.25, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3]
        for w0 in aspect_ratios:
            if time.time() > deadline - 50:
                break
            h0 = 2.0 - w0
            if h0 < 0.3 or w0 < 0.3:
                continue

            cx_t = np.zeros(N)
            cy_t = np.zeros(N)
            r_t = np.zeros(N)
            for i in range(N):
                row, col = divmod(i, side)
                cx_t[i] = (col + 0.5) / side * w0
                cy_t[i] = (row + 0.5) / side * h0
                r_t[i] = min(w0, h0) / side * 0.9

            cx_t, cy_t = _force_relax(cx_t, cy_t, r_t, w0, h0, max_iter=300)
            cx_t, cy_t, r_t = _expand_all_radii(cx_t, cy_t, r_t, w0, h0, deadline - 40)

            if _is_valid(cx_t, cy_t, r_t, w0, h0):
                val_t = np.sum(r_t)
                if val_t > best_value:
                    best_value = val_t
                    best_cx, best_cy, best_r = cx_t.copy(), cy_t.copy(), r_t.copy()
                    best_W, best_H = w0, h0

                # Try SLSQP on this starting point
                if time.time() < deadline - 30:
                    result = _slsqp_optimize(cx_t, cy_t, r_t, w0, h0, deadline - 25, maxiter=300)
                    if result is not None:
                        cx_n, cy_n, r_n, W_n, H_n = result
                        val_n = np.sum(r_n)
                        if val_n > best_value:
                            best_value = val_n
                            best_cx, best_cy, best_r = cx_n, cy_n, r_n
                            best_W, best_H = W_n, H_n

    # High-precision polish
    if best_cx is not None and time.time() < deadline - 20:
        cx_p, cy_p, r_p = _high_precision_polish(best_cx, best_cy, best_r, best_W, best_H, deadline - 5)
        if _is_valid(cx_p, cy_p, r_p, best_W, best_H):
            val_p = np.sum(r_p)
            if val_p > best_value:
                best_value = val_p
                best_cx, best_cy, best_r = cx_p, cy_p, r_p

    # Final expansion polish
    if best_cx is not None and time.time() < deadline - 10:
        cx_p, cy_p, r_p = _expansion_polish(best_cx, best_cy, best_r, best_W, best_H, deadline - 5)
        if _is_valid(cx_p, cy_p, r_p, best_W, best_H):
            val_p = np.sum(r_p)
            if val_p > best_value:
                best_value = val_p
                best_cx, best_cy, best_r = cx_p, cy_p, r_p

    if best_cx is None:
        cx = np.zeros(N)
        cy = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, 5)
            cx[i] = (col + 0.5) / 5
            cy[i] = (row + 0.5) / 5
            r[i] = 0.09
        return np.column_stack((cx, cy, r))

    for _ in range(5):
        r_trial = best_r * (1 - 1e-10)
        if _is_valid(best_cx, best_cy, r_trial, best_W, best_H):
            best_r = r_trial
        else:
            break

    return np.column_stack((best_cx, best_cy, best_r))
