import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _logsumexp(x, axis=None, keepdims=False):
    """Stable log-sum-exp implementation."""
    m = np.max(x, axis=axis, keepdims=True)
    # Handle case where max is -inf (all zeros/negatives)
    if np.any(np.isinf(m)):
        return m
    e = np.exp(x - m)
    s = np.sum(e, axis=axis, keepdims=keepdims)
    lse = np.log(s) + np.squeeze(m, axis=axis)
    return lse

def _vectorized_objective(z, beta=50.0):
    """
    Objective function: -sum(r) + penalty.
    Penalty uses LogSumExp to approximate max(0, violation).
    """
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    # Basic sanity checks
    if w < 0.1 or h < 0.1 or np.any(r < 0.001):
        return 1e6

    obj = -np.sum(r)

    # 1. Pairwise Overlap Penalty
    # We want to penalize if d_ij < r_i + r_j
    # Violation v_ij = (r_i + r_j) - d_ij
    # We use soft-max of violations.
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    d = np.sqrt(np.maximum(d_sq, 1e-12))
    
    r_sum = r[:, None] + r[None, :]
    violations_pair = r_sum - d
    
    # Mask out diagonal (self-interaction)
    mask = ~np.eye(N, dtype=bool)
    violations_pair = violations_pair[mask]
    
    # Soft penalty: beta * log(sum(exp(beta * v))) approximates max(0, v) for large beta
    # But we only want positive violations. 
    # Actually, LogSumExp of all violations (including negative) is dominated by the max.
    # If max < 0, LSE is small. If max > 0, LSE approximates max.
    # To be safe and smooth, we can use:
    # penalty = beta * log(sum(exp(beta * v))) 
    # This is convex.
    
    # However, LSE of negative numbers is not zero. 
    # A better soft max for max(0, v) is:
    # p = beta * log(1 + sum(exp(beta * v))) ? No, that's not right either.
    
    # Standard smooth max(0, v) is:
    # f(v) = 1/beta * log(1 + exp(beta * v))
    # This is the "soft plus" function.
    
    # Let's use soft plus for each violation and sum them?
    # Or LSE of the violations?
    # LSE(v) approximates max(v). If max(v) < 0, LSE is close to max(v) (negative).
    # We don't want to penalize negative violations (separated circles).
    
    # So, we should only sum the positive parts.
    # Smooth approximation of sum(max(0, v_i)):
    # Sum( 1/beta * log(1 + exp(beta * v_i)) )
    
    # Let's compute this.
    # To avoid overflow in exp(beta * v), we can shift.
    # But beta=50 and v is small, so exp(50*0.1) = exp(5) ~ 150. Safe.
    # If v is large, say 1, exp(50) is huge.
    # We should clip v or handle large values.
    
    # Alternative: Use quadratic penalty for large overlaps, smooth for small.
    # Or just use the soft plus with a cap.
    
    # Let's try: penalty = sum( 1/beta * log(1 + exp(beta * v)) )
    # For v < -10/beta, this is ~ 0.
    # For v > 10/beta, this is ~ v.
    
    # Implementation:
    # v is an array of shape (N*(N-1)/2, )
    # To be numerically stable:
    # soft_plus = 1/beta * log(1 + exp(beta * v))
    # We can compute log(1 + exp(x)) stably as:
    # np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))
    
    xv = beta * violations_pair
    # Stable log(1 + exp(xv))
    log_term = np.where(xv > 0, xv + np.log1p(np.exp(-xv)), np.log1p(np.exp(xv)))
    pair_penalty = np.sum(log_term) / beta
    
    obj += pair_penalty

    # 2. Boundary Penalty
    # Violations:
    # Left: r_i - x_i
    # Right: x_i + r_i - w
    # Bottom: r_i - y_i
    # Top: y_i + r_i - h
    
    viol_left = r - x
    viol_right = x + r - w
    viol_bottom = r - y
    viol_top = y + r - h
    
    all_viol = np.concatenate([viol_left, viol_right, viol_bottom, viol_top])
    
    xv_b = beta * all_viol
    log_term_b = np.where(xv_b > 0, xv_b + np.log1p(np.exp(-xv_b)), np.log1p(np.exp(xv_b)))
    boundary_penalty = np.sum(log_term_b) / beta
    
    obj += boundary_penalty

    return obj

def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(r <= 0):
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True

def _force_relax(z, max_iter=2000, dt=0.01):
    """Force-directed relaxation to resolve overlaps."""
    z = z.copy()
    n = N
    for _ in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w
        
        if w < 0.1 or h < 0.1:
            break
            
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall forces
        fx += np.maximum(r - x, 0) * 10
        fx -= np.maximum(x + r - w, 0) * 10
        fy += np.maximum(r - y, 0) * 10
        fy -= np.maximum(y + r - h, 0) * 10
        
        # Pairwise forces
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        d = np.sqrt(np.maximum(d_sq, 1e-12))
        r_sum = r[:, None] + r[None, :]
        
        overlap = np.maximum(r_sum - d, 0)
        # Force is proportional to overlap, direction is unit vector
        # Avoid division by zero
        mask = d > 1e-8
        ux = np.where(mask, dx / np.where(mask, d, 1), 0)
        uy = np.where(mask, dy / np.where(mask, d, 1), 0)
        
        # Symmetric force
        f_x = ux * overlap
        f_y = uy * overlap
        
        # Sum forces (each pair contributes to both i and j)
        # f_x[i,j] is force on i from j? 
        # dx = x_i - x_j. If overlap > 0, they are too close.
        # We want to push i away from j. Direction x_i - x_j is correct for i.
        # So force on i is sum over j of f_x[i,j].
        # Force on j is sum over i of -f_x[i,j] (or f_x[j,i] which is same).
        
        fx += np.sum(f_x, axis=1)
        fy += np.sum(f_y, axis=1)
        
        # Update
        x += dt * fx
        y += dt * fy
        
        # Clamp
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        
        z[0::3][:n] = x
        z[1::3][:n] = y
        
        # Check convergence
        if np.max(overlap) < 1e-6:
            break
            
    return z

def _polish(z, time_budget):
    """Polish the solution by trying to expand radii and move circles."""
    z = z.copy()
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    
    start_time = time.time()
    
    # Iterative expansion
    improved = True
    while improved and (time.time() - start_time) < time_budget:
        improved = False
        for i in range(n):
            # Try to expand r[i]
            # Max possible radius is limited by walls and other circles
            r_max = min(x[i], w - x[i], y[i], h - y[i])
            for j in range(n):
                if i == j: continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            
            if r_max > r[i] + 1e-5:
                # Binary search for new radius
                lo = r[i]
                hi = r_max
                for _ in range(20):
                    mid = (lo + hi) / 2
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(x, y, r_trial, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-6:
                    r[i] = lo
                    improved = True
            
            # If we couldn't expand or it's tight, try moving circle i
            # Try moving in 8 directions
            directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]
            for dx_dir, dy_dir in directions:
                norm = np.sqrt(dx_dir**2 + dy_dir**2)
                dx_dir /= norm
                dy_dir /= norm
                for step in [0.05, 0.02, 0.01, 0.005]:
                    new_x = x[i] + dx_dir * step
                    new_y = y[i] + dy_dir * step
                    # Check if valid
                    if new_x < r[i] or new_x > w - r[i]: continue
                    if new_y < r[i] or new_y > h - r[i]: continue
                    x_trial = x.copy()
                    y_trial = y.copy()
                    x_trial[i] = new_x
                    y_trial[i] = new_y
                    if _is_valid(x_trial, y_trial, r, w, h):
                        x[i] = new_x
                        y[i] = new_y
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break # Restart loop if we moved something

    # Final uniform expansion
    for _ in range(50):
        if (time.time() - start_time) > time_budget: break
        r_trial = r * 1.001
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w
    
    if _is_valid(x, y, r, w, h):
        return z_new
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_value = 0.0
    best_result = None
    
    def objective(z):
        return _vectorized_objective(z, beta=50.0)
    
    def check_validity(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            return float(np.sum(r))
        return 0.0

    # Initial guesses
    guesses = []
    
    # 1. Hexagonal grid
    for w0 in [1.0, 0.8, 1.2, 0.7, 1.3]:
        h0 = 2.0 - w0
        if h0 < 0.3: continue
        z = np.zeros(3 * N + 1)
        z[-1] = w0
        r_est = np.sqrt(w0 * h0 / (N * np.pi * 0.9))
        pts = []
        y = r_est
        row = 0
        while y < h0 - r_est * 0.5:
            x_off = 0 if row % 2 == 0 else r_est
            x = r_est + x_off
            while x < w0 - r_est * 0.5:
                if x >= r_est and x <= w0 - r_est and y >= r_est and y <= h0 - r_est:
                    pts.append((x, y))
                x += 2 * r_est
            y += r_est * np.sqrt(3)
            row += 1
        if len(pts) >= N:
            for i in range(N):
                cx, cy = pts[i]
                z[3*i] = cx
                z[3*i+1] = cy
                z[3*i+2] = r_est
            guesses.append(z)
        else:
            # Fallback to smaller radius
            for scale in [0.9, 0.8, 0.7]:
                r_est_new = r_est * scale
                pts = []
                y = r_est_new
                row = 0
                while y < h0 - r_est_new * 0.5:
                    x_off = 0 if row % 2 == 0 else r_est_new
                    x = r_est_new + x_off
                    while x < w0 - r_est_new * 0.5:
                        if x >= r_est_new and x <= w0 - r_est_new and y >= r_est_new and y <= h0 - r_est_new:
                            pts.append((x, y))
                        x += 2 * r_est_new
                    y += r_est_new * np.sqrt(3)
                    row += 1
                if len(pts) >= N:
                    for i in range(N):
                        cx, cy = pts[i]
                        z[3*i] = cx
                        z[3*i+1] = cy
                        z[3*i+2] = r_est_new
                    guesses.append(z)
                    break

    # 2. Square grid
    side = int(np.ceil(np.sqrt(N)))
    for w0 in [1.0, 0.8, 1.2, 0.7, 1.3]:
        h0 = 2.0 - w0
        if h0 < 0.3: continue
        z = np.zeros(3 * N + 1)
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side * w0
            cy = (row + 0.5) / side * h0
            r0 = min(w0 / side, h0 / side) * 0.45
            z[3*i] = cx
            z[3*i+1] = cy
            z[3*i+2] = r0
        z[-1] = w0
        guesses.append(z)
        
    # 3. Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        w0 = rng.uniform(0.6, 1.4)
        h0 = 2.0 - w0
        z = np.zeros(3 * N + 1)
        for i in range(N):
            r0 = rng.uniform(0.05, 0.12)
            cx = rng.uniform(r0, w0 - r0)
            cy = rng.uniform(r0, h0 - r0)
            z[3*i] = cx
            z[3*i+1] = cy
            z[3*i+2] = r0
        z[-1] = w0
        guesses.append(z)

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 20:
            break
        
        # Pre-relax
        z0_relaxed = _force_relax(z0, max_iter=500)
        
        # SLSQP
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(500, max(100, remaining * 50)))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})
            
            z_opt = res.x
            val = check_validity(z_opt)
            
            if val > best_value:
                best_value = val
                best_result = z_opt.copy()
        except Exception:
            continue

    # Polish
    if best_result is not None and time.time() < deadline - 20:
        polished = _polish(best_result, deadline - (time.time() - start_time) - 10)
        if polished is not None:
            val = check_validity(polished)
            if val > best_value:
                best_value = val
                best_result = polished

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w
    
    # Safety shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))

