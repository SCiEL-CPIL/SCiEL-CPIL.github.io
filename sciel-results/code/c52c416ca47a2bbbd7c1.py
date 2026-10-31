"""Circle packing in perimeter-4 rectangle: Optimized Vectorized SLSQP + Force Relaxation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _soft_max(v, beta=300.0):
    """Smooth approximation of max(v, 0)."""
    m = np.maximum(v, 0.0)
    return m + np.logaddexp(0.0, beta * (v - m)) / beta

def _vectorized_objective(z, penalty_weight=1000.0):
    """Vectorized objective with smooth penalties."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    overlap = _soft_max(r_sum_pairs - d_pairs, beta=300.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    # Boundary penalty
    left_viol = _soft_max(r - x, beta=300.0)
    right_viol = _soft_max(x + r - w, beta=300.0)
    bottom_viol = _soft_max(r - y, beta=300.0)
    top_viol = _soft_max(y + r - h, beta=300.0)

    obj += penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    # Aspect ratio constraint
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj

def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid. Vectorized where possible."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    
    # Pairwise distance check
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d_sq = dx * dx + dy * dy
    r_sum_sq = (r[:, None] + r[None, :]) ** 2
    
    # Check upper triangle
    iu = np.triu_indices(N, k=1)
    if np.any(d_sq[iu] < r_sum_sq[iu] - tol):
        return False
        
    return True

def _force_relax_vectorized(x, y, r, w, h, max_iter=1000, dt=0.002):
    """Vectorized force-directed relaxation to resolve overlaps."""
    x = x.copy()
    y = y.copy()
    n = N
    iu = np.triu_indices(n, k=1)
    
    for _ in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        
        fx -= left_viol * 10
        fx += right_viol * 10
        fy -= bottom_viol * 10
        fy += top_viol * 10

        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        
        if np.max(overlap) < 1e-6:
            break
            
        # Force magnitude proportional to overlap
        f_mag = overlap * 1.0
        dx_pairs = dx[iu]
        dy_pairs = dy[iu]
        
        # Unit vectors
        ux = dx_pairs / d_pairs
        uy = dy_pairs / d_pairs
        
        # Accumulate forces using add.at to avoid Python loops
        # iu[0] is i, iu[1] is j
        # Force on i is +F, on j is -F
        np.add.at(fx, iu[0], ux * f_mag)
        np.add.at(fy, iu[0], uy * f_mag)
        np.add.at(fx, iu[1], -ux * f_mag)
        np.add.at(fy, iu[1], -uy * f_mag)

        # Adaptive dt: reduce when forces are large
        max_force = max(np.max(np.abs(fx)), np.max(np.abs(fy)), 1e-12)
        effective_dt = min(dt, 0.01 / max_force)
        
        # Update
        x += effective_dt * fx
        y += effective_dt * fy
        
        # Clamp
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    return x, y

def _max_radius_for_circle(x, y, r, w, h, i):
    """Calculate the maximum possible radius for circle i given current positions."""
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    
    # Check constraints from other circles
    dx = x[i] - x
    dy = y[i] - y
    d = np.sqrt(dx*dx + dy*dy)
    d[i] = np.inf  # Ignore self
    r_allowed = d - r
    r_max = min(r_max, np.min(r_allowed))
    
    return max(0.0, r_max)

def _is_valid_move(x, y, r, w, h, i, new_r, tol=1e-8):
    """Check if expanding circle i to new_r is valid."""
    # Check walls
    if x[i] - new_r < -tol or w - x[i] - new_r < -tol:
        return False
    if y[i] - new_r < -tol or h - y[i] - new_r < -tol:
        return False
    
    # Check other circles
    for j in range(N):
        if j == i:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx*dx + dy*dy)
        if d < r[j] + new_r - tol:
            return False
    return True

def _constraint_refine(x, y, r, w, h, i, new_r, max_iter=200, tol=1e-8):
    """Try to move circle i (and potentially others) to make new_r feasible."""
    x = x.copy()
    y = y.copy()
    
    # First, try just moving circle i
    for _ in range(max_iter):
        # Compute forces on circle i from constraints
        fx = 0.0
        fy = 0.0
        
        # Wall forces
        if x[i] - new_r < 0:
            fx += (new_r - x[i]) * 50
        if w - x[i] - new_r < 0:
            fx -= (w - x[i] - new_r) * 50
        if y[i] - new_r < 0:
            fy += (new_r - y[i]) * 50
        if h - y[i] - new_r < 0:
            fy -= (h - y[i] - new_r) * 50
        
        # Circle-circle forces
        for j in range(N):
            if j == i:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < 1e-12:
                continue
            overlap = r[j] + new_r - d
            if overlap > 0:
                fx += (dx / d) * overlap * 50
                fy += (dy / d) * overlap * 50
        
        # Update position of circle i
        x[i] += 0.001 * fx
        y[i] += 0.001 * fy
        
        # Clamp
        x[i] = np.clip(x[i], new_r, w - new_r)
        y[i] = np.clip(y[i], new_r, h - new_r)
        
        # Check if valid
        if _is_valid_move(x, y, r, w, h, i, new_r, tol):
            # Check all other circles are still valid
            if _is_valid(x, y, r, w, h, tol):
                return x, y, True
    
    return x, y, False

def _expand_radii_vectorized(x, y, r, w, h, time_budget):
    """Iteratively expand radii using vectorized checks and local optimization."""
    n = N
    iu = np.triu_indices(n, k=1)
    start_time = time.time()
    
    # First, ensure we are in a valid state by relaxing positions if needed
    x, y = _force_relax_vectorized(x, y, r, w, h, max_iter=200, dt=0.001)
    
    for _ in range(150):
        if time.time() > time_budget:
            break
            
        improved = False
        # Try to expand each circle, starting with smallest (most constrained)
        order = np.argsort(r)
        for i in order:
            if time.time() > time_budget:
                break
                
            # Calculate max possible radius for circle i
            r_max = _max_radius_for_circle(x, y, r, w, h, i)
            
            if r_max > r[i] + 1e-5:
                # Binary search for best valid radius
                lo = r[i]
                hi = r_max
                for _ in range(25):
                    mid = (lo + hi) / 2.0
                    if _is_valid_move(x, y, r, w, h, i, mid):
                        lo = mid
                    else:
                        # Try to refine position to make mid feasible
                        x_try, y_try, feasible = _constraint_refine(x, y, r, w, h, i, mid, max_iter=50)
                        if feasible:
                            x, y = x_try, y_try
                            lo = mid
                        else:
                            hi = mid
                            
                new_r = lo
                if new_r > r[i] + 1e-6:
                    r[i] = new_r
                    improved = True
                    # After expanding one, relax positions slightly to recover any minor overlaps
                    x, y = _force_relax_vectorized(x, y, r, w, h, max_iter=10, dt=0.0005)
            
        if not improved:
            # If no single circle can expand, try uniform scaling
            for scale in [1.001, 1.0005, 1.0001]:
                r_new = r * scale
                if _is_valid(x, y, r_new, w, h):
                    r = r_new
                    improved = True
                    break
            if not improved:
                # Try to find a direction to move circles to allow scaling
                # Use force relaxation with slightly inflated radii
                r_inflated = r * 1.001
                x_new, y_new = _force_relax_vectorized(x, y, r_inflated, w, h, max_iter=100, dt=0.001)
                if _is_valid(x_new, y_new, r, w, h):
                    x, y = x_new, y_new
                    # Now try scaling again
                    for scale in [1.001, 1.0005]:
                        r_new = r * scale
                        if _is_valid(x, y, r_new, w, h):
                            r = r_new
                            improved = True
                            break
                if not improved:
                    break
                
    return x, y, r

def _polish_result(z, time_budget):
    """High-precision polish."""
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    
    # 1. Force relax to ensure validity
    x, y = _force_relax_vectorized(x, y, r, w, h, max_iter=500, dt=0.001)
    
    # 2. Expand radii with adaptive position optimization
    x, y, r = _expand_radii_vectorized(x, y, r, w, h, time_budget * 0.7)
    
    # 3. Tighter force relaxation after expansion
    x, y = _force_relax_vectorized(x, y, r, w, h, max_iter=300, dt=0.0005)
    
    # 4. Another expansion pass with tighter tolerance
    x, y, r = _expand_radii_vectorized(x, y, r, w, h, time_budget * 0.2)
    
    # 5. Final high-precision uniform scaling attempt
    for factor in [1.0001, 1.00005, 1.00001, 1.000005, 1.000001]:
        r_trial = r * factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break
    
    # 6. Final micro-relaxation
    x, y = _force_relax_vectorized(x, y, r, w, h, max_iter=100, dt=0.0001)
    
    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_value = 0.0
    best_result = None
    penalty_weight = 1000.0

    def objective(z):
        return _vectorized_objective(z, penalty_weight)

    def check_validity(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            return True, float(np.sum(r))
        return False, 0.0

    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8, 0.9, 1.1, 0.7, 1.4, 0.65, 1.25]
        
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.48
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        rng = np.random.default_rng(42)
        for _ in range(10):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.12)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 20:
            break

        # Relax initial overlaps
        x0 = z0[0::3][:N]
        y0 = z0[1::3][:N]
        r0 = z0[2::3][:N]
        w0 = z0[-1]
        h0 = 2.0 - w0
        
        x_rel, y_rel = _force_relax_vectorized(x0, y0, r0, w0, h0, max_iter=200, dt=0.005)
        
        z_rel = np.zeros(3 * N + 1)
        z_rel[0::3][:N] = x_rel
        z_rel[1::3][:N] = y_rel
        z_rel[2::3][:N] = r0
        z_rel[-1] = w0

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(400, max(100, remaining * 30)))
        
        try:
            res = minimize(objective, z_rel, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-10, 'maxls': 50})
            z_opt = res.x
            valid, val = check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
                
            # Try to polish even if SLSQP result is slightly invalid or just for better value
            # But only if we have time
            if time.time() < deadline - 30:
                # Use the relaxed positions if SLSQP failed to improve, or SLSQP result
                z_to_polish = z_opt if valid else z_rel
                polished = _polish_result(z_to_polish, deadline - (time.time() - start_time) - 10)
                if polished is not None:
                    v_valid, v_val = check_validity(polished)
                    if v_valid and v_val > best_value:
                        best_value = v_val
                        best_result = polished

        except Exception:
            continue

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final polish on the best result
    if time.time() < deadline - 5:
        final_polished = _polish_result(best_result, deadline - 3)
        if final_polished is not None:
            v_valid, v_val = check_validity(final_polished)
            if v_valid and v_val > best_value:
                best_value = v_val
                best_result = final_polished
        
        # Extra: try a few more polish iterations if time permits
        remaining = deadline - 3 - (time.time() - start_time)
        if remaining > 5:
            for _ in range(3):
                if time.time() > deadline - 3:
                    break
                extra_polished = _polish_result(best_result, min(remaining, 10))
                if extra_polished is not None:
                    v_valid, v_val = check_validity(extra_polished)
                    if v_valid and v_val > best_value:
                        best_value = v_val
                        best_result = extra_polished
                    remaining = deadline - 3 - (time.time() - start_time)
                    if remaining < 3:
                        break

    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    
    # Safety shrink
    for _ in range(5):
        r_trial = r * (1 - 1e-9)
        w = best_result[-1]
        h = 2.0 - w
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
