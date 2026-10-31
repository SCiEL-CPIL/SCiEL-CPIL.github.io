"""Circle packing in perimeter-4 rectangle: Radial Growth + SLSQP Polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21


def _hex_grid_init(W, H, n_circles=21, spacing_factor=0.4):
    """Initialize circles on a hexagonal grid pattern."""
    # Estimate spacing
    area = W * H
    avg_spacing = np.sqrt(area / n_circles)
    dx = avg_spacing * spacing_factor
    dy = avg_spacing * spacing_factor * np.sqrt(3) / 2
    
    positions = []
    row = 0
    while len(positions) < n_circles:
        for col in range(100):
            x = dx / 2 + col * dx + (row % 2) * dx / 2
            y = dy / 2 + row * dy
            if x < W - dx and y < H - dy and x > 0 and y > 0:
                positions.append([x, y])
                if len(positions) >= n_circles:
                    break
        row += 1
    
    positions = np.array(positions[:n_circles])
    # Initial radii: small
    r = np.full(n_circles, min(W, H) / 20.0)
    return positions, r


def _radial_growth(positions, radii, W, H, max_iters=2000, tol=1e-10):
    """
    Sequential radial growth: iteratively expand each circle to its maximum
    admissible radius given current neighbors and walls.
    """
    n = len(radii)
    radii = radii.copy()
    
    for it in range(max_iters):
        improved = False
        for i in range(n):
            # Compute max admissible radius for circle i
            # Boundary constraint
            max_r = min(positions[i, 0], W - positions[i, 0],
                        positions[i, 1], H - positions[i, 1])
            
            # Neighbor constraints
            for j in range(n):
                if i == j:
                    continue
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                # r_i + r_j <= d  =>  r_i <= d - r_j
                max_r = min(max_r, d - radii[j])
            
            if max_r < 0:
                max_r = 0.0
            
            # Expand if possible
            if max_r > radii[i] + tol:
                radii[i] = max_r
                improved = True
        
        if not improved:
            break
    
    return radii


def _vectorized_objective(z, penalty_weight=5000.0):
    """Vectorized objective with smooth penalties for SLSQP."""
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
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
    r_sum_pairs = r_sum[iu]
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    # Boundary penalty
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)
    obj += penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    # Aspect ratio penalty
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _check_validity(x, y, r, w, h, tol=1e-8):
    """Check if the packing is valid."""
    if w <= 0.05 or h <= 0.05:
        return False, 0.0
    
    for i in range(N):
        if r[i] <= 0:
            return False, 0.0
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False, 0.0
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False, 0.0
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum ** 2 - tol:
                return False, 0.0
    
    return True, float(np.sum(r))


def _slsqp_optimize(x, y, r, w, h, deadline, maxiter=300):
    """Run SLSQP optimization on the full state."""
    z = np.zeros(3 * N + 1)
    z[0::3][:N] = x
    z[1::3][:N] = y
    z[2::3][:N] = r
    z[-1] = w
    
    def objective(zz):
        return _vectorized_objective(zz)
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, w - 0.01))  # x
        bounds.append((0.01, h - 0.01))  # y
        bounds.append((0.005, None))     # r
    bounds.append((0.4, 1.6))  # w
    
    elapsed = time.time() - (deadline - maxiter * 0.05)
    try:
        res = minimize(objective, z, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-10, 'maxls': 60})
        return res.x
    except Exception:
        return z


def _polish_radii(x, y, r, w, h, deadline):
    """High-precision radius polish using bisection on each circle."""
    r = r.copy()
    n = len(r)
    
    # Multiple passes of increasing precision
    for pass_num in range(5):
        tol = 1e-8 / (10 ** pass_num)
        for i in range(n):
            # Find max radius for circle i given others fixed
            lo = r[i] * 0.5
            hi = min(x[i], w - x[i], y[i], h - y[i])
            
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                hi = min(hi, d - r[j])
            
            if hi <= lo:
                continue
            
            # Bisection
            for _ in range(30):
                mid = (lo + hi) / 2
                feasible = True
                for j in range(n):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if mid + r[j] > d + 1e-12:
                        feasible = False
                        break
                if feasible:
                    lo = mid
                else:
                    hi = mid
            
            r[i] = lo * 0.9999
    
    return r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_value = 0.0
    best_result = None
    
    # Aspect ratios to try
    aspects = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 1.2, 0.8]
    
    for w0 in aspects:
        if time.time() > deadline - 10:
            break
        h0 = 2.0 - w0
        if h0 < 0.3 or w0 < 0.3:
            continue
        
        # Phase 1: Hex grid init + radial growth
        pos, r = _hex_grid_init(w0, h0, N)
        r = _radial_growth(pos, r, w0, h0, max_iters=3000)
        
        x = pos[:, 0]
        y = pos[:, 1]
        
        valid, val = _check_validity(x, y, r, w0, h0)
        if valid and val > best_value:
            best_value = val
            best_result = np.column_stack([x, y, r])
        
        # Phase 2: SLSQP optimization
        if time.time() < deadline - 15:
            z_opt = _slsqp_optimize(x, y, r, w0, h0, deadline, maxiter=200)
            x2 = z_opt[0::3][:N]
            y2 = z_opt[1::3][:N]
            r2 = z_opt[2::3][:N]
            w2 = z_opt[-1]
            h2 = 2.0 - w2
            
            valid, val = _check_validity(x2, y2, r2, w2, h2)
            if not valid:
                # Repair: shrink radii
                r2 *= 0.95
                valid, val = _check_validity(x2, y2, r2, w2, h2)
            
            if valid and val > best_value:
                best_value = val
                best_result = np.column_stack([x2, y2, r2])
        
        # Phase 3: Polish radii
        if best_result is not None and time.time() < deadline - 5:
            x3 = best_result[:, 0]
            y3 = best_result[:, 1]
            r3 = best_result[:, 2]
            w3 = w0  # Use the aspect that produced best_result
            # Actually we need to track which aspect gave best_result
            # Let's just polish the current one
            r_polished = _polish_radii(x2, y2, r2, w2, h2, deadline)
            valid, val = _check_validity(x2, y2, r_polished, w2, h2)
            if valid and val > best_value:
                best_value = val
                best_result = np.column_stack([x2, y2, r_polished])
    
    # Final polish on best result
    if best_result is not None and time.time() < deadline - 2:
        x = best_result[:, 0]
        y = best_result[:, 1]
        r = best_result[:, 2]
        # Determine w, h from bounding box
        w = max(x + r)
        h = max(y + r)
        # Actually, we need to know the rectangle. Let's re-derive.
        # The rectangle is [0, W] x [0, H] where W+H=2
        # We need to find W, H such that all circles fit
        # For now, just ensure validity
        for w_try in [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 1.2, 0.8]:
            h_try = 2.0 - w_try
            if h_try < 0.3:
                continue
            valid, val = _check_validity(x, y, r, w_try, h_try)
            if valid:
                # Polish
                r_polished = _polish_radii(x, y, r, w_try, h_try, deadline)
                valid, val = _check_validity(x, y, r_polished, w_try, h_try)
                if valid and val > best_value:
                    best_value = val
                    best_result = np.column_stack([x, y, r_polished])
                break
    
    if best_result is None:
        # Fallback: simple grid
        W, H = 1.0, 1.0
        side = int(np.ceil(np.sqrt(N)))
        pos = np.array([[(i + 0.5) * W / side, (j + 0.5) * H / side]
                        for i in range(side) for j in range(side)])[:N]
        r = np.full(N, min(W, H) / side * 0.4)
        best_result = np.column_stack([pos, r])
    
    return best_result
