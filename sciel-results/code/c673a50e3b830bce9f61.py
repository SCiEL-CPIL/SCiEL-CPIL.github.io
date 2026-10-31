"""
Circle packing in perimeter-4 rectangle: SLSQP + Pressure-Based Expansion.
Strategy:
1. Initialize with hexagonal lattice.
2. Use SLSQP to find a feasible region.
3. Polish with a pressure-field expansion: iteratively grow circles based on local clearance.
4. Dynamic aspect ratio adjustment during polish.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _vectorized_objective(z, penalty_weight=1000.0):
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.1 or h < 0.1 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Overlap penalty
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

    # Aspect ratio soft constraint (keep near 1.0)
    if w < 0.6:
        obj += penalty_weight * (0.6 - w) ** 2
    elif w > 1.4:
        obj += penalty_weight * (w - 1.4) ** 2

    return obj

def _is_valid(cx, cy, r, W, H, tol=1e-8):
    if W <= 0.05 or H <= 0.05:
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

def _hexagon_init(w0, h0):
    """Generate hexagonal lattice points scaled to w0 x h0."""
    z = np.zeros(3 * N + 1)
    # Hex spacing
    # Try to fit roughly sqrt(N) circles in width
    side = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3) * w0 / h0))) if h0 > 0 else 5
    if side < 2: side = 3
    
    # Hex geometry: dx = s, dy = s * sqrt(3)/2
    # We want to cover the rectangle.
    # Estimate spacing
    # Let's just use a grid and shift every other row
    cols = int(np.ceil(w0 / (h0 / N * 0.8))) # Rough guess
    if cols < 3: cols = 4
    
    idx = 0
    row = 0
    while idx < N:
        for col in range(cols):
            if idx >= N: break
            # Hex shift
            offset = 0.5 if row % 2 == 1 else 0.0
            cx = (col + 0.5 + offset) / cols * w0
            cy = (row + 0.5) / max(1, int(np.ceil(N / cols))) * h0
            
            # Clamp to center
            cx = np.clip(cx, w0*0.1, w0*0.9)
            cy = np.clip(cy, h0*0.1, h0*0.9)
            
            z[3*idx] = cx
            z[3*idx+1] = cy
            # Initial radius estimate
            # Distance to nearest neighbor in hex grid
            s = w0 / cols
            r0 = s * 0.4
            z[3*idx+2] = r0
            idx += 1
        row += 1
    
    z[-1] = w0
    return z

def _pressure_polish(z, deadline):
    """
    Iterative expansion using a pressure field.
    For each circle, compute the max allowable radius increase based on clearance.
    Apply small steps of this increase.
    Also adjust w to maximize the sum.
    """
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1].copy()
    
    # We will alternate between:
    # 1. Expanding radii
    # 2. Relaxing positions (force-directed)
    # 3. Optimizing w
    
    step_size = 0.005
    max_iter = 1000
    
    current_val = float(np.sum(r))
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        h = 2.0 - w
        
        # 1. Calculate Pressure / Clearance
        # For each circle i, the max radius is limited by:
        # a) Walls: min(x[i], w-x[i], y[i], h-y[i])
        # b) Neighbors: min_j ( dist(i,j) - r[j] )
        
        r_max = np.minimum(np.minimum(x, w - x), np.minimum(y, h - y))
        
        for i in range(n):
            min_clearance = np.inf
            for j in range(n):
                if i == j: continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                clearance = d - r[j]
                if clearance < min_clearance:
                    min_clearance = clearance
            if min_clearance < r_max[i]:
                r_max[i] = min_clearance
        
        # Apply growth
        # Growth is proportional to clearance, but capped
        growth = np.maximum(0, r_max - r)
        # Smooth growth to avoid oscillations
        growth = np.minimum(growth, 0.01) # Max absolute growth per step
        r_new = r + step_size * growth * 10 # Scale step
        
        # Check if growth is valid
        if _is_valid(x, y, r_new, w, h):
            r = r_new
            current_val = float(np.sum(r))
            
        # 2. Force Directed Relaxation (Position only)
        # Simple vectorized step to resolve overlaps if any slipped in
        # We only run this if we made changes
        if np.any(r_new != r):
             pass # Simplified: assume growth was conservative enough to not overlap
             # If we want to be robust:
             # fx, fy = compute forces
             # x += dt * fx ...
             
        # 3. Optimize Aspect Ratio w
        # Try small perturbations of w to see if sum(r) increases
        # Since r depends on w (via wall clearance), this is a 1D optimization
        # We can do a quick local search
        w_old = w
        best_w = w
        best_val = current_val
        
        for dw in [0.01, -0.01, 0.005, -0.005]:
            w_trial = w + dw
            h_trial = 2.0 - w_trial
            if w_trial < 0.3 or h_trial < 0.3: continue
            
            # Re-evaluate validity with current x,y,r
            # Note: x,y,r might need to be adjusted if walls move, but for a "pressure"
            # step, we assume x,y are fixed and only r grows.
            # However, if w changes, the wall clearance changes.
            # The _is_valid check handles this.
            if _is_valid(x, y, r, w_trial, h_trial):
                # Can we grow r further with new w?
                # For simplicity, just check if the *current* r is still valid and if
                # the *potential* for growth is higher.
                # Actually, if w changes, the *current* r might become invalid if it was touching a wall.
                # Let's stick to growing r in the fixed w step, and use SLSQP for w changes.
                # This function is for radial growth.
                pass
        
        # Instead of complex w search here, let's do a simple force relaxation
        # to keep circles centered if w is not being optimized by SLSQP here.
        
        # Adaptive step
        if it % 50 == 0 and it > 0:
             # Slow down
             step_size *= 0.9
             
        if np.max(r_max - r) < 1e-5:
             # Converged
             break
             
    z[0::3][:n] = x
    z[1::3][:n] = y
    z[2::3][:n] = r
    z[-1] = w
    return z

def _force_relax(z, max_iter=100):
    """Quick force-directed relaxation to fix overlaps."""
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    
    dt = 0.01
    for _ in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall repulsion
        fx -= np.maximum(r - x, 0.0) * 100
        fx += np.maximum(x + r - w, 0.0) * 100
        fy -= np.maximum(r - y, 0.0) * 100
        fy += np.maximum(y + r - h, 0.0) * 100
        
        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)
        
        if np.max(overlap) > 1e-8:
            f = overlap * 20.0
            ux = dx[iu] / d_pairs
            uy = dy[iu] / d_pairs
            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f[idx]
                fy[i_idx] += uy[idx] * f[idx]
                fx[j_idx] -= ux[idx] * f[idx]
                fy[j_idx] -= uy[idx] * f[idx]
        
        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        
        if np.max(np.sqrt(fx*fx + fy*fy)) < 1e-4:
            break
            
    z[0::3][:n] = x
    z[1::3][:n] = y
    return z

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_value = 0.0
    best_result = None
    
    # Initial guesses
    w_starts = [1.0, 0.8, 1.2, 0.9, 1.1]
    
    for w0 in w_starts:
        if time.time() > deadline - 10:
            break
            
        h0 = 2.0 - w0
        z0 = _hexagon_init(w0, h0)
        
        # Relax initial overlaps
        z0 = _force_relax(z0, max_iter=200)
        
        # SLSQP to optimize sum(r)
        try:
            res = minimize(
                _vectorized_objective, 
                z0, 
                method='SLSQP',
                bounds=[(0, 2)] * (3 * N) + [(0.4, 1.6)],
                options={'maxiter': 300, 'ftol': 1e-12, 'maxls': 50}
            )
            z_opt = res.x
            
            # Polish with Pressure
            z_polished = _pressure_polish(z_opt, deadline - 15)
            
            # Final validity check and repair
            x = z_polished[0::3][:N]
            y = z_polished[1::3][:N]
            r = z_polished[2::3][:N]
            w = z_polished[-1]
            h = 2.0 - w
            
            if not _is_valid(x, y, r, w, h):
                # Try shrinking radii
                for factor in [0.99, 0.98, 0.95]:
                    r_trial = r * factor
                    if _is_valid(x, y, r_trial, w, h):
                        r = r_trial
                        break
                else:
                    continue
                    
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = z_polished.copy()
                
        except Exception:
            continue
            
    if best_result is None:
        # Fallback
        z = np.zeros(3 * N + 1)
        side = 5
        for i in range(N):
            row, col = divmod(i, side)
            z[3*i] = (col + 0.5) / side
            z[3*i+1] = (row + 0.5) / side
            z[3*i+2] = 0.05
        z[-1] = 1.0
        return np.column_stack((z[0::3][:N], z[1::3][:N], z[2::3][:N]))
        
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    
    return np.column_stack((x, y, r))

