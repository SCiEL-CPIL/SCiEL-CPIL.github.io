<<<<<<< SEARCH
def _force_relax_unified(z, max_iter=500):
    """Force-directed relaxation in unified parameter space with adaptive step (vectorized)."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001
    convergence_tol = 1e-8

    for it in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        if w < 0.1 or h < 0.1:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion (vectorized)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100

        # Pairwise repulsion (vectorized)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > convergence_tol:
            f_pairs = overlap * 15.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]

        # Check convergence
        if max_overlap < convergence_tol:
            break

        # Adaptive step size
        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.1:
            dt = min(0.01, dt * 1.1)

        # Update positions
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        # Clamp positions
        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z
=======
def _force_relax_unified(z, max_iter=500):
    """Force-directed relaxation in unified parameter space with adaptive step (vectorized) and corner bias."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001
    convergence_tol = 1e-8

    for it in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        if w < 0.1 or h < 0.1:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion (vectorized)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100

        # Corner Bias: Attract circles towards the nearest corner
        # This simulates the "external field" biasing circles to corners
        # We calculate distance to each corner and apply a force towards the closest one
        # Corners: (0,0), (w,0), (0,h), (w,h)
        # For each circle, find the closest corner and pull towards it
        
        # Vectorized distance to corners
        # d_c1 = sqrt(x^2 + y^2)
        # d_c2 = sqrt((w-x)^2 + y^2)
        # d_c3 = sqrt(x^2 + (h-y)^2)
        # d_c4 = sqrt((w-x)^2 + (h-y)^2)
        
        # Instead of full sqrt, we can just use the components to determine direction
        # If x < w/2 and y < h/2, pull to (0,0)
        # If x > w/2 and y < h/2, pull to (w,0)
        # If x < w/2 and y > h/2, pull to (0,h)
        # If x > w/2 and y > h/2, pull to (w,h)
        
        # This simple quadrant approach is fast and effective for biasing
        mid_w = w / 2.0
        mid_h = h / 2.0
        
        # Pull strength increases as circle moves away from center towards corner? 
        # Or constant? Constant is simpler. Let's use a moderate constant pull.
        pull_strength = 0.5
        
        # Determine target corner for each circle
        # target_x: 0 or w
        # target_y: 0 or h
        
        target_x = np.where(x < mid_w, 0.0, w)
        target_y = np.where(y < mid_h, 0.0, h)
        
        # Force towards target
        fx += pull_strength * (target_x - x) / max(1e-6, np.abs(target_x - x)) # Normalize direction? 
        # Actually, just pulling with a fixed force in the direction of the corner is better
        # Let's just add a force component towards the corner
        # F_x = k * sign(target_x - x)
        # F_y = k * sign(target_y - y)
        
        fx += pull_strength * np.sign(target_x - x)
        fy += pull_strength * np.sign(target_y - y)

        # Pairwise repulsion (vectorized)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > convergence_tol:
            f_pairs = overlap * 15.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]

        # Check convergence
        if max_overlap < convergence_tol:
            break

        # Adaptive step size
        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.1:
            dt = min(0.01, dt * 1.1)

        # Update positions
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        # Clamp positions
        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z

def _corner_bias_polish(z, time_budget):
    """
    Polish by aggressively pushing circles towards corners and edges.
    This helps in finding packings where circles are tightly packed against the boundaries.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    max_iter = int(min(2000, time_budget * 100))
    dt = 0.002
    
    for it in range(max_iter):
        if time.time() > time_budget:
            break
            
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Strong corner pull
        mid_w = w / 2.0
        mid_h = h / 2.0
        target_x = np.where(x < mid_w, 0.0, w)
        target_y = np.where(y < mid_h, 0.0, h)
        
        # Pull towards corner
        fx += 2.0 * np.sign(target_x - x)
        fy += 2.0 * np.sign(target_y - y)
        
        # Wall repulsion to prevent crossing
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100
        
        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        
        if np.max(overlap) > 1e-8:
            f_pairs = overlap * 10.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs
            
            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]
                
        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        
    # After positioning, try to expand
    r_grown = r.copy()
    for _ in range(50):
        if time.time() > time_budget: break
        r_trial = r_grown * 1.001
        if _is_valid(x, y, r_trial, w, h):
            r_grown = r_trial
        else:
            break
            
    if _is_valid(x, y, r_grown, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r_grown
        z_new[-1] = w
        return z_new
    return None
>>>>>>> REPLACE

Now, I will integrate `_corner_bias_polish` into the `construct` function, specifically after the initial force relaxation and before the SLSQP optimization, or as an additional polish phase. I'll add it as a polish phase after the main SLSQP loop to see if it improves the best result found.

