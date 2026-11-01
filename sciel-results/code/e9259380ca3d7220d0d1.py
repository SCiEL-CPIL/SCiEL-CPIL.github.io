import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = time.time() + 140.0

def _is_valid(x, y, r, w, h, tol=1e-9):
    """Check if configuration is strictly valid."""
    if np.any(r <= 0):
        return False
    if w <= 0 or h <= 0:
        return False
    if 2*(w+h) > 4.0 + tol:
        return False
        
    # Boundary check
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False
        
    # Overlap check
    n = len(r)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2
    r_sum = r[:, None] + r[None, :]
    
    # Check all pairs i < j
    diff = dist_sq - r_sum**2
    np.fill_diagonal(diff, 1.0)
    
    if np.min(diff) < -tol:
        return False
        
    return True

def _force_relax(x, y, r, w, h, max_iter=2000, dt=0.01):
    """
    Force-directed relaxation to resolve overlaps and boundary violations.
    Moves circles apart and into the rectangle.
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 10:
            break
            
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Boundary forces
        # Left
        viol = r - x
        mask = viol > 0
        fx[mask] += viol[mask]
        # Right
        viol = x + r - w
        mask = viol > 0
        fx[mask] -= viol[mask]
        # Bottom
        viol = r - y
        mask = viol > 0
        fy[mask] += viol[mask]
        # Top
        viol = y + r - h
        mask = viol > 0
        fy[mask] -= viol[mask]
        
        # Overlap forces
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    if dist < 1e-8:
                        dx = 0.01
                        dy = 0.01
                        dist = np.sqrt(dx*dx + dy*dy)
                    # Force magnitude proportional to overlap
                    f_mag = (min_dist - dist)
                    fx[i] += (dx/dist) * f_mag
                    fy[i] += (dy/dist) * f_mag
                    fx[j] -= (dx/dist) * f_mag
                    fy[j] -= (dy/dist) * f_mag
        
        # Update positions
        x += fx * dt
        y += fy * dt
        
        # Clamp to boundaries (hard constraint for stability)
        x = np.clip(x, r, w-r)
        y = np.clip(y, r, h-r)
        
        # Check convergence
        if np.max(np.abs(fx)) < 1e-6 and np.max(np.abs(fy)) < 1e-6:
            break
            
    return x, y, r

def _build_objective(w, h):
    """
    Build objective and constraint functions for SLSQP optimization.
    Variables: [x_0..x_20, y_0..y_20, r_0..r_20] (63 variables)
    """
    def objective(v):
        x, y, r = v[:N], v[N:2*N], v[2*N:3*N]
        # Maximize sum of radii => minimize negative sum
        return -np.sum(r)
    
    def constraints(v):
        x, y, r = v[:N], v[N:2*N], v[2*N:3*N]
        cons = []
        
        # Boundary constraints: r_i <= x_i <= w - r_i
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: (w - v[i]) - v[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: (h - v[N+i]) - v[2*N+i]})
        
        # Overlap constraints: dist_ij >= r_i + r_j
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: 
                        (v[i]-v[j])**2 + (v[N+i]-v[N+j])**2 - (v[2*N+i]+v[2*N+j])**2
                })
        
        return cons
    
    return objective, constraints

def _initialize_grid(w, h, seed=0):
    """
    Initialize circles in a grid pattern with some randomness.
    Returns initial variable vector [x, y, r].
    """
    rng = np.random.default_rng(seed)
    
    # Try different grid configurations
    configs = [
        (5, 5),  # 5x5 grid
        (4, 6),  # 4x6 grid  
        (6, 4),  # 6x4 grid
        (3, 7),  # 3x7 grid
        (7, 3),  # 7x3 grid
        (4, 5),  # 4x5 grid (20 spots, add 1 more)
    ]
    
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    # Pick a random config
    rows, cols = configs[rng.integers(len(configs))]
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= N:
                break
            cx = (j + 0.5) / cols * w
            cy = (i + 0.5) / rows * h
            # Add small random perturbation
            cx += rng.uniform(-0.02*w, 0.02*w)
            cy += rng.uniform(-0.02*h, 0.02*h)
            x[idx] = np.clip(cx, 0.01, w-0.01)
            y[idx] = np.clip(cy, 0.01, h-0.01)
            # Initial radius: small fraction of cell size
            cell_w = w / cols
            cell_h = h / rows
            r[idx] = 0.3 * min(cell_w, cell_h)
            idx += 1
    
    # If we didn't fill all N, add remaining at random positions
    while idx < N:
        x[idx] = rng.uniform(0.05*w, 0.95*w)
        y[idx] = rng.uniform(0.05*h, 0.95*h)
        r[idx] = 0.05 * min(w, h)
        idx += 1
    
    # Ensure initial feasibility by shrinking radii if needed
    scale = 1.0
    for _ in range(50):
        if _is_valid(x, y, r * scale, w, h, tol=1e-6):
            break
        scale *= 0.95
    r *= scale
    
    return np.concatenate([x, y, r])

def _optimize_single(w, h, v0, max_iter=500, timeout=None):
    """
    Run SLSQP optimization from initial point v0.
    Returns optimized variable vector or None if failed.
    """
    if timeout is not None and time.time() > timeout:
        return None
    
    # First, relax any initial overlaps
    x, y, r = v0[:N], v0[N:2*N], v0[2*N:3*N]
    x, y, r = _force_relax(x, y, r, w, h, max_iter=500, dt=0.01)
    v0 = np.concatenate([x, y, r])
    
    objective, constraints = _build_objective(w, h)
    
    # Bounds for each variable
    bounds = []
    for i in range(N):
        bounds.append((0.001, w - 0.001))  # x_i
    for i in range(N):
        bounds.append((0.001, h - 0.001))  # y_i
    for i in range(N):
        bounds.append((0.001, min(w, h)/2))  # r_i
    
    try:
        result = minimize(
            objective,
            v0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-8}
        )
        
        if result.success and time.time() < (timeout or DEADLINE):
            return result.x
        else:
            # Even if not fully converged, check if result is valid
            x, y, r = result.x[:N], result.x[N:2*N], result.x[2*N:3*N]
            if _is_valid(x, y, r, w, h, tol=1e-8):
                return result.x
            return None
            
    except Exception:
        return None

def _polish_with_local_search(v, w, h, num_iters=100):
    """
    Local search polish: try small perturbations to improve the solution.
    """
    best_v = v.copy()
    best_val = -best_v[2*N:3*N].sum()
    
    for it in range(num_iters):
        if time.time() > DEADLINE - 10:
            break
            
        # Try random small perturbations
        for _ in range(10):
            v_trial = best_v.copy()
            # Perturb positions slightly
            v_trial[:2*N] += np.random.uniform(-0.01, 0.01, 2*N)
            # Perturb radii slightly
            v_trial[2*N:] *= (1 + np.random.uniform(-0.02, 0.02, N))
            
            x, y, r = v_trial[:N], v_trial[N:2*N], v_trial[2*N:3*N]
            if _is_valid(x, y, r, w, h, tol=1e-9):
                val = -r.sum()
                if val < best_val:  # We're minimizing negative sum
                    best_val = val
                    best_v = v_trial
    
    return best_v

def _expansion_relaxation_polish(v, w, h, deadline):
    """
    Polish by trying to expand radii and relaxing positions.
    """
    x, y, r = v[:N].copy(), v[N:2*N].copy(), v[2*N:3*N].copy()
    best_val = r.sum()
    best_v = v.copy()
    
    for it in range(200):
        if time.time() > deadline:
            break
            
        # Try to expand each circle slightly
        improved = False
        for i in range(N):
            old_r = r[i]
            r[i] = old_r * 1.01  # 1% expansion
            
            # Relax positions to accommodate
            x_new, y_new, r_new = _force_relax(x, y, r, w, h, max_iter=100, dt=0.005)
            
            if _is_valid(x_new, y_new, r_new, w, h, tol=1e-9):
                val = r_new.sum()
                if val > best_val:
                    best_val = val
                    best_v = np.concatenate([x_new, y_new, r_new])
                    x, y, r = x_new, y_new, r_new
                    improved = True
            else:
                r[i] = old_r  # Revert if invalid
        
        if not improved:
            break
            
    return best_v

def _vectorized_objective(z, penalty_weight=1000.0):
    """Vectorized objective with smooth penalties for speed."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Vectorized pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 0.0))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)

    obj += overlap_penalty

    # Vectorized boundary penalty
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    boundary_penalty = penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    obj += boundary_penalty

    # Keep aspect ratio in reasonable range
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj

def _check_validity(z, tol=1e-9):
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

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

    val = float(np.sum(r))
    return True, val

def _force_relax_unified(z, max_iter=500):
    """Force-directed relaxation in unified parameter space with adaptive step (vectorized)."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001
    convergence_tol = 1e-7

    for it in range(max_iter):
        if time.time() > DEADLINE - 10:
            break

        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Boundary forces
        fx += np.maximum(r - x, 0.0)
        fx -= np.maximum(x + r - w, 0.0)
        fy += np.maximum(r - y, 0.0)
        fy -= np.maximum(y + r - h, 0.0)

        # Overlap forces (vectorized)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        d = np.sqrt(np.maximum(d_sq, 1e-12))

        # Avoid division by zero
        mask = d > 1e-8
        d_safe = np.where(mask, d, 1.0)

        overlap = np.maximum(r_sum - d, 0.0)
        # Only consider upper triangle to avoid double counting
        iu = np.triu_indices(n, k=1)
        overlap_iu = overlap[iu]
        d_iu = d_safe[iu]
        dx_iu = dx[iu]
        dy_iu = dy[iu]

        for i, j in zip(iu[0], iu[1]):
            if overlap[i, j] > 0:
                f_mag = overlap[i, j] / d_safe[i, j]
                fx[i] += dx[i, j] * f_mag
                fy[i] += dy[i, j] * f_mag
                fx[j] -= dx[i, j] * f_mag
                fy[j] -= dy[i, j] * f_mag

        # Update positions
        z[0::3][:n] += fx * dt
        z[1::3][:n] += fy * dt

        # Clamp to boundaries
        z[0::3][:n] = np.clip(z[0::3][:n], r, w - r)
        z[1::3][:n] = np.clip(z[1::3][:n], r, h - r)

        # Check convergence
        if np.max(np.abs(fx)) < convergence_tol and np.max(np.abs(fy)) < convergence_tol:
            break

        # Adaptive step
        if it % 50 == 0:
            dt = max(min_dt, dt * 0.9)

    return z

def construct():
    start_time = time.time()
    best_value = 0.0
    best_circles = None
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.2, 0.7, 1.3, 0.9, 1.1, 0.6, 1.4, 0.85, 1.15, 0.75, 1.25]
    
    rng = np.random.default_rng(42)
    rng.shuffle(aspect_ratios)
    
    # Also try some random aspect ratios
    for _ in range(8):
        aspect_ratios.append(rng.uniform(0.5, 1.5))
        
    for w in aspect_ratios:
        if time.time() > DEADLINE - 20:
            break
            
        h = 2.0 - w
        if h < 0.3 or w < 0.3:
            continue
            
        # Multiple random restarts for this aspect ratio
        for restart in range(3):
            if time.time() > DEADLINE - 15:
                break
                
            # Initialize
            v0 = _initialize_grid(w, h, seed=restart * 100 + int(w * 100))
            
            # Optimize
            v_opt = _optimize_single(w, h, v0, max_iter=300, timeout=DEADLINE - 15)
            
            if v_opt is not None:
                x, y, r = v_opt[:N], v_opt[N:2*N], v_opt[2*N:3*N]
                val = r.sum()
                
                # Polish with local search
                if time.time() < DEADLINE - 10:
                    v_opt = _polish_with_local_search(v_opt, w, h, num_iters=50)
                    x, y, r = v_opt[:N], v_opt[N:2*N], v_opt[2*N:3*N]
                    val = r.sum()
                    
                    # Expansion polish
                    if time.time() < DEADLINE - 8:
                        v_exp = _expansion_relaxation_polish(v_opt, w, h, deadline=DEADLINE - 5)
                        x_exp, y_exp, r_exp = v_exp[:N], v_exp[N:2*N], v_exp[2*N:3*N]
                        val_exp = r_exp.sum()
                        if val_exp > val:
                            v_opt = v_exp
                            x, y, r = x_exp, y_exp, r_exp
                            val = val_exp
                
                if val > best_value:
                    best_value = val
                    best_circles = np.column_stack((x, y, r))
                    
    if best_circles is None:
        # Fallback: Simple grid in square
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (j + 0.5) / side
                y = (i + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append([x, y, r])
        best_circles = np.array(cells[:N])
        
    return best_circles
