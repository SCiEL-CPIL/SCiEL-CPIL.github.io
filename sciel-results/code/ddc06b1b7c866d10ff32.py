"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.002
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def _random_init(n, seed=0):
    """Random initialization with small radii."""
    rng = np.random.RandomState(seed)
    xy = rng.uniform(0.1, 0.9, size=(n, 2))
    r = rng.uniform(0.02, 0.05, size=n)
    return xy, r


def _langevin_jamming(xy, r, deadline, n_steps=50000, seed=42):
    """
    Langevin dynamics for hard-disk packing.
    
    Particles are hard disks that cannot overlap. We use a velocity-Verlet-like
    scheme with:
    - Hard-disk collision resolution (position projection)
    - Wall collision resolution
    - Random thermal kicks (Langevin noise)
    - Adaptive cooling: temperature decreases over time
    - Periodic radius growth: after each "settling" phase, try to grow all radii
    
    Returns (xy, r, total, valid).
    """
    rng = np.random.RandomState(seed)
    n = len(xy)
    xy = xy.copy()
    r = r.copy()
    
    # Initial temperature
    T = 0.01
    T_min = 1e-6
    cooling_rate = 0.99995  # multiplicative per step
    
    best_total = 0.0
    best_valid = False
    best_xy = xy.copy()
    best_r = r.copy()
    
    # Check initial validity
    total, valid = _eval_packing(xy, r)
    if valid:
        best_total = total
        best_valid = True
    
    # Velocity array
    v = np.zeros((n, 2))
    
    # Friction coefficient
    gamma = 0.5
    
    # Adaptive radius growth
    growth_attempts = 0
    max_growth_attempts = 50
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # --- Apply random thermal kicks ---
        kicks = rng.normal(0, np.sqrt(T), size=(n, 2))
        v += kicks
        
        # --- Apply friction (Langevin damping) ---
        v *= (1.0 - gamma)
        
        # --- Tentative move ---
        new_xy = xy + v
        
        # --- Resolve wall collisions ---
        for i in range(n):
            ri = r[i]
            # Clamp position to be inside walls
            new_xy[i, 0] = np.clip(new_xy[i, 0], ri, 1.0 - ri)
            new_xy[i, 1] = np.clip(new_xy[i, 1], ri, 1.0 - ri)
            # Reflect velocity if hitting wall
            if new_xy[i, 0] <= ri or new_xy[i, 0] >= 1.0 - ri:
                v[i, 0] = -v[i, 0] * 0.5
            if new_xy[i, 1] <= ri or new_xy[i, 1] >= 1.0 - ri:
                v[i, 1] = -v[i, 1] * 0.5
        
        # --- Resolve pairwise hard-disk collisions (iterative) ---
        # Use multiple iterations for stability
        for iter_pass in range(3):
            for i in range(n):
                for j in range(i + 1, n):
                    dx = new_xy[i, 0] - new_xy[j, 0]
                    dy = new_xy[i, 1] - new_xy[j, 1]
                    dist_sq = dx * dx + dy * dy
                    min_dist = r[i] + r[j]
                    if dist_sq < min_dist * min_dist:
                        dist = np.sqrt(dist_sq) if dist_sq > 1e-20 else 1e-10
                        # Push apart along the collision normal
                        overlap = min_dist - dist
                        if overlap > 0:
                            nx = dx / dist
                            ny = dy / dist
                            # Equal mass: split the correction
                            new_xy[i, 0] += 0.5 * overlap * nx
                            new_xy[i, 1] += 0.5 * overlap * ny
                            new_xy[j, 0] -= 0.5 * overlap * nx
                            new_xy[j, 1] -= 0.5 * overlap * ny
                            # Reflect relative velocity
                            vrel_x = v[i, 0] - v[j, 0]
                            vrel_y = v[i, 1] - v[j, 1]
                            vdotn = vrel_x * nx + vrel_y * ny
                            if vdotn < 0:
                                v[i, 0] -= vdotn * nx
                                v[i, 1] -= vdotn * ny
                                v[j, 0] += vdotn * nx
                                v[j, 1] += vdotn * ny
        
        # --- Re-clamp to walls after collision resolution ---
        for i in range(n):
            ri = r[i]
            new_xy[i, 0] = np.clip(new_xy[i, 0], ri, 1.0 - ri)
            new_xy[i, 1] = np.clip(new_xy[i, 1], ri, 1.0 - ri)
        
        # --- Accept the move (always for hard disks) ---
        xy = new_xy
        
        # --- Cool down ---
        T *= cooling_rate
        if T < T_min:
            T = T_min
        
        # --- Periodic radius growth attempt ---
        # Every ~500 steps, try to grow all radii slightly
        if step % 500 == 499 and growth_attempts < max_growth_attempts:
            # Check if current state is valid
            total_now, valid_now = _eval_packing(xy, r)
            if valid_now and total_now > best_total:
                best_total = total_now
                best_valid = True
                best_xy = xy.copy()
                best_r = r.copy()
            
            # Try to grow radii by a small amount
            growth_factor = 1.0 + 0.002
            new_r = r * growth_factor
            # Check if growth is feasible (quick check)
            feasible = True
            for i in range(n):
                if new_r[i] > min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1]):
                    feasible = False
                    break
            if feasible:
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = xy[i, 0] - xy[j, 0]
                        dy = xy[i, 1] - xy[j, 1]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist < new_r[i] + new_r[j] - 1e-9:
                            feasible = False
                            break
                    if not feasible:
                        break
            
            if feasible:
                r = new_r
                growth_attempts += 1
                # Reheat slightly after growth to allow rearrangement
                T = max(T, 0.005)
            else:
                # Can't grow; increase cooling to settle faster
                cooling_rate = max(0.9999, cooling_rate * 0.999)
        
        # --- Track best ---
        if step % 1000 == 0:
            total_now, valid_now = _eval_packing(xy, r)
            if valid_now and total_now > best_total:
                best_total = total_now
                best_valid = True
                best_xy = xy.copy()
                best_r = r.copy()
    
    # Final evaluation
    total_now, valid_now = _eval_packing(xy, r)
    if valid_now and total_now > best_total:
        best_total = total_now
        best_valid = True
        best_xy = xy.copy()
        best_r = r.copy()
    
    return best_xy, best_r, best_total, best_valid


def _adaptive_cooling_langevin(xy, r, deadline, n_steps=80000, seed=0):
    """
    Enhanced Langevin with adaptive cooling and periodic reheating.
    
    Uses a more sophisticated schedule:
    - Start with moderate temperature
    - Cool geometrically
    - When no improvement for a while, reheat briefly
    - Gradually increase radius
    """
    rng = np.random.RandomState(seed)
    n = len(xy)
    xy = xy.copy()
    r = r.copy()
    
    T = 0.008
    T_min = 1e-7
    gamma = 0.4  # friction
    
    best_total = 0.0
    best_valid = False
    best_xy = xy.copy()
    best_r = r.copy()
    
    total, valid = _eval_packing(xy, r)
    if valid:
        best_total = total
        best_valid = True
    
    v = np.zeros((n, 2))
    
    last_improvement_step = 0
    no_improve_count = 0
    max_no_improve = 5000
    
    growth_interval = 300
    growth_attempts = 0
    max_growth_attempts = 80
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # --- Langevin step ---
        kicks = rng.normal(0, np.sqrt(T), size=(n, 2))
        v = v * (1.0 - gamma) + kicks
        
        new_xy = xy + v
        
        # Wall handling
        for i in range(n):
            ri = r[i]
            new_xy[i, 0] = np.clip(new_xy[i, 0], ri, 1.0 - ri)
            new_xy[i, 1] = np.clip(new_xy[i, 1], ri, 1.0 - ri)
            if new_xy[i, 0] <= ri + 1e-12 or new_xy[i, 0] >= 1.0 - ri - 1e-12:
                v[i, 0] *= -0.3
            if new_xy[i, 1] <= ri + 1e-12 or new_xy[i, 1] >= 1.0 - ri - 1e-12:
                v[i, 1] *= -0.3
        
        # Pairwise collision resolution
        for _ in range(4):
            for i in range(n):
                for j in range(i + 1, n):
                    dx = new_xy[i, 0] - new_xy[j, 0]
                    dy = new_xy[i, 1] - new_xy[j, 1]
                    dist_sq = dx * dx + dy * dy
                    min_d = r[i] + r[j]
                    if dist_sq < min_d * min_d:
                        dist = np.sqrt(dist_sq) if dist_sq > 1e-20 else 1e-10
                        overlap = min_d - dist
                        if overlap > 0:
                            nx = dx / dist
                            ny = dy / dist
                            new_xy[i, 0] += 0.5 * overlap * nx
                            new_xy[i, 1] += 0.5 * overlap * ny
                            new_xy[j, 0] -= 0.5 * overlap * nx
                            new_xy[j, 1] -= 0.5 * overlap * ny
                            vrel = (v[i, 0] - v[j, 0]) * nx + (v[i, 1] - v[j, 1]) * ny
                            if vrel < 0:
                                v[i, 0] -= vrel * nx
                                v[i, 1] -= vrel * ny
                                v[j, 0] += vrel * nx
                                v[j, 1] += vrel * ny
        
        # Re-clamp
        for i in range(n):
            ri = r[i]
            new_xy[i, 0] = np.clip(new_xy[i, 0], ri, 1.0 - ri)
            new_xy[i, 1] = np.clip(new_xy[i, 1], ri, 1.0 - ri)
        
        xy = new_xy
        
        # --- Cooling ---
        T *= 0.99997
        if T < T_min:
            T = T_min
        
        # --- Adaptive reheat ---
        if step > last_improvement_step + max_no_improve:
            # Reheat
            T = max(T, 0.003)
            last_improvement_step = step
        
        # --- Radius growth ---
        if step % growth_interval == growth_interval - 1 and growth_attempts < max_growth_attempts:
            total_now, valid_now = _eval_packing(xy, r)
            if valid_now:
                if total_now > best_total:
                    best_total = total_now
                    best_valid = True
                    best_xy = xy.copy()
                    best_r = r.copy()
                    last_improvement_step = step
                
                # Attempt growth
                delta_r = 0.001 + 0.003 * (1.0 - growth_attempts / max_growth_attempts)
                new_r = r + delta_r
                feasible = True
                for i in range(n):
                    if new_r[i] > min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1]):
                        feasible = False
                        break
                if feasible:
                    for i in range(n):
                        for j in range(i + 1, n):
                            dx = xy[i, 0] - xy[j, 0]
                            dy = xy[i, 1] - xy[j, 1]
                            if dx * dx + dy * dy < (new_r[i] + new_r[j]) ** 2:
                                feasible = False
                                break
                        if not feasible:
                            break
                
                if feasible:
                    r = new_r
                    growth_attempts += 1
                    T = max(T, 0.004)  # reheat after growth
                else:
                    # Reduce growth increment
                    growth_interval = max(100, growth_interval + 50)
        
        # --- Periodic check ---
        if step % 2000 == 0:
            total_now, valid_now = _eval_packing(xy, r)
            if valid_now and total_now > best_total:
                best_total = total_now
                best_valid = True
                best_xy = xy.copy()
                best_r = r.copy()
                last_improvement_step = step
    
    # Final
    total_now, valid_now = _eval_packing(xy, r)
    if valid_now and total_now > best_total:
        best_total = total_now
        best_valid = True
        best_xy = xy.copy()
        best_r = r.copy()
    
    return best_xy, best_r, best_total, best_valid


def _best_fit_local_search(xy, r, deadline, grid_size=80):
    """
    Iteratively update each circle to the best position (maximizing its radius)
    given all other circles fixed. Vectorized over a grid of candidate positions.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    max_sweeps = 25
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()

            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]

            old_r = r[i]
            if new_r > old_r + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True

            # Fine local search
            if new_r > old_r + 1e-10:
                step = 1.0 / grid_size
                fine_gx = np.linspace(max(0, new_x - 3 * step), min(1, new_x + 3 * step), 20)
                fine_gy = np.linspace(max(0, new_y - 3 * step), min(1, new_y + 3 * step), 20)
                FGX, FGY = np.meshgrid(fine_gx, fine_gy, indexing='ij')
                fine_candidates = np.column_stack([FGX.ravel(), FGY.ravel()])
                FM = len(fine_candidates)

                if FM > 0:
                    fine_wall_clear = np.minimum(np.minimum(fine_candidates[:, 0], 1.0 - fine_candidates[:, 0]),
                                                 np.minimum(fine_candidates[:, 1], 1.0 - fine_candidates[:, 1]))
                    if len(others_xy) > 0:
                        fdx = fine_candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                        fdy = fine_candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                        fdists = np.sqrt(fdx * fdx + fdy * fdy)
                        fcircle_clear = np.min(fdists - others_r[np.newaxis, :], axis=1)
                        fr_candidates = np.minimum(fine_wall_clear, fcircle_clear)
                    else:
                        fr_candidates = fine_wall_clear.copy()

                    fidx = int(np.argmax(fr_candidates))
                    fnew_x, fnew_y = fine_candidates[fidx, 0], fine_candidates[fidx, 1]
                    fnew_r = fr_candidates[fidx]

                    if fnew_r > new_r + 1e-10:
                        xy[i, 0] = fnew_x
                        xy[i, 1] = fnew_y
                        r[i] = fnew_r

        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid


def _smooth_objective(v, n):
    """
    Smooth objective: minimize -sum(r) + penalty for violations.
    """
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]

    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    beta = 50.0
    wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol) / beta)

    total_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            viol = r[i] + r[j] - dist
            total_viol += np.logaddexp(0.0, beta * viol) / beta

    wall_pen_sum = np.sum(np.logaddexp(0.0, beta * wall_viol) / beta)

    return -np.sum(r) + 100.0 * (wall_pen_sum + total_viol)


def _smooth_grad(v, n):
    """
    Gradient of the smooth objective.
    """
    x = v[0:n].copy()
    y = v[n:2*n].copy()
    r = v[2*n:3*n].copy()

    grad = np.zeros(3 * n)
    beta = 50.0

    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    sig_wall = 1.0 / (1.0 + np.exp(-beta * wall_viol))

    for i in range(n):
        s = sig_wall[i]
        if s < 1e-10:
            continue
        dists = np.array([x[i], 1.0 - x[i], y[i], 1.0 - y[i]])
        min_idx = int(np.argmin(dists))
        if min_idx == 0:
            grad[2*n + i] += 100.0 * s
            grad[i] -= 100.0 * s
        elif min_idx == 1:
            grad[2*n + i] += 100.0 * s
            grad[i] += 100.0 * s
        elif min_idx == 2:
            grad[2*n + i] += 100.0 * s
            grad[n + i] -= 100.0 * s
        else:
            grad[2*n + i] += 100.0 * s
            grad[n + i] += 100.0 * s

    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            if dist_sq < 1e-20:
                continue
            dist = np.sqrt(dist_sq)
            viol = r[i] + r[j] - dist
            s = 1.0 / (1.0 + np.exp(-beta * viol))
            if s < 1e-10:
                continue
            factor = 100.0 * s
            grad[i] += factor * dx / dist
            grad[n + i] += factor * dy / dist
            grad[j] -= factor * dx / dist
            grad[n + j] -= factor * dy / dist
            grad[2*n + i] += factor
            grad[2*n + j] += factor

    grad[2*n:3*n] -= 1.0

    return grad


def _gradient_polish(xy, r, deadline, tol=1e-8, max_iter=500):
    """
    Gradient-based polish using L-BFGS-B on the smooth objective.
    """
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])

    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.5))

    def obj(v):
        if time.time() > deadline:
            return float('inf')
        return _smooth_objective(v, n)

    def grad(v):
        if time.time() > deadline:
            return np.zeros(3 * n)
        return _smooth_grad(v, n)

    try:
        result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6,
                                   'maxfun': 10000})
    except Exception:
        return xy, r

    v_opt = result.x
    new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
    new_r = v_opt[2*n:3*n]

    total, valid = _eval_packing(new_xy, new_r)
    if valid:
        return new_xy, new_r

    return xy, r


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Grid initialization + Langevin jamming (multiple seeds)
    for side in [5, 6]:
        if time.time() > deadline:
            break
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()

        out_xy, out_r, total, valid = _adaptive_cooling_langevin(xy, r, deadline, n_steps=60000, seed=side * 100)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 2: Random initializations + Langevin
    for seed in range(6):
        if time.time() > deadline:
            break
        xy, r = _random_init(N, seed=seed + 10)
        out_xy, out_r, total, valid = _adaptive_cooling_langevin(xy, r, deadline, n_steps=50000, seed=seed + 100)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 3: Local search polish from best Langevin result
    if best_valid and best_xy is not None:
        if time.time() < deadline - 25:
            ls_xy, ls_r, ls_total, ls_valid = _best_fit_local_search(
                best_xy.copy(), best_r.copy(), deadline, grid_size=90)
            if ls_valid and ls_total > best_total + 1e-10:
                best_total = ls_total
                best_xy = ls_xy
                best_r = ls_r

    # Phase 4: Gradient polish
    if best_valid and best_xy is not None and time.time() < deadline - 15:
        polished_xy, polished_r = _gradient_polish(best_xy, best_r, deadline)
        total, valid = _eval_packing(polished_xy, polished_r)
        if valid and total > best_total:
            best_total = total
            best_xy = polished_xy
            best_r = polished_r

    # Phase 5: Final local search pass
    if best_valid and best_xy is not None and time.time() < deadline - 5:
        ls2_xy, ls2_r, ls2_total, ls2_valid = _best_fit_local_search(
            best_xy.copy(), best_r.copy(), deadline, grid_size=100)
        if ls2_valid and ls2_total > best_total + 1e-10:
            best_total = ls2_total
            best_xy = ls2_xy
            best_r = ls2_r

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        total, valid = _eval_packing(xy, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
        else:
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
