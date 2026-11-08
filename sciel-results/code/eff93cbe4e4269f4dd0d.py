"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _check_validity(pos, radii, tol=1e-7):
    """Vectorized validity check."""
    n = len(pos)
    if np.any(pos[:, 0] < radii - tol) or np.any(pos[:, 0] > 1 - radii + tol):
        return False
    if np.any(pos[:, 1] < radii - tol) or np.any(pos[:, 1] > 1 - radii + tol):
        return False
    if np.any(radii < -tol):
        return False
    # Vectorized overlap check
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = pos[i_idx, 0] - pos[j_idx, 0]
    dy = pos[i_idx, 1] - pos[j_idx, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = radii[i_idx] + radii[j_idx]
    if np.any(dist < min_dist - tol):
        return False
    return True


def _make_pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _clamp_boundaries(pos, radii):
    for i in range(len(pos)):
        pos[i, 0] = max(radii[i], min(1 - radii[i], pos[i, 0]))
        pos[i, 1] = max(radii[i], min(1 - radii[i], pos[i, 1]))


def _relax(pos, radii, pairs, max_iters=2000, time_budget=30.0, start_time=None):
    """Iterative overlap resolution with boundary clamping."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        _clamp_boundaries(pos, radii)
        
        changed = False
        for i, j in pairs:
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            
            if dist < min_dist:
                changed = True
                if dist < 1e-12:
                    angle = np.random.uniform(0, 2 * np.pi)
                    push = (min_dist - dist) / 2
                    pos[i, 0] -= push * np.cos(angle)
                    pos[i, 1] -= push * np.sin(angle)
                    pos[j, 0] += push * np.cos(angle)
                    pos[j, 1] += push * np.sin(angle)
                else:
                    push = (min_dist - dist) / 2
                    nx = dx / dist
                    ny = dy / dist
                    pos[i, 0] -= push * nx
                    pos[i, 1] -= push * ny
                    pos[j, 0] += push * nx
                    pos[j, 1] += push * ny
        
        _clamp_boundaries(pos, radii)
        
        if not changed:
            break
    
    return pos, radii


def _objective_and_grad(z, penalty=1000.0):
    """
    Objective: -sum(r) + penalty * sum(violations^2)
    z: flat array [x0, y0, r0, x1, y1, r1, ...]
    Returns (value, gradient)
    """
    n = N
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    
    # Base objective: minimize negative sum of radii
    obj = -np.sum(r)
    
    # Penalty terms
    pen = 0.0
    
    # Boundary constraints
    v_x_low = np.maximum(0.0, r - x)
    v_x_high = np.maximum(0.0, x + r - 1.0)
    v_y_low = np.maximum(0.0, r - y)
    v_y_high = np.maximum(0.0, y + r - 1.0)
    
    pen += np.sum(v_x_low**2) + np.sum(v_x_high**2) + np.sum(v_y_low**2) + np.sum(v_y_high**2)
    
    # Circle-circle constraints
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    d = np.sqrt(dx*dx + dy*dy + 1e-12)
    viol = r[i_idx] + r[j_idx] - d
    active = viol > 0
    if np.any(active):
        pen += np.sum(viol[active] ** 2)
    
    total_val = obj + penalty * pen
    
    # Gradient
    grad = np.zeros(3 * n)
    
    # Gradient of -sum(r)
    grad[2::3] = -1.0
    
    # Gradient of penalty terms
    # Boundary x low: max(0, r-x)^2 -> d/dx = 2(r-x)*(-1), d/dr = 2(r-x)
    mask_x_low = (r - x) > 0
    if np.any(mask_x_low):
        grad[0::3][mask_x_low] += penalty * 2 * (r - x)[mask_x_low] * (-1)
        grad[2::3][mask_x_low] += penalty * 2 * (r - x)[mask_x_low]
    
    # Boundary x high: max(0, x+r-1)^2 -> d/dx = 2(x+r-1), d/dr = 2(x+r-1)
    mask_x_high = (x + r - 1.0) > 0
    if np.any(mask_x_high):
        grad[0::3][mask_x_high] += penalty * 2 * (x + r - 1.0)[mask_x_high]
        grad[2::3][mask_x_high] += penalty * 2 * (x + r - 1.0)[mask_x_high]
    
    # Boundary y low: max(0, r-y)^2
    mask_y_low = (r - y) > 0
    if np.any(mask_y_low):
        grad[1::3][mask_y_low] += penalty * 2 * (r - y)[mask_y_low] * (-1)
        grad[2::3][mask_y_low] += penalty * 2 * (r - y)[mask_y_low]
    
    # Boundary y high: max(0, y+r-1)^2
    mask_y_high = (y + r - 1.0) > 0
    if np.any(mask_y_high):
        grad[1::3][mask_y_high] += penalty * 2 * (y + r - 1.0)[mask_y_high]
        grad[2::3][mask_y_high] += penalty * 2 * (y + r - 1.0)[mask_y_high]
    
    # Circle-circle gradients
    if np.any(active):
        # factor = 2 * penalty * viol
        factor = 2.0 * penalty * viol[active]
        # dx/dx_i = 1, dx/dx_j = -1
        # d(dist)/dx_i = dx/d, d(dist)/dx_j = -dx/d
        # d(viol)/dx_i = -dx/d, d(viol)/dx_j = dx/d
        # d(viol)/dr_i = 1, d(viol)/dr_j = 1
        
        # Indices in the active subset
        ii = i_idx[active]
        jj = j_idx[active]
        
        d_act = d[active]
        dx_act = dx[active]
        dy_act = dy[active]
        
        gx = dx_act / d_act
        gy = dy_act / d_act
        
        # grad[i] += factor * (-gx)
        # grad[j] += factor * (gx)
        # grad[N+i] += factor * (-gy)
        # grad[N+j] += factor * (gy)
        # grad[2N+i] += factor
        # grad[2N+j] += factor
        
        # Using np.add.at for accumulation
        np.add.at(grad, ii, factor * (-gx))
        np.add.at(grad, jj, factor * (gx))
        np.add.at(grad, ii + n, factor * (-gy))
        np.add.at(grad, jj + n, factor * (gy))
        np.add.at(grad, ii + 2*n, factor)
        np.add.at(grad, jj + 2*n, factor)
        
    return total_val, grad


def _scipy_optimize(pos0, radii0, time_budget=10.0, start_time=None, penalty=1000.0):
    """
    Use scipy L-BFGS-B to minimize the penalized objective.
    """
    if start_time is None:
        start_time = time.time()
    n = N
    
    z0 = np.zeros(3 * n)
    for i in range(n):
        z0[3*i] = pos0[i, 0]
        z0[3*i+1] = pos0[i, 1]
        z0[3*i+2] = radii[i]
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.001, 0.49))  # r
    
    def wrapper(z):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(z)
        return _objective_and_grad(z, penalty=penalty)
    
    try:
        result = minimize(wrapper, z0, method='L-BFGS-B',
                          jac=True, bounds=bounds,
                          options={'maxiter': 200, 'maxfun': 500, 'gtol': 1e-8,
                                   'maxls': 30})
    except Exception:
        return pos0.copy(), radii0.copy()
    
    if time.time() - start_time > time_budget:
        return pos0.copy(), radii0.copy()
    
    pos_new = np.zeros((n, 2))
    radii_new = np.zeros(n)
    for i in range(n):
        pos_new[i, 0] = result.x[3*i]
        pos_new[i, 1] = result.x[3*i+1]
        radii_new[i] = result.x[3*i+2]
    
    # Clamp
    _clamp_boundaries(pos_new, radii_new)
    
    return pos_new, radii_new


def _local_search(pos, radii, pairs, time_budget=20.0, start_time=None):
    """Local search with random moves and growth."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    rng = np.random.RandomState(int(time.time() * 1000) % (2**31))
    
    while time.time() - start_time < time_budget:
        # Try to grow radii slightly
        grow = 0.0005
        new_radii = radii + grow
        feasible = True
        # Quick check
        for i in range(n):
            if new_radii[i] > 0.5:
                feasible = False
                break
            if pos[i, 0] < new_radii[i] - 1e-9 or pos[i, 0] > 1 - new_radii[i] + 1e-9:
                feasible = False
                break
            if pos[i, 1] < new_radii[i] - 1e-9 or pos[i, 1] > 1 - new_radii[i] + 1e-9:
                feasible = False
                break
        if feasible:
            for i, j in pairs:
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < new_radii[i] + new_radii[j] - 1e-9:
                    feasible = False
                    break
        
        if feasible:
            radii = new_radii
            pos, radii = _relax(pos, radii, pairs, max_iters=100,
                                  time_budget=min(0.5, time_budget - (time.time() - start_time)),
                                  start_time=start_time)
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
            continue
        
        for attempt in range(50):
            if time.time() - start_time > time_budget:
                break
            
            idx = rng.randint(n)
            old_pos = pos[idx].copy()
            step_size = 0.01
            pos[idx] += rng.normal(0, step_size, size=2)
            pos[idx] = np.clip(pos[idx], radii[idx], 1 - radii[idx])
            
            valid = True
            if pos[idx, 0] < radii[idx] - 1e-9 or pos[idx, 0] > 1 - radii[idx] + 1e-9:
                valid = False
            if pos[idx, 1] < radii[idx] - 1e-9 or pos[idx, 1] > 1 - radii[idx] + 1e-9:
                valid = False
            if valid:
                for j in range(n):
                    if j == idx:
                        continue
                    dx = pos[j, 0] - pos[idx, 0]
                    dy = pos[j, 1] - pos[idx, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < radii[idx] + radii[j] - 1e-9:
                        valid = False
                        break
            
            if valid:
                # Try to grow slightly
                grow = 0.0002
                new_r = radii.copy()
                new_r[idx] += grow
                # Check validity for this single change roughly
                if new_r[idx] <= 0.5 and pos[idx, 0] >= new_r[idx] and pos[idx, 0] <= 1-new_r[idx] and pos[idx, 1] >= new_r[idx] and pos[idx, 1] <= 1-new_r[idx]:
                    ok = True
                    for j in range(n):
                        if j == idx: continue
                        dx = pos[j, 0] - pos[idx, 0]
                        dy = pos[j, 1] - pos[idx, 1]
                        dist = np.sqrt(dx*dx+dy*dy)
                        if dist < new_r[idx] + radii[j] - 1e-9:
                            ok = False
                            break
                    if ok:
                        radii = new_r
                        score = np.sum(radii)
                        if score > best_score:
                            best_score = score
                            best_pos = pos.copy()
                            best_radii = radii.copy()
            else:
                pos[idx] = old_pos
    
    return best_pos, best_radii


def _build_hex(n):
    """Build a hexagonal initial configuration."""
    pos_list = []
    row = 0
    while len(pos_list) < n:
        cols = 6
        for col in range(cols):
            if len(pos_list) >= n:
                break
            x = (col + (0.5 if row % 2 else 0)) * 0.17
            y = (row + 0.5) * 0.149
            pos_list.append([x, y])
        row += 1
    pos = np.array(pos_list[:n])
    r = 0.08
    radii = np.full(n, r)
    return pos, radii


def _build_grid(n):
    """Build a grid initial configuration."""
    side = int(np.ceil(np.sqrt(n)))
    pos = np.array([[(i + 0.5) / side, (j + 0.5) / side]
                    for j in range(side) for i in range(side)][:n])
    r = 0.42 / side
    radii = np.full(n, r)
    return pos, radii


def _build_random(n, seed=0):
    """Build a random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    radii = np.full(n, 0.07)
    return pos, radii


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_pos = None
    best_radii = None
    
    pairs = _make_pairs(N)
    
    # Multiple initial configurations
    configs = []
    
    pos, radii = _build_hex(N)
    configs.append((pos.copy(), radii.copy(), "hex"))
    
    pos, radii = _build_grid(N)
    configs.append((pos.copy(), radii.copy(), "grid"))
    
    for seed in [42, 123, 456, 789, 1000]:
        pos, radii = _build_random(N, seed=seed)
        configs.append((pos.copy(), radii.copy(), f"rand{seed}"))
    
    # Process each configuration
    for ci, (pos, radii, name) in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        remaining = deadline - elapsed - 5
        
        # Phase 1: Relax overlaps
        time_alloc = min(5.0, remaining * 0.2)
        if time_alloc > 1:
            pos, radii = _relax(pos, radii, pairs, max_iters=2000,
                                 time_budget=time_alloc, start_time=start_time)
        
        # Phase 2: Scipy optimization with increasing penalty
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 5
        time_alloc = min(20.0, remaining * 0.5)
        if time_alloc > 2:
            penalties = [100.0, 1000.0, 10000.0, 100000.0]
            for p in penalties:
                if time.time() - start_time > deadline - 5:
                    break
                t_remaining = deadline - time.time() - 5
                if t_remaining < 2:
                    break
                pos, radii = _scipy_optimize(pos, radii,
                                              time_budget=min(4.0, t_remaining),
                                              start_time=start_time, penalty=p)
                _clamp_boundaries(pos, radii)
        
        # Phase 3: Local search polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 5
        time_alloc = min(10.0, remaining * 0.5)
        if time_alloc > 1:
            pos, radii = _local_search(pos, radii, pairs, time_budget=time_alloc,
                                        start_time=start_time)
        
        # Validate
        if _check_validity(pos, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
    
    # Final polish if time remains
    if best_pos is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 3
        if remaining > 5:
            # Try scipy optimization on best with high penalty
            backup_pos = best_pos.copy()
            backup_radii = best_radii.copy()
            pos2, radii2 = _scipy_optimize(best_pos, best_radii,
                                            time_budget=min(10.0, remaining * 0.6),
                                            start_time=start_time, penalty=100000.0)
            _clamp_boundaries(pos2, radii2)
            if _check_validity(pos2, radii2):
                new_score = np.sum(radii2)
                if new_score > best_score:
                    best_pos = pos2
                    best_radii = radii2
                    best_score = new_score
            
            # Try local search on best
            elapsed = time.time() - start_time
            remaining = deadline - elapsed - 3
            if remaining > 3:
                pos3, radii3 = _local_search(best_pos, best_radii, pairs,
                                              time_budget=min(5.0, remaining),
                                              start_time=start_time)
                if _check_validity(pos3, radii3):
                    new_score = np.sum(radii3)
                    if new_score > best_score:
                        best_pos = pos3
                        best_radii = radii3
                        best_score = new_score
    
    # Return best result
    if best_pos is None:
        pos, radii = _build_grid(N)
        best_pos, best_radii = pos, radii
    
    result_arr = np.zeros((N, 3))
    for i in range(N):
        result_arr[i] = [best_pos[i, 0], best_pos[i, 1], best_radii[i]]
    
    return result_arr
# EVOLVE-BLOCK-END
