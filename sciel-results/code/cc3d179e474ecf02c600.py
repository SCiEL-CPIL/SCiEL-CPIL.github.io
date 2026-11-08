<<<<<<< SEARCH
def _penalty_objective(z, n, penalty_weight=1000.0):
    """
    Penalty-based objective for circle packing.
    z = [x1, y1, r1, x2, y2, r2, ...] (length 3n)
    Returns: -sum(radii) + penalty for violations
    """
    pos = z.reshape(n, 3)[:, :2]
    radii = z.reshape(n, 3)[:, 2]
    
    # Base objective: negative sum of radii
    obj = -np.sum(radii)
    
    # Boundary penalties
    boundary_viol = 0.0
    for i in range(n):
        if pos[i, 0] < radii[i]:
            boundary_viol += (radii[i] - pos[i, 0]) ** 2
        if pos[i, 0] > 1 - radii[i]:
            boundary_viol += (pos[i, 0] - (1 - radii[i])) ** 2
        if pos[i, 1] < radii[i]:
            boundary_viol += (radii[i] - pos[i, 1]) ** 2
        if pos[i, 1] > 1 - radii[i]:
            boundary_viol += (pos[i, 1] - (1 - radii[i])) ** 2
    
    # Pairwise overlap penalties
    overlap_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                overlap_viol += (min_dist - dist) ** 2
    
    # Radius bounds penalty
    radius_viol = 0.0
    for i in range(n):
        if radii[i] < 0.01:
            radius_viol += (0.01 - radii[i]) ** 2
        if radii[i] > 0.5:
            radius_viol += (radii[i] - 0.5) ** 2
    
    obj += penalty_weight * (boundary_viol + overlap_viol + radius_viol)
    
    return obj
=======
def _penalty_objective(z, n, penalty_weight=1000.0):
    """
    Penalty-based objective for circle packing.
    z = [x1, y1, r1, x2, y2, r2, ...] (length 3n)
    Returns: -sum(radii) + penalty for violations
    """
    pos = z.reshape(n, 3)[:, :2]
    radii = z.reshape(n, 3)[:, 2]
    
    # Base objective: negative sum of radii
    obj = -np.sum(radii)
    
    # Vectorized boundary penalties
    boundary_viol = 0.0
    # Left boundary: pos[i,0] < radii[i]
    viol = radii - pos[:, 0]
    mask = viol > 0
    if np.any(mask):
        boundary_viol += np.sum(viol[mask] ** 2)
    # Right boundary: pos[i,0] > 1 - radii[i]
    viol = pos[:, 0] - (1 - radii)
    mask = viol > 0
    if np.any(mask):
        boundary_viol += np.sum(viol[mask] ** 2)
    # Bottom boundary: pos[i,1] < radii[i]
    viol = radii - pos[:, 1]
    mask = viol > 0
    if np.any(mask):
        boundary_viol += np.sum(viol[mask] ** 2)
    # Top boundary: pos[i,1] > 1 - radii[i]
    viol = pos[:, 1] - (1 - radii)
    mask = viol > 0
    if np.any(mask):
        boundary_viol += np.sum(viol[mask] ** 2)
    
    # Vectorized pairwise overlap penalties
    overlap_viol = 0.0
    for i in range(n - 1):
        dx = pos[i + 1:, 0] - pos[i, 0]
        dy = pos[i + 1:, 1] - pos[i, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = radii[i] + radii[i + 1:]
        viol = min_dist - dist
        mask = viol > 0
        if np.any(mask):
            overlap_viol += np.sum(viol[mask] ** 2)
    
    # Radius bounds penalty (vectorized)
    radius_viol = 0.0
    viol = 0.01 - radii
    mask = viol > 0
    if np.any(mask):
        radius_viol += np.sum(viol[mask] ** 2)
    viol = radii - 0.5
    mask = viol > 0
    if np.any(mask):
        radius_viol += np.sum(viol[mask] ** 2)
    
    obj += penalty_weight * (boundary_viol + overlap_viol + radius_viol)
    
    return obj
>>>>>>> REPLACE
<<<<<<< SEARCH
def _penalty_gradient(z, n, penalty_weight=1000.0):
    """
    Gradient of penalty-based objective.
    z = [x1, y1, r1, x2, y2, r2, ...] (length 3n)
    """
    pos = z.reshape(n, 3)[:, :2].copy()
    radii = z.reshape(n, 3)[:, 2].copy()
    
    grad = np.zeros(3 * n)
    
    # Base gradient: -1 for each radius
    for i in range(n):
        grad[3*i + 2] = -1.0
    
    # Boundary gradients
    for i in range(n):
        # x boundary
        if pos[i, 0] < radii[i]:
            grad[3*i + 0] += penalty_weight * 2 * (radii[i] - pos[i, 0]) * (-1)
            grad[3*i + 2] += penalty_weight * 2 * (radii[i] - pos[i, 0]) * (1)
        if pos[i, 0] > 1 - radii[i]:
            grad[3*i + 0] += penalty_weight * 2 * (pos[i, 0] - (1 - radii[i])) * (1)
            grad[3*i + 2] += penalty_weight * 2 * (pos[i, 0] - (1 - radii[i])) * (-1)
        # y boundary
        if pos[i, 1] < radii[i]:
            grad[3*i + 1] += penalty_weight * 2 * (radii[i] - pos[i, 1]) * (-1)
            grad[3*i + 2] += penalty_weight * 2 * (radii[i] - pos[i, 1]) * (1)
        if pos[i, 1] > 1 - radii[i]:
            grad[3*i + 1] += penalty_weight * 2 * (pos[i, 1] - (1 - radii[i])) * (1)
            grad[3*i + 2] += penalty_weight * 2 * (pos[i, 1] - (1 - radii[i])) * (-1)
    
    # Pairwise overlap gradients
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist and dist > 1e-10:
                diff = min_dist - dist
                # d/dx_i of (min_dist - dist)^2 = -2*(min_dist-dist)*d(dist)/d(x_i)
                # d(dist)/d(x_i) = -dx/dist
                grad[3*i + 0] += penalty_weight * 2 * diff * (dx / dist)
                grad[3*i + 1] += penalty_weight * 2 * diff * (dy / dist)
                grad[3*j + 0] += penalty_weight * 2 * diff * (-dx / dist)
                grad[3*j + 1] += penalty_weight * 2 * diff * (-dy / dist)
                # d/d(r_i) of (min_dist - dist)^2 = 2*(min_dist-dist)*1
                grad[3*i + 2] += penalty_weight * 2 * diff
                grad[3*j + 2] += penalty_weight * 2 * diff
    
    # Radius bounds gradients
    for i in range(n):
        if radii[i] < 0.01:
            grad[3*i + 2] += penalty_weight * 2 * (0.01 - radii[i]) * (-1)
        if radii[i] > 0.5:
            grad[3*i + 2] += penalty_weight * 2 * (radii[i] - 0.5) * (1)
    
    return grad
=======
def _penalty_gradient(z, n, penalty_weight=1000.0):
    """
    Gradient of penalty-based objective.
    z = [x1, y1, r1, x2, y2, r2, ...] (length 3n)
    """
    pos = z.reshape(n, 3)[:, :2].copy()
    radii = z.reshape(n, 3)[:, 2].copy()
    
    grad = np.zeros(3 * n)
    
    # Base gradient: -1 for each radius
    grad[2::3] = -1.0
    
    # Vectorized boundary gradients
    # Left boundary: pos[i,0] < radii[i]
    viol = radii - pos[:, 0]
    mask = viol > 0
    if np.any(mask):
        grad[mask, 0] += penalty_weight * 2 * viol[mask] * (-1)
        grad[mask, 2] += penalty_weight * 2 * viol[mask] * (1)
    # Right boundary: pos[i,0] > 1 - radii[i]
    viol = pos[:, 0] - (1 - radii)
    mask = viol > 0
    if np.any(mask):
        grad[mask, 0] += penalty_weight * 2 * viol[mask] * (1)
        grad[mask, 2] += penalty_weight * 2 * viol[mask] * (-1)
    # Bottom boundary: pos[i,1] < radii[i]
    viol = radii - pos[:, 1]
    mask = viol > 0
    if np.any(mask):
        grad[mask, 1] += penalty_weight * 2 * viol[mask] * (-1)
        grad[mask, 2] += penalty_weight * 2 * viol[mask] * (1)
    # Top boundary: pos[i,1] > 1 - radii[i]
    viol = pos[:, 1] - (1 - radii)
    mask = viol > 0
    if np.any(mask):
        grad[mask, 1] += penalty_weight * 2 * viol[mask] * (1)
        grad[mask, 2] += penalty_weight * 2 * viol[mask] * (-1)
    
    # Vectorized pairwise overlap gradients
    for i in range(n - 1):
        dx = pos[i + 1:, 0] - pos[i, 0]
        dy = pos[i + 1:, 1] - pos[i, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = radii[i] + radii[i + 1:]
        diff = min_dist - dist
        mask = (diff > 0) & (dist > 1e-10)
        if np.any(mask):
            idx = np.where(mask)[0] + i + 1
            nx = dx[mask] / dist[mask]
            ny = dy[mask] / dist[mask]
            d = penalty_weight * 2 * diff[mask]
            # For circle i
            grad[i, 0] += np.sum(d * nx)
            grad[i, 1] += np.sum(d * ny)
            grad[i, 2] += np.sum(d)
            # For circles j
            grad[idx, 0] -= d * nx
            grad[idx, 1] -= d * ny
            grad[idx, 2] += d
    
    # Vectorized radius bounds gradients
    viol = 0.01 - radii
    mask = viol > 0
    if np.any(mask):
        grad[mask, 2] += penalty_weight * 2 * viol[mask] * (-1)
    viol = radii - 0.5
    mask = viol > 0
    if np.any(mask):
        grad[mask, 2] += penalty_weight * 2 * viol[mask] * (1)
    
    return grad
>>>>>>> REPLACE
<<<<<<< SEARCH
def _scipy_optimize(pos, radii, time_budget=30.0, start_time=None, penalty_weight=1000.0):
    """
    Use scipy.optimize.minimize with L-BFGS-B to optimize the packing.
    """
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    
    # Flatten to z = [x1, y1, r1, x2, y2, r2, ...]
    z0 = np.zeros(3 * n)
    for i in range(n):
        z0[3*i + 0] = pos[i, 0]
        z0[3*i + 1] = pos[i, 1]
        z0[3*i + 2] = radii[i]
    
    # Bounds: x in [0,1], y in [0,1], r in [0.01, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.01, 0.5))  # r
    
    # Time callback to stop if budget exceeded
    stop_flag = [False]
    def callback(x):
        if time.time() - start_time > time_budget:
            stop_flag[0] = True
            return True
        return False
    
    result = minimize(
        _penalty_objective,
        z0,
        args=(n, penalty_weight),
        jac=_penalty_gradient,
        method='L-BFGS-B',
        bounds=bounds,
        callback=callback,
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    z_opt = result.x
    pos_opt = np.zeros((n, 2))
    radii_opt = np.zeros(n)
    for i in range(n):
        pos_opt[i, 0] = z_opt[3*i + 0]
        pos_opt[i, 1] = z_opt[3*i + 1]
        radii_opt[i] = z_opt[3*i + 2]
    
    return pos_opt, radii_opt
=======
def _scipy_optimize(pos, radii, time_budget=30.0, start_time=None, penalty_weight=1000.0):
    """
    Use scipy.optimize.minimize with L-BFGS-B to optimize the packing.
    """
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    
    # Flatten to z = [x1, y1, r1, x2, y2, r2, ...]
    z0 = np.zeros(3 * n)
    for i in range(n):
        z0[3*i + 0] = pos[i, 0]
        z0[3*i + 1] = pos[i, 1]
        z0[3*i + 2] = radii[i]
    
    # Bounds: x in [0,1], y in [0,1], r in [0.01, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.01, 0.5))  # r
    
    # Time callback to stop if budget exceeded
    stop_flag = [False]
    def callback(x):
        if time.time() - start_time > time_budget:
            stop_flag[0] = True
            return True
        return False
    
    result = minimize(
        _penalty_objective,
        z0,
        args=(n, penalty_weight),
        jac=_penalty_gradient,
        method='L-BFGS-B',
        bounds=bounds,
        callback=callback,
        options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
    )
    
    z_opt = result.x
    pos_opt = np.zeros((n, 2))
    radii_opt = np.zeros(n)
    for i in range(n):
        pos_opt[i, 0] = z_opt[3*i + 0]
        pos_opt[i, 1] = z_opt[3*i + 1]
        radii_opt[i] = z_opt[3*i + 2]
    
    return pos_opt, radii_opt
>>>>>>> REPLACE
<<<<<<< SEARCH
        # Step 2: SciPy Optimization
        remaining = deadline - (time.time() - start_time)
        if remaining > 10:
            pos_scipy, radii_scipy = _scipy_optimize(pos, radii, 
                                                       time_budget=min(10.0, remaining * 0.6),
                                                       start_time=start_time,
                                                       penalty_weight=1000.0)
            
            # Validate
            if _check_validity(pos_scipy, radii_scipy):
                score = np.sum(radii_scipy)
                if score > best_score:
                    best_score = score
                    best_pos = pos_scipy.copy()
                    best_radii = radii_scipy.copy()
=======
        # Step 2: SciPy Optimization with multiple penalty weights
        remaining = deadline - (time.time() - start_time)
        if remaining > 10:
            for pw in [500.0, 1000.0, 2000.0]:
                if time.time() - start_time > deadline - 5:
                    break
                pos_scipy, radii_scipy = _scipy_optimize(pos, radii, 
                                                           time_budget=min(8.0, (remaining - 5) / 3),
                                                           start_time=start_time,
                                                           penalty_weight=pw)
                
                # Validate
                if _check_validity(pos_scipy, radii_scipy):
                    score = np.sum(radii_scipy)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_scipy.copy()
                        best_radii = radii_scipy.copy()
>>>>>>> REPLACE
<<<<<<< SEARCH
        # Step 3: Local Search Polish
        remaining = deadline - (time.time() - start_time)
        if remaining > 5 and best_pos is not None:
            # Use the current best or the one from this config? Use current best to polish.
            # But if this config produced a better result, best_pos is updated.
            best_pos, best_radii = _local_search(best_pos, best_radii, 
                                                   time_budget=min(5.0, remaining - 2),
                                                   start_time=start_time)
            if _check_validity(best_pos, best_radii):
                best_score = np.sum(best_radii)
=======
        # Step 3: Local Search Polish
        remaining = deadline - (time.time() - start_time)
        if remaining > 5 and best_pos is not None:
            # Use the current best or the one from this config? Use current best to polish.
            # But if this config produced a better result, best_pos is updated.
            best_pos, best_radii = _local_search(best_pos, best_radii, 
                                                   time_budget=min(5.0, remaining - 2),
                                                   start_time=start_time)
            if _check_validity(best_pos, best_radii):
                best_score = np.sum(best_radii)
    
    # Final high-precision polish with scipy
    remaining = deadline - (time.time() - start_time)
    if remaining > 15 and best_pos is not None:
        pos_final, radii_final = _scipy_optimize(best_pos, best_radii,
                                                  time_budget=min(12.0, remaining - 3),
                                                  start_time=start_time,
                                                  penalty_weight=5000.0)
        if _check_validity(pos_final, radii_final):
            score = np.sum(radii_final)
            if score > best_score:
                best_score = score
                best_pos = pos_final.copy()
                best_radii = radii_final.copy()
        
        # Another local search pass
        remaining = deadline - (time.time() - start_time)
        if remaining > 5 and best_pos is not None:
            best_pos, best_radii = _local_search(best_pos, best_radii,
                                                   time_budget=min(5.0, remaining - 2),
                                                   start_time=start_time)
            if _check_validity(best_pos, best_radii):
                best_score = np.sum(best_radii)
>>>>>>> REPLACE
BET: {mechanism: "Vectorized penalty objective/gradient enables faster scipy iterations, plus multi-penalty-weight starts and a final high-penalty polish pass should escape local optima and converge tighter", q: 0.65, delta: [0.02, 0.08]}
