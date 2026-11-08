"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize


def _check_validity(pos, radii, tol=1e-7):
    """Vectorized validity check."""
    n = len(pos)
    # Boundary checks
    if np.any(pos[:, 0] < radii - tol) or np.any(pos[:, 0] > 1 - radii + tol):
        return False
    if np.any(pos[:, 1] < radii - tol) or np.any(pos[:, 1] > 1 - radii + tol):
        return False
    if np.any(radii < -tol):
        return False
    # Pairwise overlap check
    for i in range(n):
        dx = pos[i, 0] - pos[i+1:, 0]
        dy = pos[i, 1] - pos[i+1:, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        if np.any(dist < radii[i] + radii[i+1:] - tol):
            return False
    return True


def _build_grid(n, jitter=0.0):
    """Build a grid initial configuration."""
    side = int(np.ceil(np.sqrt(n)))
    pos = np.array([[(i + 0.5) / side, (j + 0.5) / side]
                    for j in range(side) for i in range(side)][:n])
    if jitter > 0:
        pos += np.random.uniform(-jitter, jitter, size=pos.shape)
        pos = np.clip(pos, 0.01, 0.99)
    r = 0.45 / side
    radii = np.full(n, r)
    return pos, radii


def _build_hex(n, jitter=0.0):
    """Build a hexagonal initial configuration."""
    pos_list = []
    row = 0
    while len(pos_list) < n:
        cols = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
        for col in range(cols):
            if len(pos_list) >= n:
                break
            x = (col + (0.5 if row % 2 else 0)) * 0.55 / cols
            y = (row + 0.5) * 0.5 / (int(np.ceil(n / cols)) + 1)
            pos_list.append([x, y])
        row += 1
    pos = np.array(pos_list[:n])
    if jitter > 0:
        pos += np.random.uniform(-jitter, jitter, size=pos.shape)
        pos = np.clip(pos, 0.01, 0.99)
    r = 0.22
    radii = np.full(n, r)
    return pos, radii


def _build_random(n, seed=None):
    """Build a random initial configuration."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    radii = np.full(n, 0.08)
    return pos, radii


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


def _relax_and_grow(pos, radii, max_iters=5000, time_budget=30.0, start_time=None):
    """Physics-based relaxation with simultaneous radius growth."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()
    
    # Precompute pair indices
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Check boundary violations
        for i in range(n):
            # Left boundary
            if pos[i, 0] < radii[i]:
                pos[i, 0] = radii[i]
            # Right boundary
            if pos[i, 0] > 1 - radii[i]:
                pos[i, 0] = 1 - radii[i]
            # Bottom boundary
            if pos[i, 1] < radii[i]:
                pos[i, 1] = radii[i]
            # Top boundary
            if pos[i, 1] > 1 - radii[i]:
                pos[i, 1] = 1 - radii[i]
        
        # Resolve overlaps
        for i, j in pairs:
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            
            if dist < min_dist:
                if dist < 1e-10:
                    # Random small separation
                    angle = np.random.uniform(0, 2 * np.pi)
                    push = (min_dist - dist) / 2
                    pos[i, 0] -= push * np.cos(angle)
                    pos[i, 1] -= push * np.sin(angle)
                    pos[j, 0] += push * np.cos(angle)
                    pos[j, 1] += push * np.sin(angle)
                else:
                    # Push apart along the connecting line
                    push = (min_dist - dist) / 2
                    nx = dx / dist
                    ny = dy / dist
                    pos[i, 0] -= push * nx
                    pos[i, 1] -= push * ny
                    pos[j, 0] += push * nx
                    pos[j, 1] += push * ny
        
        # Re-check boundaries after overlap resolution
        for i in range(n):
            if pos[i, 0] < radii[i]:
                pos[i, 0] = radii[i]
            if pos[i, 0] > 1 - radii[i]:
                pos[i, 0] = 1 - radii[i]
            if pos[i, 1] < radii[i]:
                pos[i, 1] = radii[i]
            if pos[i, 1] > 1 - radii[i]:
                pos[i, 1] = 1 - radii[i]
        
        # Try to grow radii
        if it % 50 == 0:
            grow_amount = 0.001
            new_radii = radii + grow_amount
            # Check if growth is feasible
            feasible = True
            for i in range(n):
                if new_radii[i] > 0.5 or new_radii[i] < 0.01:
                    feasible = False
                    break
                if pos[i, 0] < new_radii[i] or pos[i, 0] > 1 - new_radii[i]:
                    feasible = False
                    break
                if pos[i, 1] < new_radii[i] or pos[i, 1] > 1 - new_radii[i]:
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
    
    return pos, radii


def _local_search(pos, radii, time_budget=10.0, start_time=None):
    """Local search: try small moves to improve packing."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    while time.time() - start_time < time_budget:
        # Try to grow all radii slightly
        grow = 0.0005
        new_radii = radii + grow
        feasible = True
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
            score = np.sum(radii)
            if score > best_score:
                best_score = score
            continue
        
        # Try small random moves
        for _ in range(100):
            if time.time() - start_time > time_budget:
                break
            idx = np.random.randint(n)
            old_pos = pos[idx].copy()
            pos[idx] += np.random.normal(0, 0.005, size=2)
            pos[idx] = np.clip(pos[idx], 0, 1)
            
            # Check validity
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
            if not valid:
                pos[idx] = old_pos
    
    return pos, radii


def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    best_score = 0.0
    best_pos = None
    best_radii = None
    
    # Fallback configuration: small circles in a grid
    def get_fallback():
        side = int(np.ceil(np.sqrt(N)))
        pos = np.array([[(i + 0.5) / side, (j + 0.5) / side]
                        for j in range(side) for i in range(side)][:N])
        radii = np.full(N, 0.45 / side)
        return pos, radii
    
    # Generate a few diverse initial configurations
    configs = []
    
    # 1. Grid
    pos, radii = _build_grid(N, jitter=0.0)
    configs.append((pos, radii))
    
    # 2. Hexagonal
    pos, radii = _build_hex(N, jitter=0.0)
    configs.append((pos, radii))
    
    # 3. Random with fixed seed
    pos, radii = _build_random(N, seed=42)
    configs.append((pos, radii))
    
    # Process each configuration
    for ci, (pos_init, radii_init) in enumerate(configs):
        if time.time() - start_time > deadline:
            break
        
        remaining = deadline - (time.time() - start_time)
        if remaining < 10:
            break
        
        pos = pos_init.copy()
        radii = radii_init.copy()
        
        # Step 1: Relax and grow to resolve overlaps and increase size
        # Reduce max_iters to ensure speed
        pos, radii = _relax_and_grow(pos, radii, max_iters=500, 
                                      time_budget=min(5.0, remaining * 0.3),
                                      start_time=start_time)
        
        # Validate
        if _check_validity(pos, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
        
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
    
    # Final check
    if best_pos is None:
        best_pos, best_radii = get_fallback()
    
    # Ensure final result is strictly valid
    if not _check_validity(best_pos, best_radii):
        best_pos, best_radii = get_fallback()
    
    result_arr = np.zeros((N, 3))
    for i in range(N):
        result_arr[i] = [best_pos[i, 0], best_pos[i, 1], best_radii[i]]
    
    return result_arr
# EVOLVE-BLOCK-END
