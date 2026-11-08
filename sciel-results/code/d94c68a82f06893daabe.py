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
    Vectorized penalty-based objective for circle packing.
    """
    pos = z.reshape(n, 3)[:, :2]
    radii = z.reshape(n, 3)[:, 2]
    
    # Base objective: negative sum of radii
    obj = -np.sum(radii)
    
    # Boundary penalties (vectorized)
    # Left: max(0, r - x)^2
    # Right: max(0, x - (1-r))^2
    # Bottom: max(0, r - y)^2
    # Top: max(0, y - (1-r))^2
    b_left = np.maximum(0, radii - pos[:, 0])
    b_right = np.maximum(0, pos[:, 0] - (1 - radii))
    b_bottom = np.maximum(0, radii - pos[:, 1])
    b_top = np.maximum(0, pos[:, 1] - (1 - radii))
    boundary_viol = np.sum(b_left**2) + np.sum(b_right**2) + np.sum(b_bottom**2) + np.sum(b_top**2)
    
    # Pairwise overlap penalties (vectorized)
    # Compute pairwise distances
    # dist[i, j] = ||pos[i] - pos[j]||
    # min_dist[i, j] = radii[i] + radii[j]
    # viol = max(0, min_dist - dist)^2
    
    # Use broadcasting to get all pair differences
    # pos_diff[i, j, k] = pos[i, k] - pos[j, k]
    pos_diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(pos_diff**2, axis=2)
    dist = np.sqrt(dist_sq + 1e-12) # avoid division by zero later if needed, though we use dist directly
    
    radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    diff = np.maximum(0, radii_sum - dist)
    
    # We only want upper triangle (i < j) to avoid double counting and self-interaction
    # Self interaction dist=0, radii_sum=2r, diff=2r. We should exclude i==j.
    # Mask for upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    overlap_viol = np.sum(diff[mask]**2)
    
    # Radius bounds penalty
    r_low = np.maximum(0, 0.01 - radii)
    r_high = np.maximum(0, radii - 0.5)
    radius_viol = np.sum(r_low**2) + np.sum(r_high**2)
    
    obj += penalty_weight * (boundary_viol + overlap_viol + radius_viol)
    
    return obj


def _penalty_gradient(z, n, penalty_weight=1000.0):
    """
    Vectorized gradient of penalty-based objective.
    """
    pos = z.reshape(n, 3)[:, :2].copy()
    radii = z.reshape(n, 3)[:, 2].copy()
    
    grad = np.zeros(3 * n)
    
    # Base gradient: -1 for each radius
    grad[2::3] = -1.0
    
    # Boundary gradients
    # Left: viol = max(0, r-x)^2. d/dx = -2(r-x) if x<r, else 0. d/dr = 2(r-x) if x<r, else 0.
    b_left = np.maximum(0, radii - pos[:, 0])
    grad[0::3] -= penalty_weight * 2 * b_left
    grad[2::3] += penalty_weight * 2 * b_left
    
    # Right: viol = max(0, x-(1-r))^2. d/dx = 2(x-(1-r)) if x>1-r, else 0. d/dr = -2(x-(1-r)) if x>1-r, else 0.
    b_right = np.maximum(0, pos[:, 0] - (1 - radii))
    grad[0::3] += penalty_weight * 2 * b_right
    grad[2::3] -= penalty_weight * 2 * b_right
    
    # Bottom: viol = max(0, r-y)^2. d/dy = -2(r-y) if y<r, else 0. d/dr = 2(r-y) if y<r, else 0.
    b_bottom = np.maximum(0, radii - pos[:, 1])
    grad[1::3] -= penalty_weight * 2 * b_bottom
    grad[2::3] += penalty_weight * 2 * b_bottom
    
    # Top: viol = max(0, y-(1-r))^2. d/dy = 2(y-(1-r)) if y>1-r, else 0. d/dr = -2(y-(1-r)) if y>1-r, else 0.
    b_top = np.maximum(0, pos[:, 1] - (1 - radii))
    grad[1::3] += penalty_weight * 2 * b_top
    grad[2::3] -= penalty_weight * 2 * b_top
    
    # Pairwise overlap gradients
    # dist[i, j] = ||pos[i] - pos[j]||
    pos_diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(pos_diff**2, axis=2)
    dist = np.sqrt(dist_sq + 1e-12)
    
    radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    diff = np.maximum(0, radii_sum - dist)
    
    # Mask for upper triangle (i < j)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Only consider pairs where diff > 0
    active_mask = mask & (diff > 0)
    
    if np.any(active_mask):
        # For active pairs (i, j), we need to accumulate gradients.
        # d/d(pos[i]) of (min_dist - dist)^2 = -2 * diff * (pos[i] - pos[j]) / dist
        # Note: pos_diff[i, j] = pos[i] - pos[j]. So grad_pos_i += -2 * diff * pos_diff[i, j] / dist
        # d/d(pos[j]) of (min_dist - dist)^2 = -2 * diff * (pos[j] - pos[i]) / dist = 2 * diff * pos_diff[i, j] / dist
        # d/d(r_i) = 2 * diff
        # d/d(r_j) = 2 * diff
        
        # We can use matrix operations or loops over active pairs.
        # With N=26, loops over active pairs are fast enough and safer for indexing.
        active_indices = np.argwhere(active_mask)
        
        for i, j in active_indices:
            d = dist[i, j]
            diff_val = diff[i, j]
            
            # Vector from i to j is pos[j] - pos[i] = -pos_diff[i, j]
            # Gradient of dist wrt pos[i] is (pos[i] - pos[j]) / dist = pos_diff[i, j] / dist
            # Gradient of (min_dist - dist)^2 wrt pos[i] is -2 * diff * (pos_diff[i, j] / dist)
            
            factor = -2 * penalty_weight * diff_val / d
            
            # pos[i] gradient
            grad[3*i + 0] += factor * pos_diff[i, j, 0]
            grad[3*i + 1] += factor * pos_diff[i, j, 1]
            
            # pos[j] gradient
            # Gradient wrt pos[j] is 2 * diff * (pos_diff[i, j] / dist)
            # Wait, let's re-verify.
            # f = (r_i + r_j - dist)^2
            # df/dpos_j = 2(r_i + r_j - dist) * d(-dist)/dpos_j = 2 diff * (- (pos_j - pos_i)/dist) = 2 diff * (pos_i - pos_j)/dist
            # pos_i - pos_j = pos_diff[i, j]
            # So grad_pos_j = 2 * penalty_weight * diff_val * pos_diff[i, j] / d
            # My previous line for grad_pos_i was: -2 * diff * pos_diff / d.
            # Let's check:
            # df/dpos_i = 2 diff * d(-dist)/dpos_i = 2 diff * (- (pos_i - pos_j)/dist) = -2 diff * pos_diff[i, j] / d.
            # Correct.
            
            grad[3*j + 0] -= factor * pos_diff[i, j, 0]
            grad[3*j + 1] -= factor * pos_diff[i, j, 1]
            
            # Radius gradients
            # df/dr_i = 2 diff
            # df/dr_j = 2 diff
            r_factor = 2 * penalty_weight * diff_val
            grad[3*i + 2] += r_factor
            grad[3*j + 2] += r_factor

    # Radius bounds gradients
    r_low = np.maximum(0, 0.01 - radii)
    r_high = np.maximum(0, radii - 0.5)
    
    # d/d(r) of (0.01 - r)^2 = -2(0.01 - r) = 2(r - 0.01)
    # d/d(r) of (r - 0.5)^2 = 2(r - 0.5)
    grad[2::3] += penalty_weight * (-2 * r_low + 2 * r_high)
    
    return grad


def _adam_optimize(pos, radii, time_budget=30.0, start_time=None, penalty_weight=1000.0):
    """
    Use Adam optimizer to optimize the packing.
    """
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    
    # Flatten to z = [x1, y1, r1, x2, y2, r2, ...]
    z = np.zeros(3 * n)
    for i in range(n):
        z[3*i + 0] = pos[i, 0]
        z[3*i + 1] = pos[i, 1]
        z[3*i + 2] = radii[i]
    
    # Adam state
    m = np.zeros_like(z)
    v = np.zeros_like(z)
    t = 0
    lr = 0.05
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.01, 0.5))
        
    def clip_to_bounds(z):
        z_clipped = z.copy()
        for i in range(n):
            z_clipped[3*i + 0] = np.clip(z_clipped[3*i + 0], bounds[i*3][0], bounds[i*3][1])
            z_clipped[3*i + 1] = np.clip(z_clipped[3*i + 1], bounds[i*3+1][0], bounds[i*3+1][1])
            z_clipped[3*i + 2] = np.clip(z_clipped[3*i + 2], bounds[i*3+2][0], bounds[i*3+2][1])
        return z_clipped

    last_obj = _penalty_objective(z, n, penalty_weight)
    
    while time.time() - start_time < time_budget:
        t += 1
        obj = _penalty_objective(z, n, penalty_weight)
        grad = _penalty_gradient(z, n, penalty_weight)
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        z -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        z = clip_to_bounds(z)
        
        # Decay learning rate
        if t % 500 == 0:
            lr *= 0.9
            
        # Check for stagnation
        if abs(last_obj - obj) < 1e-10:
            if t % 100 == 0:
                # Try a small random perturbation
                z += np.random.normal(0, 0.001, size=z.shape)
                z = clip_to_bounds(z)
                lr *= 1.1 # Increase lr slightly to escape
        last_obj = obj

    pos_opt = np.zeros((n, 2))
    radii_opt = np.zeros(n)
    for i in range(n):
        pos_opt[i, 0] = z[3*i + 0]
        pos_opt[i, 1] = z[3*i + 1]
        radii_opt[i] = z[3*i + 2]
    
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
        
        # Try to grow radii using binary search for max feasible uniform growth
        if it % 10 == 0:
            # Binary search for max growth factor
            lo, hi = 0.0, 0.1 # Max possible growth per step is limited by geometry
            best_growth = 0.0
            
            for _ in range(10): # 10 iterations of binary search
                mid = (lo + hi) / 2
                new_radii = radii + mid
                
                # Check bounds
                if np.any(new_radii > 0.5) or np.any(new_radii < 0.01):
                    hi = mid
                    continue
                
                # Check boundaries
                if np.any(pos[:, 0] < new_radii) or np.any(pos[:, 0] > 1 - new_radii):
                    hi = mid
                    continue
                if np.any(pos[:, 1] < new_radii) or np.any(pos[:, 1] > 1 - new_radii):
                    hi = mid
                    continue
                
                # Check overlaps
                # Vectorized check
                pos_diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
                dist = np.sqrt(np.sum(pos_diff**2, axis=2))
                radii_sum = new_radii[:, np.newaxis] + new_radii[np.newaxis, :]
                
                # Check upper triangle
                mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                if np.any(dist[mask] < radii_sum[mask] - 1e-9):
                    hi = mid
                else:
                    lo = mid
                    best_growth = mid
            
            if best_growth > 1e-6:
                radii += best_growth
    
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
        
        # Step 2: Adam Optimization
        remaining = deadline - (time.time() - start_time)
        if remaining > 10:
            pos_adam, radii_adam = _adam_optimize(pos, radii, 
                                                       time_budget=min(20.0, remaining * 0.7),
                                                       start_time=start_time,
                                                       penalty_weight=1000.0)
            
            # Validate
            if _check_validity(pos_adam, radii_adam):
                score = np.sum(radii_adam)
                if score > best_score:
                    best_score = score
                    best_pos = pos_adam.copy()
                    best_radii = radii_adam.copy()
        
        # Step 3: Local Search Polish
        remaining = deadline - (time.time() - start_time)
        if remaining > 5 and best_pos is not None:
            # Polish the best found so far
            best_pos, best_radii = _local_search(best_pos, best_radii, 
                                                   time_budget=min(10.0, remaining - 2),
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
