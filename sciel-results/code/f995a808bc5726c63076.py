# EVOLVE-BLOCK-START
import numpy as np
import time


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
    # Create pair indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = pos[i_idx, 0] - pos[j_idx, 0]
    dy = pos[i_idx, 1] - pos[j_idx, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = radii[i_idx] + radii[j_idx]
    if np.any(dist < min_dist - tol):
        return False
    return True


def _make_pairs(n):
    i_idx, j_idx = np.triu_indices(n, k=1)
    return i_idx, j_idx


def _feasible_grow(pos, radii, grow, pairs):
    """Check if we can grow all radii by 'grow' amount."""
    n = len(pos)
    i_idx, j_idx = pairs
    new_radii = radii + grow
    if np.any(new_radii > 0.5):
        return False, new_radii
    if np.any(pos[:, 0] < new_radii - 1e-9) or np.any(pos[:, 0] > 1 - new_radii + 1e-9):
        return False, new_radii
    if np.any(pos[:, 1] < new_radii - 1e-9) or np.any(pos[:, 1] > 1 - new_radii + 1e-9):
        return False, new_radii
    
    dx = pos[i_idx, 0] - pos[j_idx, 0]
    dy = pos[i_idx, 1] - pos[j_idx, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = new_radii[i_idx] + new_radii[j_idx]
    if np.any(dist < min_dist - 1e-9):
        return False, new_radii
    return True, new_radii


def _clamp_boundaries(pos, radii):
    pos[:, 0] = np.clip(pos[:, 0], radii, 1 - radii)
    pos[:, 1] = np.clip(pos[:, 1], radii, 1 - radii)


def _relax(pos, radii, pairs, max_iters=2000, time_budget=30.0, start_time=None):
    """Iterative overlap resolution with boundary clamping."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    i_idx, j_idx = pairs
    pos = pos.copy()
    radii = radii.copy()
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        _clamp_boundaries(pos, radii)
        
        # Vectorized overlap resolution
        dx = pos[i_idx, 0] - pos[j_idx, 0]
        dy = pos[i_idx, 1] - pos[j_idx, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = radii[i_idx] + radii[j_idx]
        
        # Find overlaps
        overlap = min_dist - dist
        # Mask where dist is very small to avoid division by zero
        valid = dist > 1e-12
        overlap[~valid] = 0 # No push if exactly same pos, handled separately or ignored
        
        # Calculate push vectors
        # push = overlap / 2 * unit_vector
        # We need to accumulate pushes for each circle
        push_x = np.zeros(n)
        push_y = np.zeros(n)
        
        if np.any(overlap > 0):
            # Unit vectors
            nx = np.zeros_like(dist)
            ny = np.zeros_like(dist)
            nx[valid] = dx[valid] / dist[valid]
            ny[valid] = dy[valid] / dist[valid]
            
            push_mag = overlap / 2.0
            
            # Accumulate
            # i_idx: circle i gets pushed in -n direction
            # j_idx: circle j gets pushed in +n direction
            np.add.at(push_x, i_idx, -push_mag * nx)
            np.add.at(push_y, i_idx, -push_mag * ny)
            np.add.at(push_x, j_idx, push_mag * nx)
            np.add.at(push_y, j_idx, push_mag * ny)
            
            pos[:, 0] += push_x
            pos[:, 1] += push_y
        
        _clamp_boundaries(pos, radii)
        
        # Check convergence
        if not np.any(overlap > 1e-9):
            break
    
    return pos, radii


def _grow_radii(pos, radii, pairs, time_budget=5.0, start_time=None):
    """Try to grow radii as much as possible."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()
    
    grow = 0.002
    max_grow_attempts = 500
    
    for _ in range(max_grow_attempts):
        if time.time() - start_time > time_budget:
            break
        
        feasible, new_radii = _feasible_grow(pos, radii, grow, pairs)
        if feasible:
            radii = new_radii
            pos, radii = _relax(pos, radii, pairs, max_iters=200,
                                  time_budget=min(1.0, time_budget - (time.time() - start_time)),
                                  start_time=start_time)
        else:
            lo, hi = 0.0, grow
            for _ in range(20):
                mid = (lo + hi) / 2
                feasible_mid, _ = _feasible_grow(pos, radii, mid, pairs)
                if feasible_mid:
                    lo = mid
                else:
                    hi = mid
            if lo > 1e-6:
                _, new_radii = _feasible_grow(pos, radii, lo, pairs)
                radii = new_radii
                pos, radii = _relax(pos, radii, pairs, max_iters=200,
                                      time_budget=min(0.5, time_budget - (time.time() - start_time)),
                                      start_time=start_time)
            grow *= 0.7
            if grow < 1e-6:
                break
    
    return pos, radii


def _local_search(pos, radii, pairs, time_budget=20.0, start_time=None):
    """Local search with random moves and growth."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    i_idx, j_idx = pairs
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    rng = np.random.RandomState(int(time.time() * 1000) % (2**31))
    
    while time.time() - start_time < time_budget:
        feasible, new_radii = _feasible_grow(pos, radii, 0.001, pairs)
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
            pos[idx, 0] = np.clip(pos[idx, 0], radii[idx], 1 - radii[idx])
            pos[idx, 1] = np.clip(pos[idx, 1], radii[idx], 1 - radii[idx])
            
            # Check validity for this move
            valid = True
            # Check boundaries
            if pos[idx, 0] < radii[idx] - 1e-9 or pos[idx, 0] > 1 - radii[idx] + 1e-9:
                valid = False
            if pos[idx, 1] < radii[idx] - 1e-9 or pos[idx, 1] > 1 - radii[idx] + 1e-9:
                valid = False
            
            if valid:
                # Check overlaps with all others
                # Vectorized check against idx
                dx = pos[:, 0] - pos[idx, 0]
                dy = pos[:, 1] - pos[idx, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                dist[idx] = np.inf # Exclude self
                min_dist = radii + radii[idx]
                min_dist[idx] = np.inf
                if np.any(dist < min_dist - 1e-9):
                    valid = False
            
            if valid:
                feasible2, new_r = _feasible_grow(pos, radii, 0.0005, pairs)
                if feasible2:
                    radii = new_r
                    score = np.sum(radii)
                    if score > best_score:
                        best_score = score
                        best_pos = pos.copy()
                        best_radii = radii.copy()
            else:
                pos[idx] = old_pos
    
    return best_pos, best_radii


def _vectorized_loss_and_grad(pos, radii, pairs, beta=100.0):
    """
    Calculate loss and gradient for Adam optimizer.
    Loss = -sum(radii) + penalty
    """
    n = len(pos)
    i_idx, j_idx = pairs
    
    # Flatten variables for easy handling? Or keep separate.
    # Let's keep pos (N,2) and radii (N,)
    
    loss = -np.sum(radii)
    grad_pos = np.zeros_like(pos)
    grad_rad = np.zeros_like(radii)
    
    # Overlap penalty
    dx = pos[i_idx, 0] - pos[j_idx, 0]
    dy = pos[i_idx, 1] - pos[j_idx, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    # Avoid div by zero
    dist_safe = np.where(dist < 1e-12, 1e-12, dist)
    
    viol = radii[i_idx] + radii[j_idx] - dist
    # Softplus: log(1 + exp(beta * viol)) / beta
    # Stable version:
    # If beta*viol > 50, approx viol
    # Else log1p(exp(beta*viol))/beta
    bv = beta * viol
    # Clip to avoid overflow in exp
    bv_clipped = np.clip(bv, -50, 50)
    soft_val = np.where(bv > 50, viol, np.log1p(np.exp(bv_clipped)) / beta)
    # Gradient of softplus: sigmoid(beta * viol)
    grad_v = np.where(bv > 50, 1.0, 1.0 / (1.0 + np.exp(-bv_clipped)))
    
    loss += np.sum(soft_val)
    
    # Gradients for overlaps
    # d(soft)/d(viol) = grad_v
    # d(viol)/d(pos_i) = -d(dist)/d(pos_i) = (pos_j - pos_i)/dist
    # d(viol)/d(pos_j) = d(dist)/d(pos_j) = (pos_i - pos_j)/dist
    # d(viol)/d(r_i) = 1
    # d(viol)/d(r_j) = 1
    
    # Accumulate gradients
    # grad_pos[i] += grad_v * (pos_j - pos_i) / dist
    # grad_pos[j] += grad_v * (pos_i - pos_j) / dist
    
    nx = (pos[j_idx, 0] - pos[i_idx, 0]) / dist_safe
    ny = (pos[j_idx, 1] - pos[i_idx, 1]) / dist_safe
    
    push_x = grad_v * nx
    push_y = grad_v * ny
    
    np.add.at(grad_pos[:, 0], i_idx, push_x)
    np.add.at(grad_pos[:, 1], i_idx, push_y)
    np.add.at(grad_pos[:, 0], j_idx, -push_x)
    np.add.at(grad_pos[:, 1], j_idx, -push_y)
    
    np.add.at(grad_rad, i_idx, grad_v)
    np.add.at(grad_rad, j_idx, grad_v)
    
    # Boundary penalties
    # x >= r
    viol_x1 = radii - pos[:, 0]
    bv_x1 = beta * viol_x1
    bv_x1_clipped = np.clip(bv_x1, -50, 50)
    soft_x1 = np.where(bv_x1 > 50, viol_x1, np.log1p(np.exp(bv_x1_clipped)) / beta)
    grad_x1 = np.where(bv_x1 > 50, 1.0, 1.0 / (1.0 + np.exp(-bv_x1_clipped)))
    
    loss += np.sum(soft_x1)
    grad_pos[:, 0] -= grad_x1
    grad_rad += grad_x1
    
    # x <= 1 - r  => x + r <= 1 => viol = x - (1-r) = x - 1 + r
    viol_x2 = pos[:, 0] - (1 - radii)
    bv_x2 = beta * viol_x2
    bv_x2_clipped = np.clip(bv_x2, -50, 50)
    soft_x2 = np.where(bv_x2 > 50, viol_x2, np.log1p(np.exp(bv_x2_clipped)) / beta)
    grad_x2 = np.where(bv_x2 > 50, 1.0, 1.0 / (1.0 + np.exp(-bv_x2_clipped)))
    
    loss += np.sum(soft_x2)
    grad_pos[:, 0] += grad_x2
    grad_rad -= grad_x2
    
    # y >= r
    viol_y1 = radii - pos[:, 1]
    bv_y1 = beta * viol_y1
    bv_y1_clipped = np.clip(bv_y1, -50, 50)
    soft_y1 = np.where(bv_y1 > 50, viol_y1, np.log1p(np.exp(bv_y1_clipped)) / beta)
    grad_y1 = np.where(bv_y1 > 50, 1.0, 1.0 / (1.0 + np.exp(-bv_y1_clipped)))
    
    loss += np.sum(soft_y1)
    grad_pos[:, 1] -= grad_y1
    grad_rad += grad_y1
    
    # y <= 1 - r
    viol_y2 = pos[:, 1] - (1 - radii)
    bv_y2 = beta * viol_y2
    bv_y2_clipped = np.clip(bv_y2, -50, 50)
    soft_y2 = np.where(bv_y2 > 50, viol_y2, np.log1p(np.exp(bv_y2_clipped)) / beta)
    grad_y2 = np.where(bv_y2 > 50, 1.0, 1.0 / (1.0 + np.exp(-bv_y2_clipped)))
    
    loss += np.sum(soft_y2)
    grad_pos[:, 1] += grad_y2
    grad_rad -= grad_y2
    
    # Objective: Minimize loss = -sum(r) + penalties
    # We already added -sum(r) to loss.
    # Gradient of -sum(r) is -1 for each r.
    grad_rad -= 1.0
    
    return loss, grad_pos, grad_rad


def _adam_optimize(pos0, radii0, pairs, time_budget=10.0, start_time=None, beta=100.0, lr=0.01):
    """
    Adam optimizer for the vectorized loss.
    """
    if start_time is None:
        start_time = time.time()
    n = len(pos0)
    
    pos = pos0.copy()
    radii = radii0.copy()
    
    # Adam state
    m_pos = np.zeros_like(pos)
    v_pos = np.zeros_like(pos)
    m_rad = np.zeros_like(radii)
    v_rad = np.zeros_like(radii)
    
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    t = 0
    
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    while time.time() - start_time < time_budget:
        t += 1
        loss, grad_pos, grad_rad = _vectorized_loss_and_grad(pos, radii, pairs, beta=beta)
        
        # Adam update
        m_pos = b1 * m_pos + (1 - b1) * grad_pos
        v_pos = b2 * v_pos + (1 - b2) * (grad_pos ** 2)
        m_hat_pos = m_pos / (1 - b1 ** t)
        v_hat_pos = v_pos / (1 - b2 ** t)
        
        m_rad = b1 * m_rad + (1 - b1) * grad_rad
        v_rad = b2 * v_rad + (1 - b2) * (grad_rad ** 2)
        m_hat_rad = m_rad / (1 - b1 ** t)
        v_hat_rad = v_rad / (1 - b2 ** t)
        
        pos -= lr * m_hat_pos / (np.sqrt(v_hat_pos) + eps)
        radii -= lr * m_hat_rad / (np.sqrt(v_hat_rad) + eps)
        
        # Clamp
        _clamp_boundaries(pos, radii)
        
        # Check validity and update best
        # We can't easily check validity with vectorized softplus directly for strict validity,
        # but we can check if the penalty is very small.
        # Or just run _check_validity periodically? It's O(N^2).
        # Let's check every 10 steps.
        if t % 10 == 0:
            if _check_validity(pos, radii):
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_pos = pos.copy()
                    best_radii = radii.copy()
            
            # Adaptive beta? Start low, increase.
            # For now, keep beta fixed or simple schedule.
    
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


def _build_corner(n):
    """Place circles at corners and edges for a structured start."""
    pos_list = []
    r_corner = 0.12
    pos_list.append([r_corner, r_corner])
    pos_list.append([1 - r_corner, r_corner])
    pos_list.append([r_corner, 1 - r_corner])
    pos_list.append([1 - r_corner, 1 - r_corner])
    
    r_edge = 0.1
    pos_list.append([0.5, r_edge])
    pos_list.append([0.5, 1 - r_edge])
    pos_list.append([r_edge, 0.5])
    pos_list.append([1 - r_edge, 0.5])
    
    remaining = n - len(pos_list)
    if remaining > 0:
        side = int(np.ceil(np.sqrt(remaining)))
        for j in range(side):
            for i in range(side):
                if len(pos_list) >= n:
                    break
                x = 0.25 + (i + 0.5) * 0.5 / side
                y = 0.25 + (j + 0.5) * 0.5 / side
                pos_list.append([x, y])
    
    pos = np.array(pos_list[:n])
    radii = np.full(n, 0.08)
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
    
    pos, radii = _build_corner(N)
    configs.append((pos.copy(), radii.copy(), "corner"))
    
    for seed in [42, 123, 456, 789, 1000]:
        pos, radii = _build_random(N, seed=seed)
        configs.append((pos.copy(), radii.copy(), f"rand{seed}"))
    
    rng = np.random.RandomState(int(time.time() * 1000) % (2**31))
    
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
        
        # Phase 2: Grow radii
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 5
        time_alloc = min(5.0, remaining * 0.2)
        if time_alloc > 1:
            pos, radii = _grow_radii(pos, radii, pairs, time_budget=time_alloc,
                                      start_time=start_time)
        
        # Phase 3: Adam Optimization (Vectorized Gradient Descent)
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 5
        time_alloc = min(20.0, remaining * 0.5)
        if time_alloc > 2:
            # Use multiple betas for annealing
            betas = [50.0, 100.0, 200.0, 500.0]
            for b in betas:
                if time.time() - start_time > deadline - 5:
                    break
                t_remaining = deadline - time.time() - 5
                if t_remaining < 2:
                    break
                pos, radii = _adam_optimize(pos, radii, pairs,
                                              time_budget=min(5.0, t_remaining),
                                              start_time=start_time, beta=b, lr=0.02)
                _clamp_boundaries(pos, radii)
        
        # Phase 4: Local search polish
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
            # Try Adam optimization on best with high beta
            backup_pos = best_pos.copy()
            backup_radii = best_radii.copy()
            pos2, radii2 = _adam_optimize(best_pos, best_radii, pairs,
                                            time_budget=min(10.0, remaining * 0.6),
                                            start_time=start_time, beta=500.0, lr=0.01)
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
