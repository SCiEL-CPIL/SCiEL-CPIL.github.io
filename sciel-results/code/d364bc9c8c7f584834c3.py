# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-8):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True

def _objective(x, penalty_weight=5000.0):
    """
    Objective: minimize negative sum of radii + smooth penalty for violations.
    Uses a high penalty weight to strictly enforce constraints.
    """
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    radii = np.maximum(radii, 1e-5)
    
    penalty = 0.0
    
    # Boundary penalties (smooth quadratic)
    left_viol = np.maximum(0.0, radii - pos[:, 0])
    right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    bottom_viol = np.maximum(0.0, radii - pos[:, 1])
    top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    
    penalty += np.sum(left_viol**2) + np.sum(right_viol**2) + np.sum(bottom_viol**2) + np.sum(top_viol**2)
    
    # Pairwise overlap penalties (vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    # Only consider upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Overlap is when dist_sq < min_dist_sq
    # We penalize the square of the violation amount
    overlap = np.maximum(0.0, min_dist_sq - dist_sq)
    
    # Normalize by min_dist_sq to make penalty scale-invariant
    penalty += np.sum(overlap[mask] ** 2 / (4.0 * np.maximum(min_dist_sq[mask], 1e-8)))
    
    return -np.sum(radii) + penalty_weight * penalty

def _initial_grid():
    """Create a 5x5 grid of circles with the 26th placed in a corner gap."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.098
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            pos[idx, 0] = (i + 0.5) / 5.0
            pos[idx, 1] = (j + 0.5) / 5.0
            radii[idx] = r
            
    # Place 26th circle in a gap, e.g., center of a 2x2 block if possible, or just perturb
    # For simplicity, place it in a corner-ish spot
    pos[25, 0] = 0.5
    pos[25, 1] = 0.5
    radii[25] = 0.08
    
    return pos, radii

def _initial_hexagonal():
    """Create a hexagonal packing pattern."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.095
    y_spacing = r * np.sqrt(3)
    
    # 5 rows: 5, 4, 5, 4, 5 = 23 circles. Plus 3 more?
    # Let's do 5, 4, 5, 4, 4 = 22? No, N=26.
    # 6, 5, 6, 5, 4 = 26.
    
    rows = [6, 5, 6, 5, 4]
    idx = 0
    y_pos = 0.1
    for row_idx, count in enumerate(rows):
        y = 0.1 + row_idx * y_spacing
        x_start = (1.0 - (count - 1) * 2 * r) / 2.0
        for i in range(count):
            pos[idx, 0] = x_start + i * 2 * r
            pos[idx, 1] = y
            radii[idx] = r
            idx += 1
            
    return pos, radii

def _initial_corner_heavy():
    """
    Place large circles in corners and smaller ones in the center.
    This often yields a better sum of radii because corner circles can be larger.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 4 corners
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    for i, (cx, cy) in enumerate(corners):
        pos[i, 0] = cx
        pos[i, 1] = cy
        radii[i] = 0.12
        
    # 4 edges (midpoints)
    edges = [(0.5, 0.08), (0.5, 0.92), (0.08, 0.5), (0.92, 0.5)]
    for i, (cx, cy) in enumerate(edges):
        idx = 4 + i
        pos[idx, 0] = cx
        pos[idx, 1] = cy
        radii[idx] = 0.08
        
    # Remaining 18 circles in the center area
    # Arrange in a 4x4 grid in the center, plus 2 extra
    center_r = 0.06
    center_x_start = 0.25
    center_y_start = 0.25
    spacing = 0.15
    
    idx = 8
    for i in range(4):
        for j in range(4):
            pos[idx, 0] = center_x_start + i * spacing
            pos[idx, 1] = center_y_start + j * spacing
            radii[idx] = center_r
            idx += 1
            
    # 2 extra circles
    pos[idx, 0] = 0.5
    pos[idx, 1] = 0.5
    radii[idx] = 0.05
    idx += 1
    
    pos[idx, 0] = 0.5
    pos[idx, 1] = 0.3
    radii[idx] = 0.05
    
    return pos, radii

def _initial_random(seed=42):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.15, 0.85, size=(N, 2))
    radii = rng.uniform(0.05, 0.12, size=N)
    return pos, radii

def _post_process(pos, radii):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(100):
        valid = True
        
        for i in range(N):
            # Boundary check
            max_r_bound = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r_bound * 0.99:
                radii[i] = max_r_bound * 0.99
                valid = False
            if radii[i] < 1e-5:
                radii[i] = 1e-5
                valid = False
                
        # Pairwise check
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Shrink both
                    scale = dist / (min_dist + 1e-12) * 0.99
                    radii[i] = max(radii[i] * scale, 1e-5)
                    radii[j] = max(radii[j] * scale, 1e-5)
                    valid = False
                    
        if valid:
            break
            
    return pos, radii

def _force_anneal(pos, radii, deadline, n_iter=8000):
    """
    Vectorized Force Annealing.
    Uses repulsive forces to separate overlapping circles and attractive forces
    to grow radii. Much faster than loop-based grow-and-push.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    rng = np.random.RandomState(54321)
    
    # Learning rate schedule
    lr_start = 0.05
    lr_end = 0.0001
    decay = (lr_end / lr_start) ** (1.0 / n_iter)
    
    for i_iter in range(n_iter):
        if time.time() > deadline - 2:
            break
            
        lr = lr_start * (decay ** i_iter)
        
        # Compute pairwise forces
        # dx[i,j] = x_j - x_i
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(dist_sq + 1e-12)
        
        # Min distance required
        min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Overlap amount (positive if overlapping)
        overlap = min_dist - dist
        
        # Mask for valid pairs (i != j)
        mask = ~np.eye(N, dtype=bool)
        
        # Force: repulsive if overlapping, attractive (grow) otherwise?
        # We want to maximize radii, so we can treat radius growth as a force.
        # But standard approach: separate overlaps, then grow.
        # Let's use a combined potential.
        
        # Repulsive force on position
        # F_pos = sum_j (overlap_ij / dist_ij) * (r_i + r_j) * dir_ij
        # Simplified: push apart proportional to overlap
        
        # Gradient of penalty w.r.t pos
        # Penalty term: max(0, r_i + r_j - dist)^2
        # d/dx_i = -2 * max(0, r_i + r_j - dist) * (x_i - x_j) / dist
        
        grad_pos = np.zeros((N, 2))
        
        # Vectorized gradient calculation
        # overlap is (N, N)
        # We only care about where overlap > 0
        # Force on i from j is proportional to overlap * (pos_i - pos_j) / dist
        # Note: overlap[i,j] = r_i + r_j - dist_ij
        
        # Let's compute the force matrix
        # f_ij = overlap_ij / dist_ij
        f = np.zeros_like(dist)
        valid_overlap = (overlap > 0) & mask
        f[valid_overlap] = overlap[valid_overlap] / dist[valid_overlap]
        
        # Force on i is sum_j f_ij * (pos_i - pos_j)
        # pos_i - pos_j = -dx_ij
        # So force on i is -sum_j f_ij * dx_ij
        
        # f is symmetric? No, overlap is symmetric, dist is symmetric.
        # f_ij = f_ji.
        # Force on i: -sum_j f_ij * (pos_j - pos_i) = sum_j f_ij * (pos_i - pos_j)
        
        # Vectorized sum:
        # grad_pos_x[i] = sum_j f[i,j] * (pos[i,0] - pos[j,0]) = sum_j f[i,j] * (-dx[i,j])
        # Wait, dx[i,j] = pos[i,0] - pos[j,0].
        # So grad_pos_x[i] = -sum_j f[i,j] * dx[i,j]
        
        grad_pos[:, 0] = -np.sum(f * dx, axis=1)
        grad_pos[:, 1] = -np.sum(f * dy, axis=1)
        
        # Boundary forces
        # Push away from boundaries if close
        # r_i > x_i -> push right
        # r_i > 1-x_i -> push left
        
        # Gradient of boundary penalty
        # P = max(0, r-x)^2 + ...
        # dP/dx = -2*max(0, r-x)
        
        b_left = np.maximum(0, radii - pos[:, 0])
        b_right = np.maximum(0, pos[:, 0] - (1 - radii))
        b_bottom = np.maximum(0, radii - pos[:, 1])
        b_top = np.maximum(0, pos[:, 1] - (1 - radii))
        
        grad_pos[:, 0] += -2 * b_left + 2 * b_right
        grad_pos[:, 1] += -2 * b_bottom + 2 * b_top
        
        # Update positions
        pos -= lr * grad_pos
        
        # Clip to [0, 1]
        pos = np.clip(pos, 0, 1)
        
        # Update radii
        # We want to increase radii.
        # Limit for radius i is min(dist to boundary, min_j (dist_ij - r_j))
        # Let's just try to increase radii by a small amount and let the next step's repulsion handle conflicts?
        # Or compute max possible radius and move towards it.
        
        # Max radius based on boundaries
        max_r_bound = np.minimum(np.minimum(pos[:, 0], 1 - pos[:, 0]), np.minimum(pos[:, 1], 1 - pos[:, 1]))
        
        # Max radius based on neighbors
        # dist_ij - r_j
        # min over j != i
        # dist is (N, N)
        # limits_ij = dist_ij - r_j
        limits = dist - radii[np.newaxis, :]
        np.fill_diagonal(limits, np.inf)
        max_r_pair = np.min(limits, axis=1)
        
        max_r = np.minimum(max_r_bound, max_r_pair)
        max_r = np.maximum(max_r, 1e-5)
        
        # Move radii towards max_r
        # radii += lr_r * (max_r - radii)
        lr_r = lr * 2.0 # Grow a bit faster
        radii += lr_r * (max_r - radii)
        
        # Ensure radii don't shrink too much or go negative
        radii = np.maximum(radii, 1e-5)
        
        # Check validity and update best
        if _is_valid(pos, radii, tol=1e-6):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
        else:
            # If invalid, repair slightly
            pos_pp, radii_pp = _post_process(pos, radii)
            pos = pos_pp
            radii = radii_pp
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
                
    return best_pos, best_radii, best_score

def _polish(pos, radii, deadline):
    """High-precision local search using L-BFGS-B and SLSQP."""
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.5))
        
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # 1. L-BFGS-B
    if time.time() < deadline - 5:
        try:
            result = minimize(lambda x: _objective(x, 5000.0), x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if result.fun < _objective(x0, 5000.0) - 1e-15:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
    
    # 2. SLSQP (often better for tight constraints)
    if time.time() < deadline - 2:
        try:
            result = minimize(lambda x: _objective(x, 5000.0), x0, method='SLSQP', bounds=bounds,
                              options={'maxiter': 1000, 'ftol': 1e-14})
            
            if result.fun < _objective(x0, 5000.0) - 1e-15:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
            
    return best_pos, best_radii, best_score

def _optimize_from(pos, radii, deadline):
    """Run optimization from given initial configuration."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # 1. Force Anneal
    if time.time() < deadline - 15:
        fa_pos, fa_radii, fa_score = _force_anneal(best_pos, best_radii, deadline, n_iter=5000)
        if fa_score > best_score + 1e-12:
            best_score = fa_score
            best_pos = fa_pos.copy()
            best_radii = fa_radii.copy()
            
    # 2. Gradient Polish
    if time.time() < deadline - 5:
        pol_pos, pol_radii, pol_score = _polish(best_pos, best_radii, deadline)
        if pol_score > best_score + 1e-12:
            best_score = pol_score
            best_pos = pol_pos.copy()
            best_radii = pol_radii.copy()
            
    # 3. Second Force Anneal (finer)
    if time.time() < deadline - 5:
        fa_pos, fa_radii, fa_score = _force_anneal(best_pos, best_radii, deadline, n_iter=3000)
        if fa_score > best_score + 1e-12:
            best_score = fa_score
            best_pos = fa_pos.copy()
            best_radii = fa_radii.copy()
            
    # 4. Final Polish
    if time.time() < deadline - 1:
        pol_pos, pol_radii, pol_score = _polish(best_pos, best_radii, deadline)
        if pol_score > best_score + 1e-12:
            best_score = pol_score
            best_pos = pol_pos.copy()
            best_radii = pol_radii.copy()
            
    return best_pos, best_radii, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    init_configs = []
    
    # 1. Grid
    pos, radii = _initial_grid()
    init_configs.append((pos, radii))
    
    # 2. Hexagonal
    pos, radii = _initial_hexagonal()
    init_configs.append((pos, radii))
    
    # 3. Corner Heavy
    pos, radii = _initial_corner_heavy()
    init_configs.append((pos, radii))
    
    # 4. Random
    for seed in [42, 123]:
        pos, radii = _initial_random(seed)
        init_configs.append((pos, radii))
        
    # 5. Perturbed Grid
    rng = np.random.RandomState(99)
    pos, radii = _initial_grid()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # 6. Perturbed Hex
    pos, radii = _initial_hexagonal()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # 7. Perturbed Corner Heavy
    pos, radii = _initial_corner_heavy()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # Run optimization from each config
    # Order: Corner heavy is often good, then Hex, then Grid
    config_order = [2, 6, 1, 5, 0, 4, 3]
    
    for idx in config_order:
        if time.time() > deadline - 15:
            break
        if idx >= len(init_configs):
            continue
            
        pos, radii = init_configs[idx]
        pos_opt, radii_opt, score = _optimize_from(pos, radii, deadline)
        
        if score > best_score + 1e-12:
            best_score = score
            best_pos = pos_opt.copy()
            best_radii = radii_opt.copy()
            
    # Fallback
    if best_pos is None or best_score < 0:
        pos, radii = _initial_grid()
        best_pos, best_radii = _post_process(pos, radii)
        best_score = np.sum(best_radii)
        
    # Final high-precision polish on the best found
    if time.time() < deadline - 1:
        best_pos, best_radii, best_score = _polish(best_pos, best_radii, deadline)
        
    # Final post-processing to ensure validity
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
