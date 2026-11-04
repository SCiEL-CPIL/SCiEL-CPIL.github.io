<<<<<<< SEARCH
def _discrete_best_fit(pos, radii, n, deadline, step_sizes=None):
    """
    Discrete best-fit local search:
    For each circle, try moving it to various candidate positions
    and keep the one that maximizes the sum of radii.
    """
    if step_sizes is None:
        step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    
    for step_size in step_sizes:
        if time.time() > deadline:
            break
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            for i in range(n):
                orig_x, orig_y = best_pos[i]
                best_local_sum = best_sum
                best_local_pos = best_pos.copy()
                best_local_radii = best_radii.copy()
                
                # Generate candidate positions
                candidates = []
                # 8 neighbors + center
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        if dx == 0 and dy == 0:
                            continue
                        new_x = orig_x + dx
                        new_y = orig_y + dy
                        # Keep in bounds
                        if 0 <= new_x <= 1 and 0 <= new_y <= 1:
                            candidates.append((new_x, new_y))
                
                # Also try diagonal moves
                diag = step_size * np.sqrt(2) / 2
                for dx in [-diag, diag]:
                    for dy in [-diag, diag]:
                        new_x = orig_x + dx
                        new_y = orig_y + dy
                        if 0 <= new_x <= 1 and 0 <= new_y <= 1:
                            candidates.append((new_x, new_y))
                
                # Try each candidate
                for cx, cy in candidates:
                    trial_pos = best_local_pos.copy()
                    trial_pos[i] = [cx, cy]
                    trial_radii = _compute_radii(trial_pos, n)
                    trial_sum = np.sum(trial_radii)
                    
                    if trial_sum > best_local_sum + 1e-8:
                        best_local_sum = trial_sum
                        best_local_pos = trial_pos
                        best_local_radii = trial_radii
                
                if best_local_sum > best_sum + 1e-8:
                    best_pos = best_local_pos
                    best_radii = best_local_radii
                    best_sum = best_local_sum
                    improved = True
    
    return best_pos, best_radii
=======
def _momentum_local_search(pos, radii, n, deadline, max_iters=2000):
    """
    Momentum-based local search: treat circles as particles with forces
    from overlap constraints, using velocity Verlet with damping.
    """
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    
    pos_curr = best_pos.copy()
    vel = np.zeros((n, 2))
    damping = 0.85
    dt = 0.01
    force_scale = 5.0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute forces from overlaps
        dx = pos_curr[:, 0:1] - pos_curr[0:1, 0]
        dy = pos_curr[:, 1:2] - pos_curr[0:2, 1]
        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
        np.fill_diagonal(dist, np.inf)
        
        excess = radii[:, 0:1] + radii[0:1, :] - dist
        excess = np.maximum(excess, 0)
        np.fill_diagonal(excess, 0)
        
        # Force: push apart overlapping circles
        dir_x = dx / (dist + 1e-12)
        dir_y = dy / (dist + 1e-12)
        force = force_scale * excess
        fx = np.sum(force * dir_x, axis=1)
        fy = np.sum(force * dir_y, axis=1)
        
        # Boundary forces: push circles inward if too close to boundary
        for i in range(n):
            if pos_curr[i, 0] < radii[i]:
                fx[i] += force_scale * (radii[i] - pos_curr[i, 0])
            if pos_curr[i, 0] > 1 - radii[i]:
                fx[i] -= force_scale * (pos_curr[i, 0] - (1 - radii[i]))
            if pos_curr[i, 1] < radii[i]:
                fy[i] += force_scale * (radii[i] - pos_curr[i, 1])
            if pos_curr[i, 1] > 1 - radii[i]:
                fy[i] -= force_scale * (pos_curr[i, 1] - (1 - radii[i]))
        
        # Update velocity and position
        vel = damping * vel + dt * np.column_stack([fx, fy])
        pos_curr = pos_curr + dt * vel
        
        # Clamp to bounds
        pos_curr[:, 0] = np.clip(pos_curr[:, 0], 0.01, 0.99)
        pos_curr[:, 1] = np.clip(pos_curr[:, 1], 0.01, 0.99)
        
        # Recompute radii and check improvement
        trial_radii = _compute_radii(pos_curr, n)
        trial_sum = np.sum(trial_radii)
        
        if trial_sum > best_sum + 1e-10:
            best_sum = trial_sum
            best_pos = pos_curr.copy()
            best_radii = trial_radii.copy()
        else:
            # If no improvement for a while, reduce step size
            if it % 100 == 99 and it > 0:
                dt *= 0.7
                damping *= 0.98
                if dt < 0.001:
                    break
    
    return best_pos, best_radii

def _discrete_best_fit(pos, radii, n, deadline, step_sizes=None):
    """
    Discrete best-fit local search:
    For each circle, try moving it to various candidate positions
    and keep the one that maximizes the sum of radii.
    """
    if step_sizes is None:
        step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    
    for step_size in step_sizes:
        if time.time() > deadline:
            break
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            for i in range(n):
                orig_x, orig_y = best_pos[i]
                best_local_sum = best_sum
                best_local_pos = best_pos.copy()
                best_local_radii = best_radii.copy()
                
                # Generate candidate positions
                candidates = []
                # 8 neighbors
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        if dx == 0 and dy == 0:
                            continue
                        new_x = orig_x + dx
                        new_y = orig_y + dy
                        if 0 <= new_x <= 1 and 0 <= new_y <= 1:
                            candidates.append((new_x, new_y))
                
                # Diagonal moves
                diag = step_size * np.sqrt(2) / 2
                for dx in [-diag, diag]:
                    for dy in [-diag, diag]:
                        new_x = orig_x + dx
                        new_y = orig_y + dy
                        if 0 <= new_x <= 1 and 0 <= new_y <= 1:
                            candidates.append((new_x, new_y))
                
                # Try each candidate
                for cx, cy in candidates:
                    trial_pos = best_local_pos.copy()
                    trial_pos[i] = [cx, cy]
                    trial_radii = _compute_radii(trial_pos, n)
                    trial_sum = np.sum(trial_radii)
                    
                    if trial_sum > best_local_sum + 1e-8:
                        best_local_sum = trial_sum
                        best_local_pos = trial_pos
                        best_local_radii = trial_radii
                
                if best_local_sum > best_sum + 1e-8:
                    best_pos = best_local_pos
                    best_radii = best_local_radii
                    best_sum = best_local_sum
                    improved = True
    
    return best_pos, best_radii
>>>>>>> REPLACE

