import numpy as np
import time

N = 50

def compute_radius(positions):
    """
    Compute the maximum radius for the given positions.
    Returns radius. If invalid, returns a large negative value to act as penalty.
    """
    # positions shape (N, 2)
    
    # Boundary distances
    max_abs = np.max(np.abs(positions), axis=1)
    bound_dists = 0.5 - max_abs
    
    # If any point is outside, return penalty
    if np.any(bound_dists < 0):
        return -1.0 - np.max(-bound_dists)
        
    min_bound = np.min(bound_dists)
    
    # Pairwise distance constraint
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    
    # Ignore self-distances
    np.fill_diagonal(dists_sq, np.inf)
    
    min_dist_sq = np.min(dists_sq)
    min_dist = np.sqrt(min_dist_sq)
    
    r = min(min_bound, min_dist / 2.0)
    
    return r

def soft_min(values, k=1000.0):
    """
    Soft minimum of an array of values using log-sum-exp trick.
    This provides a smooth surrogate for the min function.
    """
    # To avoid overflow in exp, shift by max
    max_val = np.max(values)
    # We want min, so we use -max of -values
    # min(x) = -max(-x)
    # soft_min(x) = -soft_max(-x)
    # soft_max(x) = log(sum(exp(x - max_x))) + max_x
    neg_vals = -values
    max_neg = np.max(neg_vals)
    exp_vals = np.exp(neg_vals - max_neg)
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        return max_neg
    soft_max_neg = np.log(sum_exp) + max_neg
    return -soft_max_neg

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    # 1. Initialization: Hexagonal Grid
    s = 1.0
    points = []
    for j in range(-15, 16):
        for i in range(-15, 16):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3.0) / 2.0
            points.append([x, y])
            
    points = np.array(points)
    dists_to_center = np.sum(points**2, axis=1)
    indices = np.argsort(dists_to_center)[:N]
    positions = points[indices].copy()
    
    max_abs = np.max(np.abs(positions))
    if max_abs > 0:
        scale = 0.5 / max_abs
        positions = positions * scale
        
    r_init_val = compute_radius(positions)
    noise_std = max(0.001, r_init_val * 0.01)
    positions += np.random.normal(0, noise_std, size=positions.shape)
    positions = np.clip(positions, -0.4999, 0.4999)
    
    current_radius = compute_radius(positions)
    best_radius = current_radius
    best_positions = positions.copy()
    
    # 2. Simulated Annealing
    T = 0.001
    T_end = 1e-7
    alpha = 0.999
    step_size = 0.01
    min_step_size = 0.0001
    max_step_size = 0.05
    
    accept_count = 0
    total_count = 0
    
    sa_deadline = start_time + 95.0
    
    while time.time() < sa_deadline:
        for _ in range(100):
            if time.time() >= sa_deadline:
                break
                
            total_count += 1
            idx = np.random.randint(N)
            
            if total_count % 100 == 0:
                accept_rate = accept_count / max(1, total_count)
                if accept_rate < 0.2:
                    step_size *= 1.2
                elif accept_rate > 0.8:
                    step_size *= 0.8
                step_size = np.clip(step_size, min_step_size, max_step_size)
                accept_count = 0
                total_count = 0
                
            delta = np.random.normal(0, step_size, size=2)
            new_pos = positions[idx].copy() + delta
            
            old_pos = positions[idx].copy()
            positions[idx] = new_pos
            new_radius = compute_radius(positions)
            
            if new_radius > current_radius:
                current_radius = new_radius
                accept_count += 1
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_positions = positions.copy()
            else:
                delta_r = new_radius - current_radius
                if delta_r > -T * 10:
                    prob = np.exp(delta_r / T)
                    if np.random.random() < prob:
                        current_radius = new_radius
                        accept_count += 1
                    else:
                        positions[idx] = old_pos
                else:
                    positions[idx] = old_pos
                    
        T *= alpha
        if T < T_end:
            T = T_end
            
    # 3. Polish Phase: Gradient-based local search using soft-min surrogate
    def polish(pos_in):
        pos = pos_in.copy()
        
        # Objective: Maximize r
        # r = min(bound_dists, dists/2)
        # Surrogate: -soft_min(bound_dists, dists/2)
        # We want to minimize -r, so maximize soft_min
        
        # To compute gradients efficiently, we can use finite differences
        # or analytic gradients if we derive them.
        # Given the complexity of analytic gradients for soft_min of min(bound, dist),
        # we use a simple gradient ascent with finite differences.
        
        # Let's define the objective function to maximize
        def obj_func(p):
            # p is flat array of size 2N
            pp = p.reshape(N, 2)
            
            # Boundary distances
            max_abs = np.max(np.abs(pp), axis=1)
            bound_dists = 0.5 - max_abs
            
            # If any point is outside, return a large penalty
            if np.any(bound_dists < 0):
                return -100.0
            
            # Pairwise distances
            diff = pp[:, np.newaxis, :] - pp[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=-1)
            np.fill_diagonal(dists_sq, np.inf)
            dists = np.sqrt(dists_sq)
            
            # We want min of (bound_dists, dists/2)
            # Combine them into a single array for soft_min
            # Note: dists is NxN, we only need upper triangle
            # But soft_min works on any array
            # We can just take the min of the whole matrix (excluding diagonal)
            # and the bound_dists
            
            # To be efficient, we can just compute the actual min for the "hard" part
            # but use soft_min for the surrogate.
            # However, for gradient ascent, we need the gradient of the soft_min.
            
            # Let's just use the actual compute_radius for evaluation but 
            # for the polish step, we will do a simple local search 
            # that moves points to increase the minimum distance.
            
            # Actually, let's stick to the constraint-based approach but refine it.
            # The previous polish was a bit crude.
            # Let's try a "Repulsion" method:
            # Each point feels a repulsive force from other points if they are too close.
            # And a centripetal force if they are too close to the boundary.
            
            return 0.0 # Placeholder
            
        # Let's implement a proper gradient-free local search that is more effective
        # than the simple SA. We will use a "push" method.
        
        current_r = compute_radius(pos)
        
        for _ in range(500):
            if time.time() > deadline:
                break
                
            # Compute forces
            forces = np.zeros_like(pos)
            
            # Boundary forces
            max_abs = np.max(np.abs(pos), axis=1)
            bound_dists = 0.5 - max_abs
            
            # If a point is close to boundary, push it towards center
            for i in range(N):
                if bound_dists[i] < current_r * 1.01:
                    # Push towards center
                    forces[i] -= pos[i] * 0.01
                    
            # Pairwise repulsion
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=-1)
            np.fill_diagonal(dists_sq, np.inf)
            dists = np.sqrt(dists_sq)
            
            # We want to increase the minimum distance
            # So we push pairs that are closer than 2*r apart
            threshold = 2.0 * current_r * 1.01
            
            for i in range(N):
                for j in range(i+1, N):
                    if dists[i, j] < threshold:
                        # Push apart
                        d = dists[i, j]
                        if d < 1e-9:
                            forces[i] += np.random.normal(0, 0.001, 2)
                            forces[j] -= np.random.normal(0, 0.001, 2)
                        else:
                            dir_ij = (pos[i] - pos[j]) / d
                            # Strength proportional to violation
                            violation = threshold - d
                            force_mag = 0.01 * violation
                            forces[i] += dir_ij * force_mag
                            forces[j] -= dir_ij * force_mag
                            
            # Apply forces with a small step
            pos += forces * 0.1
            pos = np.clip(pos, -0.4999, 0.4999)
            
            new_r = compute_radius(pos)
            if new_r > current_r:
                current_r = new_r
            else:
                # If it got worse, revert? Or just keep going?
                # Let's revert if it's significantly worse
                if new_r < current_r - 1e-5:
                    pos -= forces * 0.1
                    pos = np.clip(pos, -0.4999, 0.4999)
                    
        return pos, current_r

    if time.time() < deadline - 2.0:
        polished_pos, polished_r = polish(best_positions)
        if polished_r > best_radius:
            best_positions = polished_pos
            best_radius = polished_r
            
    return best_positions
