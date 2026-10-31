"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START

def _relax(pos, radii, max_iters=1000, step_scale=0.1):
    """
    Relax the positions of circles to resolve overlaps and boundary violations.
    Uses a simple force-directed method:
    - Repulsion between overlapping circles.
    - Attraction to corners/edges? No, just push out of boundaries.
    - We want to maximize radii, so we assume radii are fixed during relaxation.
    """
    pos = pos.copy()
    
    for _ in range(max_iters):
        forces = np.zeros_like(pos)
        
        # Boundary forces: Push circles inside the box
        # If x - r < 0, force right
        # If x + r > 1, force left
        # We use a strong linear force proportional to violation
        for i in range(N):
            r = radii[i]
            if pos[i, 0] < r:
                forces[i, 0] += (r - pos[i, 0]) * 10.0
            elif pos[i, 0] > 1.0 - r:
                forces[i, 0] -= (pos[i, 0] - (1.0 - r)) * 10.0
            else:
                # Weak attraction to center? No, we want to fill space.
                # Actually, for packing, we don't need to attract to center.
                # We just need to resolve overlaps.
                pass
            
            if pos[i, 1] < r:
                forces[i, 1] += (r - pos[i, 1]) * 10.0
            elif pos[i, 1] > 1.0 - r:
                forces[i, 1] -= (pos[i, 1] - (1.0 - r)) * 10.0
        
        # Overlap forces
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    overlap = min_dist - dist
                    # Force is proportional to overlap
                    # Normalize direction
                    fx = dx / dist
                    fy = dy / dist
                    # Push apart
                    forces[i, 0] += fx * overlap * 5.0
                    forces[i, 1] += fy * overlap * 5.0
                    forces[j, 0] -= fx * overlap * 5.0
                    forces[j, 1] -= fy * overlap * 5.0
        
        # Apply forces with damping
        # Adaptive step size to ensure stability
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force < 1e-6:
            break
            
        step = step_scale / max_force
        # Cap step size to prevent oscillations
        step = min(step, 0.1)
        
        pos += forces * step
        
        # Clamp to box to prevent escaping
        pos[:, 0] = np.clip(pos[:, 0], 0, 1)
        pos[:, 1] = np.clip(pos[:, 1], 0, 1)
        
    return pos

def _is_valid(pos, radii, tol=1e-8):
    """Check if configuration is strictly valid."""
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    dist = np.sqrt(dist_sq)
    if np.any(dist[mask] < min_dist[mask] - tol):
        return False
    return True

def _enforce_validity(pos, radii):
    """
    Greedily shrink radii to ensure validity.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for i in range(N):
        max_r = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
        if radii[i] > max_r:
            radii[i] = max(0.0, max_r)
            
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    if sum_r > 1e-10:
                        scale = dist / sum_r
                        radii[i] *= scale
                        radii[j] *= scale
                    else:
                        radii[i] = 0.0
                        radii[j] = 0.0
                    changed = True
        if not changed:
            break
            
    for i in range(N):
        max_r = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
        if radii[i] > max_r:
            radii[i] = max_r
            
    return pos, radii

def _hard_particle_optimization(start_pos, start_radii, time_limit):
    """
    Hard Particle Dynamics:
    1. Start with some configuration.
    2. Iteratively increase the radius of the smallest circle (or all circles).
    3. Relax positions to resolve overlaps.
    4. If relaxation fails (still overlaps), shrink the increased radius slightly.
    """
    pos = start_pos.copy()
    radii = start_radii.copy()
    
    # Ensure valid start
    pos, radii = _enforce_validity(pos, radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    
    start_time = time.time()
    
    # Strategy: Increase all radii by a small factor, or target the smallest?
    # Increasing all radii proportionally tends to keep the packing tight.
    # Let's try increasing the smallest circle first, as it has the most "slack".
    
    max_iters = 10000
    growth_factor = 1.001 # 0.1% growth per step
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Find the smallest circle
        min_idx = np.argmin(radii)
        
        # Try to grow it
        old_r = radii[min_idx]
        new_r = old_r * growth_factor
        
        # Limit by boundary
        max_r_wall = min(pos[min_idx,0], 1-pos[min_idx,0], pos[min_idx,1], 1-pos[min_idx,1])
        if new_r > max_r_wall:
            new_r = max_r_wall
            
        if new_r <= old_r:
            # Can't grow anymore due to walls? Maybe try another circle?
            # For now, if the smallest can't grow, we might be stuck or done.
            # Let's try growing the next smallest?
            # Or just break if we can't grow the smallest.
            # Actually, we should probably grow all circles proportionally in a global step?
            # Let's stick to growing the smallest for now. If it's blocked, try relaxing more?
            # If it's blocked by walls, we might need to move it.
            # Relaxation handles movement.
            # If new_r is not greater, we are at a local maximum for this circle.
            # Let's try a small random perturbation and re-relax to escape local min?
            # For simplicity, if we can't grow, we stop.
            break
        
        radii[min_idx] = new_r
        
        # Relax
        relaxed_pos = _relax(pos, radii, max_iters=200)
        
        # Check validity
        if _is_valid(relaxed_pos, radii):
            pos = relaxed_pos
            current_sum = np.sum(radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_pos = pos.copy()
                best_radii = radii.copy()
        else:
            # Relaxation failed to resolve overlaps with new radius
            # Shrink the radius back
            radii[min_idx] = old_r
            # Maybe try a smaller growth?
            # For now, break or try next circle.
            # Let's try to grow a different circle?
            # No, let's just stop if the smallest can't grow.
            # This is a bit simplistic. A better approach:
            # Try to grow ALL circles by a small factor.
            # Let's switch strategy: Global Growth.
            pass
            
    # Alternative Strategy: Global Growth
    # This is often more robust.
    # Reset to best found so far?
    # Let's run a second phase with global growth if the first phase finished early or didn't improve much.
    
    # Let's implement a global growth loop instead, as it's more standard for HPC.
    # We'll restart the optimization with a global growth strategy.
    
    pos = best_pos.copy()
    radii = best_radii.copy()
    
    # Ensure valid
    pos, radii = _enforce_validity(pos, radii)
    
    start_time = time.time()
    remaining_time = time_limit - (time.time() - start_time)
    
    # Global growth: multiply all radii by (1 + epsilon)
    # If valid, keep. If not, backtrack.
    
    epsilon = 0.005
    min_epsilon = 1e-5
    
    while time.time() - start_time < time_limit + 5: # Buffer
        if time.time() - start_time > time_limit:
            break
            
        # Grow all
        new_radii = radii * (1.0 + epsilon)
        
        # Check wall constraints
        valid_walls = True
        for i in range(N):
            max_r_wall = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
            if new_radii[i] > max_r_wall:
                valid_walls = False
                break
        
        if not valid_walls:
            # Can't grow globally due to walls.
            # Try to grow only those not hitting walls?
            # Or stop.
            # Let's try to relax first to see if we can move away from walls?
            # Relaxation pushes into box, doesn't pull away.
            # So if a circle is against a wall, it can't grow.
            # We should probably stop global growth.
            break
            
        # Relax
        relaxed_pos = _relax(pos, new_radii, max_iters=500)
        
        if _is_valid(relaxed_pos, new_radii):
            pos = relaxed_pos
            radii = new_radii
            # Success, maybe increase epsilon?
            # Keep epsilon constant for stability.
        else:
            # Failed, decrease epsilon
            epsilon *= 0.9
            if epsilon < min_epsilon:
                break
                
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_pos = None
    best_radii = None
    best_score = 0.0
    
    rng = np.random.RandomState(42)
    
    # Multi-start
    n_starts = 5
    
    for s in range(n_starts):
        if time.time() > deadline - 20:
            break
            
        # Initialization
        if s == 0:
            # Grid
            pos = np.zeros((N, 2))
            radii = np.zeros(N)
            k = 0
            for i in range(5):
                for j in range(5):
                    if k < N:
                        pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                        radii[k] = 0.09
                        k += 1
            if k < N:
                pos[k] = [0.5, 0.5]
                radii[k] = 0.01
        else:
            # Random
            pos = rng.uniform(0.1, 0.9, (N, 2))
            radii = rng.uniform(0.05, 0.15, N)
            pos, radii = _enforce_validity(pos, radii)
            
        # Optimize
        time_left = deadline - time.time() - 5
        if time_left <= 0:
            continue
            
        pos_opt, rad_opt = _hard_particle_optimization(pos, radii, time_limit=time_left)
        
        # Final polish/validity
        pos_final, rad_final = _enforce_validity(pos_opt, rad_opt)
        
        if _is_valid(pos_final, rad_final):
            score = np.sum(rad_final)
            if score > best_score:
                best_score = score
                best_pos = pos_final
                best_radii = rad_final
                
    # Fallback
    if best_pos is None:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.09
                    k += 1
        if k < N:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
        best_pos, best_radii = _enforce_validity(pos, radii)
        
    # Final strict validation
    best_pos, best_radii = _enforce_validity(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
