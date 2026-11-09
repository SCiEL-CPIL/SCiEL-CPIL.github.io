"""Physics-based soft-body simulation for circle packing in perimeter-4 rectangle."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np

def _compute_energy(pos, radii, W, H):
    """
    Compute energy for Simulated Annealing.
    Uses soft repulsive potentials.
    """
    n = len(radii)
    energy = 0.0
    k_rep = 50.0  # Repulsion strength
    k_wall = 50.0 # Wall strength
    k_growth = 1.0 # Growth incentive (maximize sum radii)
    
    # Inter-circle repulsion (LJ-like soft repulsion)
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(dist_sq) + 1e-8
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                # Soft potential: quadratic overlap penalty
                overlap = min_dist - dist
                energy += k_rep * overlap * overlap
    
    # Wall repulsion
    for i in range(n):
        r = radii[i]
        x, y = pos[i]
        # Left
        if x < r:
            d = r - x
            energy += k_wall * d * d
        # Right
        if x > W - r:
            d = x - (W - r)
            energy += k_wall * d * d
        # Bottom
        if y < r:
            d = r - y
            energy += k_wall * d * d
        # Top
        if y > H - r:
            d = y - (H - r)
            energy += k_wall * d * d
            
    # Incentive to grow radii
    energy -= k_growth * np.sum(radii)
    
    return energy

def construct():
    # Perimeter = 2*(W + H) = 4 => W + H = 2
    best_result = None
    best_sum = 0.0
    
    # Aspect ratios to try
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.5]
    
    # Simulated Annealing parameters
    T_start = 0.1
    T_end = 1e-6
    alpha = 0.995
    max_iters = 2000
    step_scale = 0.05
    
    for aspect in aspect_ratios:
        W = 2.0 * aspect / (1.0 + aspect)
        H = 2.0 / (1.0 + aspect)
        
        # Random initialization to avoid grid traps
        # Ensure initial positions are somewhat spread out
        pos = np.random.uniform(0.1, 0.9, (N, 2)) * np.array([W, H])
        # Initial radii: small, allowing growth
        radii = np.full(N, 0.05)
        
        # Ensure initial state is not catastrophically bad (optional, but SA handles it)
        # Clamp initial positions to be inside
        pos[:, 0] = np.clip(pos[:, 0], radii, W - radii)
        pos[:, 1] = np.clip(pos[:, 1], radii, H - radii)
        
        current_energy = _compute_energy(pos, radii, W, H)
        best_local_energy = current_energy
        best_local_pos = pos.copy()
        best_local_radii = radii.copy()
        
        T = T_start
        
        for it in range(max_iters):
            # Generate a new candidate state
            new_pos = pos.copy()
            new_radii = radii.copy()
            
            # Perturb positions
            new_pos += np.random.normal(0, step_scale * T, (N, 2))
            
            # Perturb radii (positive only)
            new_radii += np.random.normal(0, step_scale * T * 0.5, N)
            new_radii = np.maximum(new_radii, 0.01) # Min radius
            
            # Clamp positions to stay roughly inside (helps stability)
            new_pos[:, 0] = np.clip(new_pos[:, 0], 0, W)
            new_pos[:, 1] = np.clip(new_pos[:, 1], 0, H)
            
            new_energy = _compute_energy(new_pos, new_radii, W, H)
            
            delta_E = new_energy - current_energy
            
            # Metropolis criterion
            if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
                pos = new_pos
                radii = new_radii
                current_energy = new_energy
                
                if current_energy < best_local_energy:
                    best_local_energy = current_energy
                    best_local_pos = pos.copy()
                    best_local_radii = radii.copy()
            
            T *= alpha
            
            # Break if temperature is too low
            if T < T_end:
                break
        
        # Final refinement: use the best state found during SA
        pos_opt = best_local_pos
        radii_opt = best_local_radii
        
        # Post-processing: Repair violations
        # 1. Ensure inside bounds
        for i in range(N):
            radii_opt[i] = min(radii_opt[i], 
                               pos_opt[i, 0], 
                               W - pos_opt[i, 0], 
                               pos_opt[i, 1], 
                               H - pos_opt[i, 1])
            radii_opt[i] = max(radii_opt[i], 1e-6)
        
        # 2. Resolve overlaps by shrinking
        for _ in range(50):
            resolved = True
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.linalg.norm(pos_opt[i] - pos_opt[j])
                    min_dist = radii_opt[i] + radii_opt[j]
                    if dist < min_dist - 1e-9:
                        resolved = False
                        shrink = (min_dist - dist) / 2.0 + 1e-4
                        radii_opt[i] = max(radii_opt[i] - shrink, 1e-6)
                        radii_opt[j] = max(radii_opt[j] - shrink, 1e-6)
            if resolved:
                break
        
        # Final check and sum
        valid = True
        for i in range(N):
            if pos_opt[i, 0] < radii_opt[i] - 1e-9 or pos_opt[i, 0] > W - radii_opt[i] + 1e-9:
                valid = False
                break
            if pos_opt[i, 1] < radii_opt[i] - 1e-9 or pos_opt[i, 1] > H - radii_opt[i] + 1e-9:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.linalg.norm(pos_opt[i] - pos_opt[j])
                    if dist < radii_opt[i] + radii_opt[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            total = np.sum(radii_opt)
            if total > best_sum:
                best_sum = total
                best_result = (pos_opt, radii_opt, W, H)
    
    if best_result is None:
        # Fallback
        W, H = 1.0, 1.0
        cells = []
        for i in range(N):
            row = i // 5
            col = i % 5
            x = (col + 0.5) * W / 5
            y = (row + 0.5) * H / 5
            r = 0.1
            cells.append([x, y, r])
        return np.array(cells)
    
    pos_opt, radii_opt, W, H = best_result
    result = np.column_stack([pos_opt, radii_opt])
    return result
# EVOLVE-BLOCK-END
