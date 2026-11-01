import numpy as np
import time
import random

N = 200

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # Helper: Compute the "badness" of the configuration.
    # We want to maximize r. 
    # r = min(min_pair_dist/2, min_wall_dist)
    # Equivalently, minimize max( 2*r_target - pair_dist, 2*r_target - wall_dist ) 
    # But since we don't know r_target, we can minimize the sum of violations 
    # or maximize the minimum constraint.
    # For Simulated Annealing, a smooth objective is better.
    # Let's minimize: sum( max(0, 2*r_current - d_ij)^2 ) + sum( max(0, 2*r_current - w_i)^2 )
    # However, r_current is not fixed. 
    # Alternative: Minimize the "energy" E = sum_{i<j} 1/d_ij + sum_i 1/w_i.
    # Minimizing this energy pushes points apart and away from walls.
    # As E -> 0, d_ij -> inf (impossible) and w_i -> inf (impossible).
    # Actually, for packing, we want to MAXIMIZE the minimum distance.
    # A common objective for packing is to minimize the sum of inverse distances.
    # E = sum_{i<j} (1/d_ij) + sum_i (1/w_i)
    # This is differentiable and smooth.
    
    def compute_energy(pos):
        # pos: (N, 2)
        # Wall distances
        # w_i = 0.5 - max(|x_i|, |y_i|)
        # To avoid division by zero, clip w_i to a small epsilon
        abs_pos = np.abs(pos)
        max_abs = np.max(abs_pos, axis=1)
        w = 0.5 - max_abs
        w = np.clip(w, 1e-6, None)
        
        # Pairwise distances
        # Using broadcasting for (N, N, 2)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        d_sq = np.sum(diff * diff, axis=2)
        # Avoid division by zero on diagonal
        np.fill_diagonal(d_sq, np.inf)
        d = np.sqrt(d_sq)
        d = np.clip(d, 1e-6, None)
        
        # Energy terms
        # We only sum over i < j for pairs to avoid double counting and diagonal
        # But for vectorization, summing all i!=j is fine, just divide by 2 or accept factor
        pair_energy = np.sum(1.0 / d)
        wall_energy = np.sum(1.0 / w)
        
        return pair_energy + wall_energy

    def compute_radius(pos):
        # Exact radius calculation for scoring
        abs_pos = np.abs(pos)
        max_abs = np.max(abs_pos, axis=1)
        w = 0.5 - max_abs
        min_w = np.min(w)
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        d_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d_sq, np.inf)
        d = np.sqrt(d_sq)
        min_d = np.min(d)
        
        return min(min_d / 2.0, min_w)

    def init_grid():
        # Hexagonal grid initialization is usually better for packing
        # Approximate number of rows/cols
        # Area per circle ~ 1/200. Hex packing density ~ 0.9069
        # s ~ sqrt(1/200 / (0.9069 * sqrt(3)/2))
        # Let's just generate a dense hex grid and pick best N
        points = []
        # Generate enough points
        for row in range(30):
            for col in range(30):
                x = col * 0.06 + (row % 2) * 0.03
                y = row * 0.052
                points.append([x - 0.5, y - 0.5])
        points = np.array(points)
        # Filter points inside square
        mask = np.all(np.abs(points) < 0.5, axis=1)
        points = points[mask]
        
        # Randomly select N points
        if len(points) >= N:
            idx = np.random.choice(len(points), N, replace=False)
            pos = points[idx].copy()
        else:
            # Fallback to random
            pos = np.random.uniform(-0.49, 0.49, (N, 2))
            
        # Add small noise to break symmetry
        pos += np.random.normal(0, 0.001, pos.shape)
        return pos

    def init_random():
        return np.random.uniform(-0.49, 0.49, (N, 2))

    best_pos = None
    best_r = 0.0
    
    # Try multiple initializations
    initializers = [init_grid, init_random]
    
    for init_idx, init_func in enumerate(initializers):
        if time.time() - start_time > time_limit - 10:
            break
            
        pos = init_func()
        
        # Simulated Annealing
        # Objective: Minimize Energy
        current_energy = compute_energy(pos)
        best_energy = current_energy
        best_sa_pos = pos.copy()
        
        # Parameters
        T = 0.01  # Initial temperature
        T_min = 1e-6
        cooling = 0.995
        max_iter = 20000
        step_size = 0.05
        
        for i in range(max_iter):
            if time.time() - start_time > time_limit - 5:
                break
                
            # Propose move: pick random point, move random direction
            idx = np.random.randint(0, N)
            angle = np.random.uniform(0, 2 * np.pi)
            step = np.random.normal(0, step_size)
            
            new_pos = pos.copy()
            new_pos[idx, 0] += np.cos(angle) * step
            new_pos[idx, 1] += np.sin(angle) * step
            
            # Ensure inside square (or penalize heavily)
            # If it goes outside, energy will be huge due to 1/w term
            # But we can clip to avoid inf/nan issues in compute_energy if w becomes negative
            # compute_energy clips w to 1e-6, so it's safe.
            
            new_energy = compute_energy(new_pos)
            
            delta_E = new_energy - current_energy
            
            if delta_E < 0 or np.random.uniform(0, 1) < np.exp(-delta_E / T):
                pos = new_pos
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_sa_pos = pos.copy()
            
            T *= cooling
            if T < T_min:
                break
        
        # Local Refinement on the best SA result
        # Use a simple gradient-free optimization on the radius function?
        # Or just continue with smaller steps in SA?
        # Let's do a few more iterations with very small step size (annealing tail)
        
        T = 1e-4
        step_size = 0.005
        for i in range(5000):
            if time.time() - start_time > time_limit - 2:
                break
                
            idx = np.random.randint(0, N)
            angle = np.random.uniform(0, 2 * np.pi)
            step = np.random.normal(0, step_size)
            
            new_pos = best_sa_pos.copy()
            new_pos[idx, 0] += np.cos(angle) * step
            new_pos[idx, 1] += np.sin(angle) * step
            
            new_energy = compute_energy(new_pos)
            current_energy = compute_energy(best_sa_pos)
            
            delta_E = new_energy - current_energy
            
            if delta_E < 0 or np.random.uniform(0, 1) < np.exp(-delta_E / T):
                best_sa_pos = new_pos
            
            T *= 0.99
            if T < T_min:
                break

        # Calculate final radius
        r = compute_radius(best_sa_pos)
        
        if r > best_r:
            best_r = r
            best_pos = best_sa_pos.copy()
            
        # If we have time, try another restart with different init
        if time.time() - start_time < time_limit - 30:
             continue
        else:
             break

    # Final validation
    if best_pos is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pos = grid[:N]
        
    # Clip to ensure validity (should be inside if energy was minimized, but just in case)
    # Note: Clipping might violate non-overlap if it was close to edge.
    # But compute_energy penalizes edge proximity, so points should be away from edges.
    best_pos = np.clip(best_pos, -0.5, 0.5)
    
    return best_pos

# Keep the function name exactly as construct
