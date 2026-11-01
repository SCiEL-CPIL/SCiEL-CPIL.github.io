import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def compute_radius(c):
        if not np.all(np.isfinite(c)):
            return 0.0
        if np.any(c > 0.5) or np.any(c < -0.5):
            return 0.0
        wall_dists = 0.5 - np.abs(c)
        r_wall = np.min(wall_dists)
        
        # Vectorized pairwise distance
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
        
        return min(r_wall, r_pair)

    def get_energy(c):
        # E = sum(1/d^2) + sum(1/w^2)
        if np.any(c > 0.5) or np.any(c < -0.5):
            return 1e10
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1e-10) # Avoid div by 0
        # Only sum upper triangle to avoid double counting
        iu = np.triu_indices(N, k=1)
        pair_energy = np.sum(1.0 / dists[iu]**2)
        
        wall_dists = 0.5 - np.abs(c)
        w_min = np.min(wall_dists, axis=1)
        wall_energy = np.sum(1.0 / w_min**2)
        
        return pair_energy + wall_energy

    def incremental_delta_energy(c, idx, new_pos):
        # Calculate change in energy if c[idx] moves to new_pos
        # Old distances from idx to others
        others = np.delete(c, idx, axis=0)
        old_diff = c[idx] - others
        old_dists = np.linalg.norm(old_diff, axis=1)
        # Handle coincident points (shouldn't happen often, but safe)
        old_dists[old_dists < 1e-10] = 1e-10
        
        new_diff = new_pos - others
        new_dists = np.linalg.norm(new_diff, axis=1)
        new_dists[new_dists < 1e-10] = 1e-10
        
        delta_pair = np.sum(1.0 / new_dists**2 - 1.0 / old_dists**2)
        
        # Wall energy
        old_w = np.min(0.5 - np.abs(c[idx]))
        new_w = np.min(0.5 - np.abs(new_pos))
        
        if new_w < 1e-10:
            return 1e10 # Invalid move
        
        old_w_safe = max(old_w, 1e-10)
        new_w_safe = max(new_w, 1e-10)
        
        delta_wall = (1.0 / new_w_safe**2) - (1.0 / old_w_safe**2)
        
        return delta_pair + delta_wall

    # 1. Initialization
    np.random.seed(42)
    
    best_centers = None
    best_radius = 0.0
    
    # Grid Init
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.3, 0.3, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    init_centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    init_centers += np.random.normal(0, 0.01, init_centers.shape)
    init_centers = np.clip(init_centers, -0.49, 0.49)
    
    initializations = [init_centers]
    
    # Random Init
    for _ in range(3):
        rc = np.random.uniform(-0.45, 0.45, (N, 3))
        initializations.append(rc)
        
    # 2. Simulated Annealing Optimization
    for init in initializations:
        if time.time() > deadline - 10:
            break
            
        centers = init.copy()
        current_energy = get_energy(centers)
        
        # Annealing parameters
        T = 1.0
        step_size = 0.1
        alpha = 0.999 # Cooling factor
        min_T = 0.01
        
        # We run until time limit for this init, or T is very low
        # Budget: ~30s per init?
        init_start = time.time()
        
        while time.time() - init_start < 30.0 and T > min_T:
            if time.time() > deadline - 5:
                break
                
            idx = np.random.randint(N)
            # Gaussian move
            move = np.random.normal(0, step_size, 3)
            new_pos = centers[idx] + move
            
            # Bounds check
            if np.any(new_pos > 0.5) or np.any(new_pos < -0.5):
                # Reject or push back? Reject is simpler for MH
                continue
                
            delta_e = incremental_delta_energy(centers, idx, new_pos)
            
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
                centers[idx] = new_pos
                current_energy += delta_e
                
            T *= alpha
            # Slowly reduce step size
            step_size *= 0.9999
            if step_size < 0.001:
                step_size = 0.001

        # Evaluate this candidate
        r = compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()

    # 3. Local Polish (Hill Climbing)
    if best_centers is not None:
        current_centers = best_centers.copy()
        current_r = best_radius
        
        # Polish time budget: remaining time - 5s
        polish_start = time.time()
        polish_deadline = deadline - 5
        
        step_size = 0.01
        no_improve_count = 0
        max_no_improve = 1000
        
        while time.time() < polish_deadline and no_improve_count < max_no_improve:
            idx = np.random.randint(N)
            # Try multiple random directions for this sphere
            improved = False
            for _ in range(10):
                move = np.random.normal(0, step_size, 3)
                new_pos = current_centers[idx] + move
                
                if np.any(new_pos > 0.5) or np.any(new_pos < -0.5):
                    continue
                
                old_pos = current_centers[idx].copy()
                current_centers[idx] = new_pos
                
                new_r = compute_radius(current_centers)
                
                if new_r > current_r:
                    current_r = new_r
                    improved = True
                    if new_r > best_radius:
                        best_radius = new_r
                        best_centers = current_centers.copy()
                    break # Accept and move to next sphere
                else:
                    current_centers[idx] = old_pos
            
            if not improved:
                no_improve_count += 1
                # Reduce step size if stuck
                step_size *= 0.95
                if step_size < 0.0001:
                    break
            else:
                no_improve_count = 0
                # Keep step size or slightly reduce
                step_size *= 0.99

    return best_centers
