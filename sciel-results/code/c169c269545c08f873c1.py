import numpy as np

N = 50

def construct():
    # We will use a Simulated Annealing approach to maximize the minimum radius.
    # This allows escaping local minima which hill-climbing often gets stuck in.
    
    best_global_r = 0.0
    best_global_pos = None
    
    n_restarts = 5
    n_iterations = 20000
    
    for _ in range(n_restarts):
        # Initialization: Hexagonal grid based
        s = 0.14  # approximate spacing
        xs = []
        ys = []
        max_j = int(1.0 / (s * np.sqrt(3) / 2)) + 2
        max_i = int(1.0 / s) + 2
        for j in range(-max_j, max_j + 1):
            for i in range(-max_i, max_i + 1):
                x = i * s + (j % 2) * s / 2.0
                y = j * s * np.sqrt(3) / 2.0
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    xs.append(x)
                    ys.append(y)
        
        xs = np.array(xs)
        ys = np.array(ys)
        
        if len(xs) >= N:
            indices = np.random.choice(len(xs), N, replace=False)
            pos = np.column_stack((xs[indices], ys[indices]))
        else:
            # Fallback to random if grid is too sparse (unlikely with s=0.14)
            pos = np.random.uniform(-0.45, 0.45, (N, 2))
            
        # Add small noise to break symmetry
        pos += np.random.normal(0, 0.01, pos.shape)
        pos = np.clip(pos, -0.5, 0.5)
        
        current_r = calculate_radius(pos)
        best_local_r = current_r
        best_local_pos = pos.copy()
        
        # Simulated Annealing Parameters
        T = 0.01 # Initial temperature
        T_min = 0.0001
        cooling_rate = 0.9999
        step_size = 0.05
        
        for it in range(n_iterations):
            # Pick a random point
            idx = np.random.randint(N)
            old_pos = pos[idx].copy()
            
            # Generate new position
            step = np.random.normal(0, step_size, 2)
            new_pos = old_pos + step
            new_pos = np.clip(new_pos, -0.5, 0.5)
            
            pos[idx] = new_pos
            new_r = calculate_radius(pos)
            
            delta_r = new_r - current_r
            
            if delta_r > 0:
                # Accept improvement
                current_r = new_r
                if new_r > best_local_r:
                    best_local_r = new_r
                    best_local_pos = pos.copy()
            else:
                # Accept worse solution with probability exp(delta_r / T)
                # delta_r is negative, so exp(negative) < 1
                if np.random.rand() < np.exp(delta_r / T):
                    current_r = new_r
                else:
                    # Revert
                    pos[idx] = old_pos
            
            # Cool down
            T = max(T_min, T * cooling_rate)
            # Slightly decrease step size to allow fine-tuning
            step_size = max(0.001, step_size * 0.9999)
            
        if best_local_r > best_global_r:
            best_global_r = best_local_r
            best_global_pos = best_local_pos.copy()
            
    return best_global_pos

def calculate_radius(pos):
    # Calculate min pairwise distance
    # N=50, so 50x50 distance matrix is fine
    dists = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=-1)
    # Exclude self-distances
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists)
    
    # Calculate min boundary distance
    # min_i min_k (1/2 - |c_ik|)
    # This is 0.5 - max(|c_ik|)
    max_abs = np.max(np.abs(pos))
    boundary_dist = 0.5 - max_abs
    
    return min(min_pairwise / 2.0, boundary_dist)
