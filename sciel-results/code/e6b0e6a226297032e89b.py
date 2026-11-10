import numpy as np
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    i_vals, j_vals = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_vals.flatten().astype(np.float64)
    j_flat = j_vals.flatten().astype(np.float64)
    
    # Baseline
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    base_x = keich[i_flat]
    base_y = keich[j_flat]
    baseline_positions = np.stack([base_x, base_y], axis=1)
    
    def estimate_volume(positions, n_samples=50000, seed=12345):
        x_k = positions[:, 0]
        y_k = positions[:, 1]
        
        X0 = np.min(x_k)
        X1 = np.max(x_k + (i_flat + 1) / N)
        Y0 = np.min(y_k)
        Y1 = np.max(y_k + (j_flat + 1) / N)
        
        box_area = (X1 - X0) * (Y1 - Y0)
        if box_area <= 1e-9:
            return float('inf')
            
        rng = np.random.default_rng(seed)
        a = rng.uniform(X0, X1, n_samples)
        b = rng.uniform(Y0, Y1, n_samples)
        c = rng.uniform(0, 1, n_samples)
        
        shift_x = c[:, None] * i_flat[None, :] / N
        shift_y = c[:, None] * j_flat[None, :] / N
        
        cond_x_low = a[:, None] >= (x_k[None, :] + shift_x)
        cond_x_high = a[:, None] <= (x_k[None, :] + shift_x + 1.0/N)
        cond_y_low = b[:, None] >= (y_k[None, :] + shift_y)
        cond_y_high = b[:, None] <= (y_k[None, :] + shift_y + 1.0/N)
        
        inside = cond_x_low & cond_x_high & cond_y_low & cond_y_high
        hit = np.any(inside, axis=1)
        
        fraction = np.sum(hit) / n_samples
        return fraction * box_area

    start_time = time.time()
    deadline = start_time + 590.0
    
    best_positions = baseline_positions
    best_vol = estimate_volume(best_positions, 50000, 12345)
    
    # Local search function
    def local_search(initial_positions, max_time):
        current_positions = initial_positions.copy()
        current_vol = estimate_volume(current_positions, 50000, 12345)
        step_size = 0.05
        no_improve = 0
        
        while time.time() < max_time:
            n_perturb = np.random.randint(1, 4)
            indices = np.random.choice(64, n_perturb, replace=False)
            
            new_positions = current_positions.copy()
            for idx in indices:
                new_positions[idx, 0] += np.random.normal(0, step_size)
                new_positions[idx, 1] += np.random.normal(0, step_size)
                
            new_vol = estimate_volume(new_positions, 50000, 12345)
            
            if new_vol < current_vol:
                current_positions = new_positions
                current_vol = new_vol
                no_improve = 0
                # step_size *= 0.9
            else:
                no_improve += 1
                if no_improve > 20:
                    # Reset step size or restart
                    step_size = 0.1
                    no_improve = 0
                    # Occasionally try a bigger jump
                    if np.random.rand() < 0.1:
                        n_perturb = np.random.randint(5, 10)
                        indices = np.random.choice(64, n_perturb, replace=False)
                        current_positions = current_positions.copy()
                        for idx in indices:
                            current_positions[idx, 0] += np.random.normal(0, step_size * 2)
                            current_positions[idx, 1] += np.random.normal(0, step_size * 2)
                        current_vol = estimate_volume(current_positions, 50000, 12345)
                        
        return current_positions, current_vol

    # Run several restarts
    restarts = 0
    while time.time() < deadline:
        # Time for one restart: 30 seconds
        restart_deadline = time.time() + 30.0
        if restart_deadline > deadline:
            restart_deadline = deadline
            
        if restarts == 0:
            start_pos = baseline_positions
        else:
            # Random start near baseline or fully random?
            # Fully random might be too far. Let's perturb baseline significantly.
            start_pos = baseline_positions + np.random.normal(0, 0.5, (64, 2))
            
        pos, vol = local_search(start_pos, restart_deadline)
        
        if vol < best_vol:
            best_positions = pos
            best_vol = vol
            
        restarts += 1
        
    return best_positions
