import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    time_limit = 115
    
    # Precompute pair indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    best_sum_r = -np.inf
    best_xs = None
    best_ys = None
    best_rs = None
    
    # Multiple restarts with different initializations
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Initialization
        if restart == 0:
            # Grid
            side = 5
            xs = []
            ys = []
            for i in range(side):
                for j in range(side):
                    if len(xs) < N:
                        xs.append((i + 0.5) / side)
                        ys.append((j + 0.5) / side)
            # Pad if needed (5x5=25, need 1 more)
            if len(xs) < N:
                xs.append(0.5)
                ys.append(0.5)
            xs = np.array(xs[:N])
            ys = np.array(ys[:N])
        else:
            # Random initialization
            xs = np.random.uniform(0.1, 0.9, N)
            ys = np.random.uniform(0.1, 0.9, N)
            
        rs = np.full(N, 0.02)
        
        delta_r = 0.002
        lr = 0.1
        max_iter_per_step = 500
        
        current_sum_r = np.sum(rs)
        
        # Main loop for this restart
        while time.time() - start_time < time_limit - 2:
            new_rs = rs + delta_r
            
            # Optimize positions
            cur_xs = xs.copy()
            cur_ys = ys.copy()
            
            # Gradient Descent
            for it in range(max_iter_per_step):
                dx = cur_xs[i_idx] - cur_xs[j_idx]
                dy = cur_ys[i_idx] - cur_ys[j_idx]
                d = np.sqrt(dx**2 + dy**2)
                d = np.maximum(d, 1e-6)
                
                r_sum = new_rs[i_idx] + new_rs[j_idx]
                overlap = np.maximum(0, r_sum - d)
                
                energy = np.sum(overlap**2)
                
                if energy < 1e-9:
                    break
                    
                coeff = 2.0 * overlap / d
                
                grad_x = np.zeros(N)
                grad_y = np.zeros(N)
                
                np.add.at(grad_x, i_idx, -coeff * dx)
                np.add.at(grad_y, i_idx, -coeff * dy)
                
                np.add.at(grad_x, j_idx, coeff * dx)
                np.add.at(grad_y, j_idx, coeff * dy)
                
                cur_xs -= lr * grad_x
                cur_ys -= lr * grad_y
                
                # Boundary constraints
                cur_xs = np.clip(cur_xs, new_rs, 1 - new_rs)
                cur_ys = np.clip(cur_ys, new_rs, 1 - new_rs)
                
            # Final check
            dx = cur_xs[i_idx] - cur_xs[j_idx]
            dy = cur_ys[i_idx] - cur_ys[j_idx]
            d = np.sqrt(dx**2 + dy**2)
            d = np.maximum(d, 1e-6)
            r_sum = new_rs[i_idx] + new_rs[j_idx]
            overlap = np.maximum(0, r_sum - d)
            final_energy = np.sum(overlap**2)
            
            if final_energy < 1e-7:
                # Accept
                xs = cur_xs
                ys = cur_ys
                rs = new_rs
                current_sum_r = np.sum(rs)
                
                if current_sum_r > best_sum_r:
                    best_sum_r = current_sum_r
                    best_xs = xs.copy()
                    best_ys = ys.copy()
                    best_rs = rs.copy()
                    
                # Increase delta_r
                delta_r *= 1.02
            else:
                # Reject
                delta_r *= 0.9
                if delta_r < 1e-7:
                    break
                    
                # Add jitter to escape local minima
                if np.random.rand() < 0.1:
                    xs += np.random.uniform(-0.01, 0.01, N)
                    ys += np.random.uniform(-0.01, 0.01, N)
                    xs = np.clip(xs, rs, 1-rs)
                    ys = np.clip(ys, rs, 1-rs)

    if best_xs is None:
        # Fallback
        xs = np.full(N, 0.5)
        ys = np.full(N, 0.5)
        rs = np.full(N, 0.01)
        return np.column_stack((xs, ys, rs))
        
    return np.column_stack((best_xs, best_ys, best_rs))
