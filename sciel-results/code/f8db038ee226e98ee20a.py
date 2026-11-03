"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    i_indices, j_indices = np.triu_indices(N, k=1)
    
    def is_valid(pos, rad):
        if np.any(pos[:, 0] - rad < -1e-10) or np.any(pos[:, 0] + rad > 1 + 1e-10):
            return False
        if np.any(pos[:, 1] - rad < -1e-10) or np.any(pos[:, 1] + rad > 1 + 1e-10):
            return False
        
        dx = pos[i_indices, 0] - pos[j_indices, 0]
        dy = pos[i_indices, 1] - pos[j_indices, 1]
        dist_sq = dx*dx + dy*dy
        rad_sum = rad[i_indices] + rad[j_indices]
        if np.any(dist_sq < rad_sum**2 - 1e-10):
            return False
        return True
    
    def initial_placement(seed):
        np.random.seed(seed)
        positions = np.zeros((N, 2))
        radii = np.zeros(N)
        
        # Hexagonal packing initialization
        # 5x5 grid + 1
        s = 0.2
        idx = 0
        for r in range(5):
            for c in range(5):
                if idx >= N: break
                x = c * s + 0.1 + (0.1 if r % 2 == 1 else 0)
                y = r * s * 0.866 + 0.1
                positions[idx, 0] = x
                positions[idx, 1] = y
                idx += 1
            if idx >= N: break
        
        if idx < N:
            positions[idx, 0] = 0.5
            positions[idx, 1] = 0.9
            idx += 1
            
        # Center
        cx = np.mean(positions[:, 0])
        cy = np.mean(positions[:, 1])
        positions[:, 0] -= cx - 0.5
        positions[:, 1] -= cy - 0.5
        positions[:, 0] = np.clip(positions[:, 0], 0.01, 0.99)
        positions[:, 1] = np.clip(positions[:, 1], 0.01, 0.99)
        
        radii[:] = 0.1
        return positions, radii
    
    def hill_climb(init_pos, init_rad, time_budget, seed):
        np.random.seed(seed)
        current_pos = init_pos.copy()
        current_rad = init_rad.copy()
        
        if not is_valid(current_pos, current_rad):
            return None, None, -np.inf
        
        current_score = np.sum(current_rad)
        best_pos = current_pos.copy()
        best_rad = current_rad.copy()
        best_score = current_score
        
        start_time = time.time()
        delta_r = 0.01
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() - start_time < time_budget:
            if no_improve_count > max_no_improve:
                break
            
            i = np.random.randint(N)
            move_type = np.random.rand()
            
            new_pos = current_pos.copy()
            new_rad = current_rad.copy()
            improved = False
            
            if move_type < 0.5:
                # Try to increase radius
                new_rad[i] += delta_r
                if new_rad[i] > 0.5:
                    new_rad[i] = 0.5
                if is_valid(new_pos, new_rad):
                    new_score = np.sum(new_rad)
                    if new_score > current_score:
                        current_pos = new_pos
                        current_rad = new_rad
                        current_score = new_score
                        improved = True
            else:
                # Try to move position
                dx = np.random.uniform(-0.05, 0.05)
                dy = np.random.uniform(-0.05, 0.05)
                new_pos[i, 0] += dx
                new_pos[i, 1] += dy
                if 0 <= new_pos[i, 0] <= 1 and 0 <= new_pos[i, 1] <= 1:
                    if is_valid(new_pos, new_rad):
                        current_pos = new_pos
                        improved = True
                        
                        # Try to increase radius after move
                        new_rad2 = current_rad.copy()
                        new_rad2[i] += delta_r
                        if is_valid(current_pos, new_rad2):
                            current_rad = new_rad2
                            current_score = np.sum(current_rad)
                            if current_score > best_score:
                                best_score = current_score
                                best_pos = current_pos.copy()
                                best_rad = current_rad.copy()
                            improved = True
                        else:
                            if current_score > best_score:
                                best_score = current_score
                                best_pos = current_pos.copy()
                                best_rad = current_rad.copy()
            
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count % 100 == 0:
                    delta_r *= 0.95
        
        return best_pos, best_rad, best_score
    
    start_time = time.time()
    time_limit = 108
    
    best_overall_pos = None
    best_overall_rad = None
    best_overall_score = -np.inf
    
    num_runs = 10
    time_per_run = time_limit / num_runs
    
    for run in range(num_runs):
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 1:
            break
        
        seed = 42 + run * 100
        init_pos, init_rad = initial_placement(seed)
        
        pos, rad, score = hill_climb(init_pos, init_rad, min(time_per_run, remaining_time), seed)
        
        if pos is not None and score > best_overall_score:
            best_overall_score = score
            best_overall_pos = pos
            best_overall_rad = rad
    
    if best_overall_pos is None or not is_valid(best_overall_pos, best_overall_rad):
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            cells.append([(col + 0.5) / side, (row + 0.5) / side, 0.3 / side])
        return np.array(cells)
    
    return np.column_stack([best_overall_pos, best_overall_rad])
# EVOLVE-BLOCK-END
