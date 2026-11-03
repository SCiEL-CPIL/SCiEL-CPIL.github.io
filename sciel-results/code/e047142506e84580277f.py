"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    np.random.seed(42)
    
    # Initialization: Place circles on a grid with small radii
    side = int(np.ceil(np.sqrt(N)))
    positions = np.zeros((N, 2))
    for i in range(N):
        row = i // side
        col = i % side
        positions[i, 0] = (col + 0.5) / side
        positions[i, 1] = (row + 0.5) / side
        
    radii = np.full(N, 0.02)
    
    # Pre-compute indices for pair checks
    i_indices, j_indices = np.triu_indices(N, k=1)
    
    def is_valid(pos, rad):
        # Check boundaries
        if np.any(pos[:, 0] - rad < 0) or np.any(pos[:, 0] + rad > 1):
            return False
        if np.any(pos[:, 1] - rad < 0) or np.any(pos[:, 1] + rad > 1):
            return False
        
        # Check overlaps
        dx = pos[i_indices, 0] - pos[j_indices, 0]
        dy = pos[i_indices, 1] - pos[j_indices, 1]
        dist_sq = dx*dx + dy*dy
        rad_sum = rad[i_indices] + rad[j_indices]
        if np.any(dist_sq < rad_sum**2 - 1e-9):
            return False
        return True

    def get_score(pos, rad):
        if not is_valid(pos, rad):
            return -np.inf
        return np.sum(rad)

    current_pos = positions.copy()
    current_rad = radii.copy()
    current_score = get_score(current_pos, current_rad)
    
    best_pos = current_pos.copy()
    best_rad = current_rad.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 0.1
    T_min = 0.001
    alpha = 0.9995
    max_iters = 2000000 # High count, will break on time
    
    start_time = time.time()
    time_limit = 110 # Keep some buffer for 120s limit
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Generate a move
        i = np.random.randint(N)
        move_type = np.random.rand()
        
        new_pos = current_pos.copy()
        new_rad = current_rad.copy()
        
        if move_type < 0.5:
            # Move position
            delta = np.random.normal(0, 0.05, 2)
            new_pos[i] += delta
        else:
            # Change radius
            delta_r = np.random.normal(0, 0.05)
            new_rad[i] += delta_r
            if new_rad[i] < 0.01:
                new_rad[i] = 0.01
            if new_rad[i] > 0.5:
                new_rad[i] = 0.5

        # Check validity
        if is_valid(new_pos, new_rad):
            new_score = np.sum(new_rad)
            delta_score = new_score - current_score
            
            # Acceptance probability
            if delta_score > 0:
                accept = True
            else:
                if T > T_min:
                    accept_prob = np.exp(delta_score / T)
                    accept = np.random.rand() < accept_prob
                else:
                    accept = False
            
            if accept:
                current_pos = new_pos
                current_rad = new_rad
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_pos = current_pos.copy()
                    best_rad = current_rad.copy()
        else:
            # If invalid, reject
            pass
            
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            
    # Final check
    if not is_valid(best_pos, best_rad):
        # Fallback to initial grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return np.column_stack([best_pos, best_rad])
# EVOLVE-BLOCK-END
