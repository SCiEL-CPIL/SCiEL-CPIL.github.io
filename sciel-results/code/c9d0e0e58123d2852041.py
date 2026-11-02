import numpy as np

N = 26

def construct():
    # Initialize with a grid
    rows = 5
    cols = 6
    
    x_coords = (np.arange(cols) + 0.5) / cols
    y_coords = (np.arange(rows) + 0.5) / rows
    
    points = []
    for i in range(rows):
        for j in range(cols):
            points.append([x_coords[j], y_coords[i]])
            
    points = np.array(points[:N])
    
    # Start with slightly larger radii to push towards boundary/overlap quickly
    # Max non-overlapping radius in this grid is 1/(2*6) = 0.0833
    # Let's start at 0.07
    radii = np.full(N, 0.07)
    
    def energy(state, radii):
        pen_x = np.sum(np.maximum(0, radii - state[:, 0])**2) + np.sum(np.maximum(0, state[:, 0] - (1 - radii))**2)
        pen_y = np.sum(np.maximum(0, radii - state[:, 1])**2) + np.sum(np.maximum(0, state[:, 1] - (1 - radii))**2)
        
        diff = state[:, np.newaxis, :] - state[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap = np.maximum(0, r_sum - dist)
        pen_overlap = np.sum(overlap**2) / 2.0
        
        weight = 1000.0
        return -np.sum(radii) + weight * (pen_x + pen_y + pen_overlap)
        
    current_state = points.copy()
    current_radii = radii.copy()
    current_energy = energy(current_state, current_radii)
    
    T = 0.1
    T_min = 0.001
    alpha = 0.9995
    n_iter = 100000
    
    pos_step = 0.05
    rad_step = 0.02
    
    for i in range(n_iter):
        idx = np.random.randint(N)
        
        new_state = current_state.copy()
        new_radii = current_radii.copy()
        
        new_state[idx, 0] += np.random.normal(0, pos_step)
        new_state[idx, 1] += np.random.normal(0, pos_step)
        
        new_radii[idx] += np.random.normal(0, rad_step)
        new_radii[idx] = np.clip(new_radii[idx], 0.01, 0.5)
        
        new_energy = energy(new_state, new_radii)
        
        delta_E = new_energy - current_energy
        
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            current_state = new_state
            current_radii = new_radii
            current_energy = new_energy
            
        T *= alpha
        if T < T_min:
            T = T_min
            
    result = np.hstack([current_state, current_radii.reshape(-1, 1)])
    return result
