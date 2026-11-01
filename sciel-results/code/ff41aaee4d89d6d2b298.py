import numpy as np
import time

N = 50

def construct():
    # Start with a jittered grid to break symmetry
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small jitter to avoid perfect grid symmetries
    centers += np.random.uniform(-0.01, 0.01, size=(N, 3))
    
    # Clip to ensure inside
    centers = np.clip(centers, -0.49, 0.49)
    
    start_time = time.time()
    max_time = 110  # Leave 10s buffer
    
    # Local search: Move the bottleneck
    step_size = 0.01
    min_step = 1e-6
    
    while time.time() - start_time < max_time:
        # Compute pairwise distances
        # diff[i, j] = centers[i] - centers[j]
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        # Compute wall distances
        # min distance to any wall for each point
        # walls are at +/- 0.5
        # dist to -0.5 is x + 0.5, to 0.5 is 0.5 - x
        wall_dists = np.minimum(
            np.minimum(centers[:, 0] + 0.5, 0.5 - centers[:, 0]),
            np.minimum(centers[:, 1] + 0.5, 0.5 - centers[:, 1]),
            np.minimum(centers[:, 2] + 0.5, 0.5 - centers[:, 2])
        )
        
        # Find global minimum distance
        min_pair_dist = np.min(dist)
        min_wall_dist = np.min(wall_dists)
        
        if min_pair_dist < min_wall_dist:
            # Bottleneck is a pair
            i, j = np.unravel_index(np.argmin(dist), dist.shape)
            # Move i and j apart
            d_vec = centers[i] - centers[j]
            d_norm = np.linalg.norm(d_vec)
            if d_norm > 1e-9:
                d_dir = d_vec / d_norm
                centers[i] += step_size * d_dir
                centers[j] -= step_size * d_dir
            else:
                # If coincident, random kick
                centers[i] += np.random.uniform(-step_size, step_size, 3)
                centers[j] -= np.random.uniform(-step_size, step_size, 3)
        else:
            # Bottleneck is a wall
            i = np.argmin(wall_dists)
            # Determine which wall is closest
            x, y, z = centers[i]
            dist_to_walls = {
                'x_neg': x + 0.5,
                'x_pos': 0.5 - x,
                'y_neg': y + 0.5,
                'y_pos': 0.5 - y,
                'z_neg': z + 0.5,
                'z_pos': 0.5 - z
            }
            min_wall_key = min(dist_to_walls, key=dist_to_walls.get)
            
            move_vec = np.zeros(3)
            if min_wall_key == 'x_neg': move_vec[0] = 1
            elif min_wall_key == 'x_pos': move_vec[0] = -1
            elif min_wall_key == 'y_neg': move_vec[1] = 1
            elif min_wall_key == 'y_pos': move_vec[1] = -1
            elif min_wall_key == 'z_neg': move_vec[2] = 1
            elif min_wall_key == 'z_pos': move_vec[2] = -1
            
            centers[i] += step_size * move_vec
            
        # Clip to box
        centers = np.clip(centers, -0.5, 0.5)
        
        # Optional: Add small random noise to escape local minima
        # This is a simple hill climbing, so it might get stuck.
        # We can periodically add noise.
        if np.random.random() < 0.1:
            idx = np.random.randint(N)
            centers[idx] += np.random.uniform(-0.05, 0.05, 3)
            centers[idx] = np.clip(centers[idx], -0.5, 0.5)
            
    return centers
