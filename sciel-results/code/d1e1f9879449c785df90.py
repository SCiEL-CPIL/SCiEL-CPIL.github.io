import numpy as np

N = 50

def construct():
    # Initial Guess: 4x4x4 grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    x = grid[:N].copy()
    
    t0 = np.time.time()
    timeout = 100.0
    
    # Precompute pair indices
    indices = np.triu_indices(N, 1)
    
    for step in range(5000):
        if np.time.time() - t0 > timeout:
            break
            
        # Calculate distances
        # x: (N, 3)
        # diff: (N, N, 3)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(d2, np.inf)
        
        # Boundary distances
        b_dist = np.min(np.stack([
            0.5 - x[:, 0], 0.5 + x[:, 0],
            0.5 - x[:, 1], 0.5 + x[:, 1],
            0.5 - x[:, 2], 0.5 + x[:, 2]
        ]), axis=0)
        
        min_pair_d = np.sqrt(np.min(d2)) / 2
        min_b = np.min(b_dist)
        r_min = min(min_pair_d, min_b)
        
        # Threshold for "active" constraints
        # We want to push points that are close to the limit
        thresh = r_min + 0.01
        
        # Forces
        forces = np.zeros_like(x)
        
        # Pair forces
        # For each pair (i, j) with d < 2*thresh
        # Force is proportional to 1/d^2 * unit_vector
        # Vectorized:
        # d2 < (2*thresh)^2
        mask = d2 < (2*thresh)**2
        # Get indices
        i_idx, j_idx = np.where(mask)
        
        for i, j in zip(i_idx, j_idx):
            if i == j: continue
            vec = x[i] - x[j]
            d = np.sqrt(d2[i, j])
            if d > 1e-9:
                # Repulsive force
                f = 1.0 / (d**2)
                forces[i] += f * vec / d
                forces[j] -= f * vec / d
                
        # Boundary forces
        # For each point i with b_dist < thresh
        b_mask = b_dist < thresh
        b_idx = np.where(b_mask)[0]
        
        for i in b_idx:
            p = x[i]
            # Determine face
            dists = np.array([0.5-p[0], 0.5+p[0], 0.5-p[1], 0.5+p[1], 0.5-p[2], 0.5+p[2]])
            face_idx = np.argmin(dists)
            d = dists[face_idx]
            if d > 1e-9:
                f = 1.0 / (d**2)
                move = np.zeros(3)
                if face_idx == 0: move[0] = -1
                elif face_idx == 1: move[0] = 1
                elif face_idx == 2: move[1] = -1
                elif face_idx == 3: move[1] = 1
                elif face_idx == 4: move[2] = -1
                elif face_idx == 5: move[2] = 1
                forces[i] += f * move
        
        # Apply forces
        # Learning rate
        lr = 0.005
        x += lr * forces
        
        # Keep inside cube (hard constraint)
        # If a point goes out, push it back
        x = np.clip(x, -0.5, 0.5)
        
        # Jitter
        if step % 50 == 0:
            x += np.random.normal(0, 0.001, x.shape)
            x = np.clip(x, -0.5, 0.5)

    return x
