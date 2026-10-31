import numpy as np

def construct():
    N = 21

    def optimize(W, H, seed=0):
        rng = np.random.default_rng(seed)
        
        best_r = 0
        best_pos = None
        
        # Try different grid layouts
        layouts = [(5, 5), (4, 6), (3, 7), (6, 4), (7, 3), (2, 11), (11, 2)]
        
        for nx, ny in layouts:
            if nx * ny < N:
                continue
            
            # Create grid points
            # Centered
            xs = np.linspace(W / (2 * nx), W - W / (2 * nx), nx)
            ys = np.linspace(H / (2 * ny), H - H / (2 * ny), ny)
            X, Y = np.meshgrid(xs, ys, indexing='ij')
            pos = np.column_stack((X.flatten(), Y.flatten()))[:N]
            
            # Add small random jitter
            pos += rng.uniform(-0.01, 0.01, pos.shape)
            
            # Initial radius
            r = min(W, H) / max(nx, ny) * 0.3
            
            # Local Search
            step = 0.05
            for it in range(2000):
                # Calculate gradients
                grads = np.zeros_like(pos)
                
                # Pairwise repulsion
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < 2*r:
                            # Force is proportional to overlap
                            overlap = 2*r - dist
                            if dist > 1e-6:
                                fx = overlap * dx / dist
                                fy = overlap * dy / dist
                            else:
                                fx = rng.uniform(-1, 1) * overlap
                                fy = rng.uniform(-1, 1) * overlap
                            
                            grads[i, 0] += fx
                            grads[i, 1] += fy
                            grads[j, 0] -= fx
                            grads[j, 1] -= fy
                
                # Boundary repulsion
                for i in range(N):
                    if pos[i, 0] < r:
                        grads[i, 0] += (r - pos[i, 0]) * 10
                    elif pos[i, 0] > W - r:
                        grads[i, 0] -= (pos[i, 0] - (W - r)) * 10
                    
                    if pos[i, 1] < r:
                        grads[i, 1] += (r - pos[i, 1]) * 10
                    elif pos[i, 1] > H - r:
                        grads[i, 1] -= (pos[i, 1] - (H - r)) * 10
                        
                # Update positions
                pos += step * grads
                pos = np.clip(pos, r, np.array([W-r, H-r]))
                
                # Calculate max possible r
                min_dist = np.inf
                for i in range(N):
                    for j in range(i+1, N):
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        d = np.sqrt(dx*dx + dy*dy)
                        if d < min_dist:
                            min_dist = d
                
                min_boundary = np.inf
                for i in range(N):
                    min_boundary = min(min_boundary, pos[i, 0], W - pos[i, 0], pos[i, 1], H - pos[i, 1])
                
                max_possible_r = min(min_dist / 2, min_boundary)
                
                if max_possible_r > r * 1.01:
                    r = max_possible_r * 0.95
                    step *= 0.9 # Shrink step as we grow
                else:
                    # If stuck, try to increase step or restart?
                    # For now, just continue
                    pass
            
            if r > best_r:
                best_r = r
                best_pos = pos.copy()
        
        return best_pos, best_r

    best_sum = 0
    best_final_pos = None
    best_W = 1.0
    
    # Search aspect ratios
    for W in np.linspace(0.8, 1.2, 10):
        H = 2.0 - W
        # Run a few times with different seeds
        for seed in range(3):
            pos, r = optimize(W, H, seed)
            s = N * r
            if s > best_sum:
                best_sum = s
                best_final_pos = pos
                best_W = W
    
    result = np.zeros((N, 3))
    if best_final_pos is not None:
        result[:, 0] = best_final_pos[:, 0]
        result[:, 1] = best_final_pos[:, 1]
        result[:, 2] = best_sum / N
    else:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        result = np.array(cells[:N])
        
    return result
