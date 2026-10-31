import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    def max_radius_for_circle(i, xs, ys, rs, W, H):
        """Compute max radius for circle i given others fixed."""
        r_max = min(xs[i], W - xs[i], ys[i], H - ys[i])
        for j in range(N):
            if i == j:
                continue
            dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
            r_max = min(r_max, dist - rs[j])
        return max(0.0, r_max)
    
    def local_search(xs, ys, rs, W, H, max_iter=3000):
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Expand all radii to max
            for i in range(N):
                new_r = max_radius_for_circle(i, xs, ys, rs, W, H)
                if new_r > rs[i] + 1e-12:
                    rs[i] = new_r
                    improved = True
            
            # Try moving each circle
            for i in range(N):
                # Try to find a move that increases the potential radius
                best_pos = (xs[i], ys[i])
                best_potential = -1
                
                # Current potential
                cur_pot = max_radius_for_circle(i, xs, ys, rs, W, H)
                
                # Try moves in 8 directions with various step sizes
                for step in [0.005, 0.01, 0.02, 0.05]:
                    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
                        nx = xs[i] + dx * step
                        ny = ys[i] + dy * step
                        if nx < 0 or nx > W or ny < 0 or ny > H:
                            continue
                        # Temporarily move
                        old_x, old_y = xs[i], ys[i]
                        xs[i], ys[i] = nx, ny
                        pot = max_radius_for_circle(i, xs, ys, rs, W, H)
                        xs[i], ys[i] = old_x, old_y
                        
                        if pot > best_potential + 1e-12:
                            best_potential = pot
                            best_pos = (nx, ny)
                
                if best_pos[0] != xs[i] or best_pos[1] != ys[i]:
                    xs[i], ys[i] = best_pos
                    improved = True
            
            if not improved:
                break
        
        # Final expansion
        for i in range(N):
            rs[i] = max_radius_for_circle(i, xs, ys, rs, W, H)
        
        return xs, ys, rs
    
    best_score = 0
    best_config = None
    
    # Multiple trials with different aspect ratios and seeds
    for trial in range(40):
        if time.time() > deadline:
            break
        
        # Vary aspect ratio
        H = 0.3 + (trial % 15) * 0.08
        W = 4.0 - 2.0 * H
        if W < 0.2 or H < 0.2:
            continue
        
        # Grid init with slight random perturbation
        rng = np.random.RandomState(trial)
        cols = max(1, int(np.ceil(np.sqrt(N * W / H))))
        rows = max(1, int(np.ceil(N / cols)))
        
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        
        count = 0
        for r in range(rows):
            for c in range(cols):
                if count >= N:
                    break
                cx = (c + 0.5) * W / cols
                cy = (r + 0.5) * H / rows
                # Add small perturbation
                cx += rng.uniform(-0.02, 0.02)
                cy += rng.uniform(-0.02, 0.02)
                cx = np.clip(cx, 0.01, W - 0.01)
                cy = np.clip(cy, 0.01, H - 0.01)
                xs[count] = cx
                ys[count] = cy
                dx = W / cols
                dy = H / rows
                rs[count] = min(dx, dy) * 0.35
                rs[count] = min(rs[count], cx, W-cx, cy, H-cy)
                rs[count] = max(rs[count], 0.005)
                count += 1
        
        xs, ys, rs = local_search(xs, ys, rs, W, H, max_iter=1500)
        
        # Validate
        valid = True
        for i in range(N):
            if xs[i] < -1e-8 or xs[i] > W + 1e-8 or ys[i] < -1e-8 or ys[i] > H + 1e-8:
                valid = False
                break
            if xs[i] - rs[i] < -1e-8 or xs[i] + rs[i] > W + 1e-8:
                valid = False
                break
            if ys[i] - rs[i] < -1e-8 or ys[i] + rs[i] > H + 1e-8:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    if dist < rs[i] + rs[j] - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            s = np.sum(rs)
            if s > best_score:
                best_score = s
                best_config = np.column_stack([xs, ys, rs])
    
    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_config
