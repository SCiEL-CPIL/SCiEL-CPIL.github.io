import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    best_score = -1.0
    best_config = None
    
    # Restart with different initializations
    # 1. Grid
    # 2. Hexagonal
    # 3. Random
    
    def run_heuristic(x, y, r, max_iters=5000, initial_delta=0.005):
        # x, y, r are arrays of length N
        # Returns best_score_found, best_x, best_y, best_r
        local_best_score = np.sum(r)
        local_best_x = x.copy()
        local_best_y = y.copy()
        local_best_r = r.copy()
        
        delta = initial_delta
        stuck_count = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            
            # 1. Expand
            r_new = r + delta
            
            # 2. Resolve Collisions and Boundaries
            # We work on copies to check validity
            x_curr = x.copy()
            y_curr = y.copy()
            r_curr = r_new.copy()
            
            # Iterate resolution
            resolved = False
            for res_iter in range(50):
                max_overlap = 0.0
                for i in range(N):
                    # Boundaries
                    if x_curr[i] < r_curr[i]:
                        x_curr[i] = r_curr[i]
                    elif x_curr[i] > 1.0 - r_curr[i]:
                        x_curr[i] = 1.0 - r_curr[i]
                        
                    if y_curr[i] < r_curr[i]:
                        y_curr[i] = r_curr[i]
                    elif y_curr[i] > 1.0 - r_curr[i]:
                        y_curr[i] = 1.0 - r_curr[i]
                    
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x_curr[i] - x_curr[j]
                        dy = y_curr[i] - y_curr[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_dist = r_curr[i] + r_curr[j]
                        
                        if dist < min_dist:
                            overlap = min_dist - dist
                            if overlap > max_overlap:
                                max_overlap = overlap
                            
                            if dist < 1e-9:
                                dx = np.random.uniform(-0.01, 0.01)
                                dy = np.random.uniform(-0.01, 0.01)
                                dist = np.sqrt(dx*dx + dy*dy)
                                
                            share = overlap / 2.0
                            ux = dx / dist
                            uy = dy / dist
                            
                            x_curr[i] -= share * ux
                            y_curr[i] -= share * uy
                            x_curr[j] += share * ux
                            y_curr[j] += share * uy
                            
                if max_overlap < 1e-6:
                    resolved = True
                    break
            
            if resolved:
                # Check if strictly valid
                valid = True
                for i in range(N):
                    if x_curr[i] < r_curr[i] - 1e-9 or x_curr[i] > 1.0 - r_curr[i] + 1e-9:
                        valid = False
                        break
                    if y_curr[i] < r_curr[i] - 1e-9 or y_curr[i] > 1.0 - r_curr[i] + 1e-9:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = x_curr[i] - x_curr[j]
                            dy = y_curr[i] - y_curr[j]
                            dist_sq = dx*dx + dy*dy
                            min_dist_sq = (r_curr[i] + r_curr[j])**2
                            if dist_sq < min_dist_sq - 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    score = np.sum(r_curr)
                    if score > local_best_score:
                        local_best_score = score
                        local_best_x = x_curr.copy()
                        local_best_y = y_curr.copy()
                        local_best_r = r_curr.copy()
                    
                    # Update state for next iteration
                    x = x_curr
                    y = y_curr
                    r = r_curr
                    stuck_count = 0
                    # Slowly decrease delta to converge
                    delta *= 0.995 
                    if delta < 1e-5:
                        break
                else:
                    # Should not happen if resolved, but safety
                    break
            else:
                # Failed to resolve
                # Try smaller delta?
                delta *= 0.5
                stuck_count += 1
                if stuck_count > 10:
                    break
        
        return local_best_score, local_best_x, local_best_y, local_best_r

    # Restart 1: Grid
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.01))
    x0 = np.array([c[0] for c in cells[:N]])
    y0 = np.array([c[1] for c in cells[:N]])
    r0 = np.array([c[2] for c in cells[:N]])
    
    score1, x1, y1, r1 = run_heuristic(x0, y0, r0)
    if score1 > best_score:
        best_score = score1
        best_config = np.column_stack((x1, y1, r1))
        
    # Restart 2-10: Random
    for k in range(10):
        if time.time() > deadline - 10.0:
            break
        x_rand = np.random.uniform(0.1, 0.9, N)
        y_rand = np.random.uniform(0.1, 0.9, N)
        r_rand = np.full(N, 0.01)
        
        # Ensure initial random config is valid?
        # With r=0.01 and random positions, they might overlap.
        # Run a quick resolution first?
        # Or just let run_heuristic handle it. It starts by expanding, which makes overlaps worse.
        # Better to start with a valid config.
        # Let's resolve the random config first without expanding.
        x_curr = x_rand.copy()
        y_curr = y_rand.copy()
        r_curr = r_rand.copy()
        for _ in range(100):
             max_ov = 0
             for i in range(N):
                 x_curr[i] = np.clip(x_curr[i], r_curr[i], 1-r_curr[i])
                 y_curr[i] = np.clip(y_curr[i], r_curr[i], 1-r_curr[i])
             for i in range(N):
                 for j in range(i+1, N):
                     dx = x_curr[i] - x_curr[j]
                     dy = y_curr[i] - y_curr[j]
                     dist = np.sqrt(dx*dx + dy*dy)
                     min_d = r_curr[i] + r_curr[j]
                     if dist < min_d:
                         ov = min_d - dist
                         if ov > max_ov: max_ov = ov
                         if dist < 1e-9:
                             dx = np.random.uniform(-0.01, 0.01)
                             dy = np.random.uniform(-0.01, 0.01)
                             dist = np.sqrt(dx*dx+dy*dy)
                         x_curr[i] -= 0.5 * ov * dx/dist
                         y_curr[i] -= 0.5 * ov * dy/dist
                         x_curr[j] += 0.5 * ov * dx/dist
                         y_curr[j] += 0.5 * ov * dy/dist
             if max_ov < 1e-6: break
             
        # Now run heuristic from this valid state
        score, x_best, y_best, r_best = run_heuristic(x_curr, y_curr, r_curr)
        if score > best_score:
            best_score = score
            best_config = np.column_stack((x_best, y_best, r_best))

    if best_config is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_config
