import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    max_time = 140.0  # Safety margin
    
    # Best solution tracking
    best_score = 0.0
    best_pos = None
    best_rad = None
    
    def evaluate(pos, rad):
        # Check validity and return score
        # Valid if all in bounds and no overlap
        if np.any(pos[:, 0] < rad) or np.any(pos[:, 0] > 1 - rad) or \
           np.any(pos[:, 1] < rad) or np.any(pos[:, 1] > 1 - rad):
            return -1.0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < rad[i] + rad[j] - 1e-9:
                    return -1.0
        return np.sum(rad)

    def optimize(initial_pos, initial_rad, time_limit):
        pos = initial_pos.copy()
        rad = initial_rad.copy()
        current_score = evaluate(pos, rad)
        best_local_score = current_score
        best_local_pos = pos.copy()
        best_local_rad = rad.copy()
        
        step_size = 0.01
        no_improve_count = 0
        
        while time.time() - start_time < time_limit:
            improved = False
            for i in range(N):
                # Calculate max radius for i
                min_dist = 1.0 # Boundary
                # Check boundaries
                min_dist = min(min_dist, pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
                
                # Check other circles
                for j in range(N):
                    if i == j: continue
                    d = np.linalg.norm(pos[i] - pos[j])
                    max_r_i = d - rad[j]
                    if max_r_i < min_dist:
                        min_dist = max_r_i
                
                new_rad = max(0, min_dist)
                
                # If we can increase radius, do it
                if new_rad > rad[i] + 1e-6:
                    rad[i] = new_rad
                    improved = True
                else:
                    # If stuck, try to move
                    # Gradient calculation
                    grad = np.zeros(2)
                    # From boundaries
                    if pos[i,0] - rad[i] < 1e-3: grad[0] += 1
                    if 1 - pos[i,0] - rad[i] < 1e-3: grad[0] -= 1
                    if pos[i,1] - rad[i] < 1e-3: grad[1] += 1
                    if 1 - pos[i,1] - rad[i] < 1e-3: grad[1] -= 1
                    
                    # From neighbors
                    for j in range(N):
                        if i == j: continue
                        d = np.linalg.norm(pos[i] - pos[j])
                        if d > 1e-9 and d < rad[i] + rad[j] + 1e-3:
                            # Vector from j to i
                            vec = (pos[i] - pos[j]) / d
                            grad += vec
                    
                    norm_grad = np.linalg.norm(grad)
                    if norm_grad > 1e-6:
                        grad /= norm_grad
                        # Move
                        pos[i] += step_size * grad
                        # Project back
                        pos[i,0] = np.clip(pos[i,0], rad[i], 1-rad[i])
                        pos[i,1] = np.clip(pos[i,1], rad[i], 1-rad[i])
                        improved = True
            
            # Validity check and repair
            score = evaluate(pos, rad)
            if score > 0:
                if score > best_local_score:
                    best_local_score = score
                    best_local_pos = pos.copy()
                    best_local_rad = rad.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Invalid, push apart
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.linalg.norm(pos[i] - pos[j])
                        min_d = rad[i] + rad[j]
                        if d < min_d:
                            if d < 1e-9:
                                pos[i,0] += 0.01
                                pos[j,0] -= 0.01
                            else:
                                vec = (pos[i] - pos[j]) / d
                                push = (min_d - d) / 2 + 1e-5
                                pos[i] += push * vec
                                pos[j] -= push * vec
                    # Clamp to bounds
                    pos[i,0] = np.clip(pos[i,0], rad[i], 1-rad[i])
                    pos[i,1] = np.clip(pos[i,1], rad[i], 1-rad[i])
                
                # If still invalid, shrink radii slightly
                rad *= 0.99
                score = evaluate(pos, rad)
                if score > 0:
                    if score > best_local_score:
                        best_local_score = score
                        best_local_pos = pos.copy()
                        best_local_rad = rad.copy()
                
                # Reduce step size if struggling
                step_size *= 0.95
                if no_improve_count > 100:
                    # Perturb
                    pos += np.random.normal(0, 0.01, pos.shape)
                    pos[:,0] = np.clip(pos[:,0], rad, 1-rad)
                    pos[:,1] = np.clip(pos[:,1], rad, 1-rad)
                    no_improve_count = 0
                    step_size = 0.01
            
            # Global time check inside loop is handled by while condition
            # But we should break if step_size is too small
            if step_size < 1e-6:
                break
                
        return best_local_pos, best_local_rad, best_local_score

    # Multi-restart strategy
    # 1. Grid start
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
    init_pos = np.array(cells[:N])
    init_rad = np.full(N, 0.05)
    
    p, r, s = optimize(init_pos, init_rad, max_time - 20)
    if s > best_score:
        best_score = s
        best_pos = p
        best_rad = r
        
    # 2. Random starts if time permits
    remaining = max_time - (time.time() - start_time)
    if remaining > 10:
        for _ in range(3):
            if time.time() - start_time > max_time - 5: break
            init_pos = np.random.rand(N, 2) * 0.8 + 0.1
            init_rad = np.full(N, 0.05)
            p, r, s = optimize(init_pos, init_rad, max_time - (time.time() - start_time) - 5)
            if s > best_score:
                best_score = s
                best_pos = p
                best_rad = r

    # Final validation and formatting
    if best_pos is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Ensure strict validity one last time
    # Slightly shrink radii to ensure disjoint interiors
    final_rad = best_rad * (1 - 1e-5)
    final_pos = best_pos
    
    # Construct result
    result = np.column_stack((final_pos, final_rad))
    return result
