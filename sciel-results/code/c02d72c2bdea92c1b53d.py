import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = 110.0  # 10s margin for 120s limit
    
    # Precompute pairs
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    def is_valid(x, y, r):
        # Check boundaries
        for i in range(N):
            if x[i] < r[i] - 1e-9 or x[i] > 1 - r[i] + 1e-9:
                return False
            if y[i] < r[i] - 1e-9 or y[i] > 1 - r[i] + 1e-9:
                return False
        # Check overlaps
        for i, j in pairs:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist*min_dist - 1e-9:
                return False
        return True

    best_score = 0.0
    best_x, best_y, best_r = None, None, None
    
    # Try a few different initializations
    inits = []
    
    # Init 1: 6x6 Grid
    side = 6
    coords = []
    for i in range(side):
        for j in range(side):
            coords.append([(i + 0.5) / side, (j + 0.5) / side])
    coords = coords[:N]
    inits.append((np.array([c[0] for c in coords]), np.array([c[1] for c in coords]), np.full(N, 0.08)))
    
    # Init 2: 5x5 Grid + 1 center
    side = 5
    coords = []
    for i in range(side):
        for j in range(side):
            coords.append([(i + 0.5) / side, (j + 0.5) / side])
    coords.append([0.5, 0.5])
    inits.append((np.array([c[0] for c in coords]), np.array([c[1] for c in coords]), np.full(N, 0.09)))
    
    # Init 3: Random
    x_rand = np.random.uniform(0.1, 0.9, N)
    y_rand = np.random.uniform(0.1, 0.9, N)
    r_rand = np.full(N, 0.05)
    inits.append((x_rand, y_rand, r_rand))
    
    for x0, y0, r0 in inits:
        if time.time() - start_time > deadline:
            break
            
        x = x0.copy()
        y = y0.copy()
        r = r0.copy()
        
        learning_rate = 0.05
        penalty_weight = 20.0
        iterations = 3000
        
        for it in range(iterations):
            if time.time() - start_time > deadline:
                break
                
            grad_x = np.zeros(N)
            grad_y = np.zeros(N)
            grad_r = np.zeros(N)
            
            # Objective gradient
            grad_r -= 1.0
            
            # Boundary penalties
            for i in range(N):
                if x[i] - r[i] < 0:
                    viol = r[i] - x[i]
                    grad_x[i] += 2 * penalty_weight * (-viol)
                    grad_r[i] += 2 * penalty_weight * (viol)
                if 1 - x[i] - r[i] < 0:
                    viol = x[i] + r[i] - 1
                    grad_x[i] += 2 * penalty_weight * (viol)
                    grad_r[i] += 2 * penalty_weight * (viol)
                if y[i] - r[i] < 0:
                    viol = r[i] - y[i]
                    grad_y[i] += 2 * penalty_weight * (-viol)
                    grad_r[i] += 2 * penalty_weight * (viol)
                if 1 - y[i] - r[i] < 0:
                    viol = y[i] + r[i] - 1
                    grad_y[i] += 2 * penalty_weight * (viol)
                    grad_r[i] += 2 * penalty_weight * (viol)
                    
            # Overlap penalties
            for i, j in pairs:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                
                if dist_sq < min_dist*min_dist:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-6:
                        dx = 1e-6
                        dy = 0
                        dist = 1e-6
                    viol = min_dist - dist
                    factor = 2 * penalty_weight * viol / dist
                    
                    grad_x[i] += -factor * dx
                    grad_y[i] += -factor * dy
                    grad_x[j] += factor * dx
                    grad_y[j] += factor * dy
                    
                    grad_r[i] += 2 * penalty_weight * viol
                    grad_r[j] += 2 * penalty_weight * viol
            
            # Update
            x += learning_rate * grad_x
            y += learning_rate * grad_y
            r += learning_rate * grad_r
            
            # Clip
            r = np.maximum(r, 0.01)
            
            # Decay learning rate
            if it % 100 == 0:
                learning_rate *= 0.95
                # Add noise
                x += 0.01 * np.random.randn(N)
                y += 0.01 * np.random.randn(N)
                r += 0.005 * np.random.randn(N)
                r = np.maximum(r, 0.01)
                
            # Check validity
            if is_valid(x, y, r):
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
                    
    # If no valid solution found (unlikely), return a trivial one
    if best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Final polish: Ensure strict validity
    if not is_valid(best_x, best_y, best_r):
        scale = 1.0
        while not is_valid(best_x, best_y, best_r * scale) and scale > 0.01:
            scale *= 0.95
        best_r = best_r * scale
        
    result = np.column_stack((best_x, best_y, best_r))
    return result
