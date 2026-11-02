import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    best_score = -np.inf
    best_x = None
    best_y = None
    best_r = None
    best_W = 1.0
    
    # Helper to calculate score
    def calc_score(x, y, r, W):
        H = 2.0 - W
        # Boundary penalties
        pen = 0.0
        for i in range(N):
            if x[i] < r[i]:
                pen += (r[i] - x[i])**2
            if x[i] > W - r[i]:
                pen += (x[i] - (W - r[i]))**2
            if y[i] < r[i]:
                pen += (r[i] - y[i])**2
            if y[i] > H - r[i]:
                pen += (y[i] - (H - r[i]))**2
        
        # Overlap penalties
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    pen += (min_dist - dist)**2
        
        total_r = np.sum(r)
        return total_r - 10.0 * pen, pen
    
    # Helper to check validity
    def is_valid(x, y, r, W):
        H = 2.0 - W
        for i in range(N):
            if x[i] < 1e-9 or x[i] > W - 1e-9 or y[i] < 1e-9 or y[i] > H - 1e-9:
                return False
            if r[i] < 1e-9:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9:
                    return False
        return True

    def run_sa(restart_idx):
        # Initialize
        # Try different aspect ratios and initial layouts
        W = 1.0
        if restart_idx % 3 == 1:
            W = 1.1
            H = 0.9
        elif restart_idx % 3 == 2:
            W = 0.9
            H = 1.1
        else:
            H = 1.0
        
        # Grid initialization
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N: break
                x[idx] = (i + 0.5) / side * W
                y[idx] = (j + 0.5) / side * H
                # Initial radius: slightly less than cell width to avoid overlap
                cell_w = W / side
                cell_h = H / side
                r[idx] = min(cell_w, cell_h) * 0.4
                idx += 1
        
        # Fill remaining if any (should be none for 21 with 5x5 grid)
        
        current_x = x.copy()
        current_y = y.copy()
        current_r = r.copy()
        current_W = W
        
        score, pen = calc_score(current_x, current_y, current_r, current_W)
        
        T = 0.1
        T_min = 0.0001
        cooling = 0.995
        
        while T > T_min:
            if time.time() - start_time > time_limit:
                break
            
            # Perturb
            new_x = current_x.copy()
            new_y = current_y.copy()
            new_r = current_r.copy()
            new_W = current_W
            
            # Randomly choose what to perturb
            # 1. Move a circle
            if np.random.rand() < 0.7:
                i = np.random.randint(N)
                step = 0.05 * T + 0.001
                new_x[i] += np.random.uniform(-step, step)
                new_y[i] += np.random.uniform(-step, step)
            # 2. Change radius
            elif np.random.rand() < 0.9:
                i = np.random.randint(N)
                step = 0.01 * T + 0.0001
                new_r[i] += np.random.uniform(-step, step)
                if new_r[i] < 0.001: new_r[i] = 0.001
            # 3. Change W
            else:
                step = 0.01 * T + 0.0001
                new_W += np.random.uniform(-step, step)
                if new_W < 0.5: new_W = 0.5
                if new_W > 1.5: new_W = 1.5
            
            new_score, new_pen = calc_score(new_x, new_y, new_r, new_W)
            delta = new_score - score
            
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                current_x = new_x
                current_y = new_y
                current_r = new_r
                current_W = new_W
                score = new_score
                pen = new_pen
                
                # Track best
                if score > best_score and pen < 1e-5: # Only update best if valid-ish
                    # Actually, we should update best if it's the best *valid* packing
                    # But during SA, we might have overlaps.
                    # Let's just track the best score, and validate at the end.
                    # Or better, track best valid score.
                    pass # We will collect candidates at the end or periodically
            
            T *= cooling
        
        return current_x, current_y, current_r, current_W, score

    # Run multiple restarts
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        x, y, r, W, sc = run_sa(i)
        
        # Validate and refine
        # Simple local search to remove any tiny overlaps
        for _ in range(100):
            valid = True
            H = 2.0 - W
            for k in range(N):
                if x[k] < r[k]: x[k] = r[k]
                if x[k] > W - r[k]: x[k] = W - r[k]
                if y[k] < r[k]: y[k] = r[k]
                if y[k] > H - r[k]: y[k] = H - r[k]
            
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_d = r[i] + r[j]
                    if dist < min_d:
                        valid = False
                        # Push apart
                        if dist == 0:
                            dx = np.random.uniform(-0.01, 0.01)
                            dy = np.random.uniform(-0.01, 0.01)
                            dist = np.sqrt(dx*dx + dy*dy)
                        push = (min_d - dist) / 2.0
                        ux = dx / dist
                        uy = dy / dist
                        x[i] += push * ux
                        y[i] += push * uy
                        x[j] -= push * ux
                        y[j] -= push * uy
            
            if valid:
                break
        
        if valid:
            final_score = np.sum(r)
            if final_score > best_score:
                best_score = final_score
                best_x = x.copy()
                best_y = y.copy()
                best_r = r.copy()
                best_W = W

    if best_x is None:
        # Fallback: simple grid
        side = 5
        W = 1.0
        H = 1.0
        x = np.array([(i + 0.5) / side for i in range(side) for _ in range(side)])[:N]
        y = np.array([j / side for j in range(side) for _ in range(side)])[:N]
        # Correct y generation
        ys = []
        xs = []
        for i in range(side):
            for j in range(side):
                if len(xs) < N:
                    xs.append((i + 0.5) / side)
                    ys.append((j + 0.5) / side)
        x = np.array(xs)
        y = np.array(ys)
        r = np.full(N, 0.5 / side * 0.9)
        best_x = x
        best_y = y
        best_r = r
        best_W = W

    result = np.column_stack((best_x, best_y, best_r))
    return result
