import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    max_time = 115.0  # Safety margin for 120s limit
    
    best_score = -1.0
    best_result = None

    # Initial configurations to try
    # 1. Square grid
    # 2. Rectangular grid (wider)
    # 3. Rectangular grid (taller)
    # 4. Random
    
    def run_optimization(w_init, h_init, positions_init, radii_init):
        w = w_init
        h = h_init
        pos = positions_init.copy()
        rad = radii_init.copy()
        
        # Ensure initial validity or close to it
        # Simple projection to boundaries
        for i in range(N):
            rad[i] = min(rad[i], w/2, h/2)
            pos[i][0] = np.clip(pos[i][0], rad[i], w - rad[i])
            pos[i][1] = np.clip(pos[i][1], rad[i], h - rad[i])
            
        lr_pos = 0.01
        lr_rad = 0.005
        lr_wh = 0.001
        penalty_weight = 10.0
        
        for it in range(2000):
            if time.time() - start_time > max_time:
                break
                
            # Calculate gradients
            grad_pos = np.zeros_like(pos)
            grad_rad = np.zeros_like(rad)
            grad_w = 0.0
            grad_h = 0.0
            
            # Objective: sum(rad)
            # Penalty: overlaps + boundaries
            
            # Overlaps
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[i][0] - pos[j][0]
                    dy = pos[i][1] - pos[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < 1e-9:
                        dist = 1e-9
                        dx, dy = 1e-9, 0.0
                    min_dist = rad[i] + rad[j]
                    if dist < min_dist:
                        # Penalty gradient
                        # d/dpos (min_dist - dist)^2 = -2(min_dist - dist) * (pos_i - pos_j)/dist
                        coeff = 2.0 * penalty_weight * (min_dist - dist) / dist
                        grad_pos[i][0] -= coeff * dx
                        grad_pos[i][1] -= coeff * dy
                        grad_pos[j][0] += coeff * dx
                        grad_pos[j][1] += coeff * dy
                        grad_rad[i] -= 2.0 * penalty_weight * (min_dist - dist)
                        grad_rad[j] -= 2.0 * penalty_weight * (min_dist - dist)
            
            # Boundaries
            for i in range(N):
                # x bounds
                if pos[i][0] < rad[i]:
                    # Penalty: (rad - x)^2
                    # d/dx = -2(rad - x)
                    # d/drad = 2(rad - x)
                    diff = rad[i] - pos[i][0]
                    grad_pos[i][0] += 2.0 * penalty_weight * diff
                    grad_rad[i] -= 2.0 * penalty_weight * diff
                elif pos[i][0] > w - rad[i]:
                    diff = pos[i][0] - (w - rad[i])
                    grad_pos[i][0] -= 2.0 * penalty_weight * diff
                    grad_rad[i] -= 2.0 * penalty_weight * diff
                    grad_w += 2.0 * penalty_weight * diff # d/dw of (x - (w-r))^2 is -2(x-w+r) = 2(w-r-x) = -2diff? 
                    # Let's re-evaluate: f = (x - w + r)^2. df/dw = 2(x-w+r)(-1) = -2diff. 
                    # Wait, penalty is added to loss, so we subtract gradient from objective.
                    # Objective J = Sum(rad) - Penalty.
                    # Grad J wrt w = - d(Penalty)/dw.
                    # d(Penalty)/dw = -2diff. So Grad J wrt w = 2diff.
                    # We want to increase J, so w += lr * Grad J.
                    pass # Handled below in general case

                if pos[i][0] < rad[i]:
                    # Already handled above
                    pass
                if pos[i][0] > w - rad[i]:
                    # Already handled above
                    pass
                
                # y bounds
                if pos[i][1] < rad[i]:
                    diff = rad[i] - pos[i][1]
                    grad_pos[i][1] += 2.0 * penalty_weight * diff
                    grad_rad[i] -= 2.0 * penalty_weight * diff
                elif pos[i][1] > h - rad[i]:
                    diff = pos[i][1] - (h - rad[i])
                    grad_pos[i][1] -= 2.0 * penalty_weight * diff
                    grad_rad[i] -= 2.0 * penalty_weight * diff
                    grad_h += 2.0 * penalty_weight * diff # Similar logic

            # Add objective gradient (sum of radii)
            grad_rad += 1.0
            
            # Update
            pos += lr_pos * grad_pos
            rad += lr_rad * grad_rad
            w += lr_wh * grad_w
            h += lr_wh * grad_h
            
            # Constraint: w + h = 2
            # Project w, h to satisfy w+h=2
            mid = (w + h) / 2.0
            w = mid - (w - h)/2.0 # This is just w
            # Actually, let's just enforce w+h=2 by adjusting w and h symmetrically or fixing one.
            # Better: w = 1 + delta, h = 1 - delta.
            # Let's re-parameterize: optimize delta.
            # For now, let's just project:
            w_new = w
            h_new = h
            # If we optimize w and h independently, they drift.
            # Let's stick to w and h as independent variables but project back.
            w = w_new
            h = h_new
            # Force w+h=2
            # Let's just use w as the variable and h = 2-w.
            # This requires rewriting the loop.
            
            # Simpler: Just fix aspect ratio in this simple version?
            # No, aspect ratio is key.
            # Let's restart the update logic with w as single variable for aspect.
            
            # To keep code simple and robust within time, let's fix aspect ratio to 1.0 (square) 
            # or try a few fixed aspect ratios. 
            # Given the complexity of dynamic aspect optimization in a short snippet, 
            # I will implement a solver that optimizes positions and radii for a FIXED w, h.
            # Then I will try a few common aspect ratios (1.0, 1.2, 0.8, 1.5, 0.6).
            
            # The loop above is messy. Let's rewrite the inner solver cleanly.
            return None # Placeholder, will replace with clean implementation

    def solve_fixed_aspect(w, h, init_pos, init_rad, time_limit):
        pos = init_pos.copy()
        rad = init_rad.copy()
        
        # Project initial state
        for i in range(N):
            rad[i] = min(rad[i], w/2, h/2)
            pos[i][0] = np.clip(pos[i][0], rad[i], w - rad[i])
            pos[i][1] = np.clip(pos[i][1], rad[i], h - rad[i])
            
        lr_pos = 0.01
        lr_rad = 0.005
        penalty_weight = 50.0
        
        for it in range(5000):
            if time.time() - start_time > time_limit:
                break
                
            grad_pos = np.zeros_like(pos)
            grad_rad = np.zeros_like(rad)
            
            # Overlaps
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[i][0] - pos[j][0]
                    dy = pos[i][1] - pos[j][1]
                    dist_sq = dx*dx + dy*dy
                    min_dist = rad[i] + rad[j]
                    if dist_sq < min_dist*min_dist:
                        dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                        if dist < min_dist:
                            coeff = 2.0 * penalty_weight * (min_dist - dist) / dist
                            grad_pos[i][0] -= coeff * dx
                            grad_pos[i][1] -= coeff * dy
                            grad_pos[j][0] += coeff * dx
                            grad_pos[j][1] += coeff * dy
                            grad_rad[i] -= 2.0 * penalty_weight * (min_dist - dist)
                            grad_rad[j] -= 2.0 * penalty_weight * (min_dist - dist)
            
            # Boundaries
            for i in range(N):
                if pos[i][0] < rad[i]:
                    diff = rad[i] - pos[i][0]
                    grad_pos[i][0] += 2.0 * penalty_weight * diff
                    grad_rad[i] -= 2.0 * penalty_weight * diff
                elif pos[i][0] > w - rad[i]:
                    diff = pos[i][0] - (w - rad[i])
                    grad_pos[i][0] -= 2.0 * penalty_weight * diff
                    grad_rad[i] -= 2.0 * penalty_weight * diff
                    
                if pos[i][1] < rad[i]:
                    diff = rad[i] - pos[i][1]
                    grad_pos[i][1] += 2.0 * penalty_weight * diff
                    grad_rad[i] -= 2.0 * penalty_weight * diff
                elif pos[i][1] > h - rad[i]:
                    diff = pos[i][1] - (h - rad[i])
                    grad_pos[i][1] -= 2.0 * penalty_weight * diff
                    grad_rad[i] -= 2.0 * penalty_weight * diff
            
            grad_rad += 1.0
            
            pos += lr_pos * grad_pos
            rad += lr_rad * grad_rad
            
            # Clamp radii to positive
            rad = np.maximum(rad, 1e-4)
            
            # Project positions to stay within bounds roughly
            for i in range(N):
                pos[i][0] = np.clip(pos[i][0], 0, w)
                pos[i][1] = np.clip(pos[i][1], 0, h)
                # Hard constraint enforcement for radii vs bounds is done by penalty, 
                # but we can also shrink radius if it violates too much to help convergence
                # Actually, the penalty handles it.
                
        # Final check and cleanup
        score = np.sum(rad)
        
        # Verify validity
        valid = True
        for i in range(N):
            if rad[i] < 0: valid = False
            if pos[i][0] < rad[i] - 1e-5 or pos[i][0] > w - rad[i] + 1e-5: valid = False
            if pos[i][1] < rad[i] - 1e-5 or pos[i][1] > h - rad[i] + 1e-5: valid = False
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rad[i] + rad[j] - 1e-5: valid = False
                
        if not valid:
            # Try to fix by shrinking radii slightly
            for i in range(N):
                rad[i] *= 0.95
            # Re-check
            valid = True
            for i in range(N):
                if pos[i][0] < rad[i] - 1e-5 or pos[i][0] > w - rad[i] + 1e-5: valid = False
                if pos[i][1] < rad[i] - 1e-5 or pos[i][1] > h - rad[i] + 1e-5: valid = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[i][0] - pos[j][0]
                    dy = pos[i][1] - pos[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < rad[i] + rad[j] - 1e-5: valid = False
            
            if not valid:
                return -1, None

        return score, (pos, rad, w, h)

    # Try different aspect ratios
    aspect_ratios = [1.0, 1.2, 1.5, 0.8, 0.6, 1.1, 1.3]
    
    for ar in aspect_ratios:
        w = 2.0 * ar / (1 + ar)
        h = 2.0 / (1 + ar)
        
        # Initialize with grid
        side_x = int(np.ceil(np.sqrt(N * ar)))
        side_y = int(np.ceil(N / side_x))
        
        init_pos = []
        init_rad = []
        for i in range(N):
            ix = i % side_x
            iy = i // side_x
            x = (ix + 0.5) * w / side_x
            y = (iy + 0.5) * h / side_y
            r = 0.5 * min(w/side_x, h/side_y) * 0.9
            init_pos.append([x, y])
            init_rad.append(r)
            
        init_pos = np.array(init_pos)
        init_rad = np.array(init_rad)
        
        score, result = solve_fixed_aspect(w, h, init_pos, init_rad, max_time)
        
        if score > best_score and result is not None:
            best_score = score
            pos, rad, w_res, h_res = result
            # Format result: (x, y, r)
            res_arr = np.zeros((N, 3))
            for i in range(N):
                res_arr[i][0] = pos[i][0]
                res_arr[i][1] = pos[i][1]
                res_arr[i][2] = rad[i]
            best_result = res_arr
            
    if best_result is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
