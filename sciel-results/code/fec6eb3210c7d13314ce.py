import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_score = -np.inf
    best_result = None
    
    # Initial layouts: Grid
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
            
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 - W
        
        # Grid initialization
        side = int(np.ceil(np.sqrt(N)))
        # We might need to adjust grid to fit N circles better
        # For N=21, 5x5 grid is 25 cells. We use 21.
        # Or 4x6 = 24 cells.
        
        # Let's use a simple grid for init
        # Determine grid dims
        grid_w = int(np.ceil(np.sqrt(N * W/H)))
        grid_h = int(np.ceil(N / grid_w))
        
        xs = []
        ys = []
        rs = []
        
        # Calculate cell size
        cw = W / grid_w
        ch = H / grid_h
        
        count = 0
        for i in range(grid_w):
            for j in range(grid_h):
                if count >= N:
                    break
                cx = (i + 0.5) * cw
                cy = (j + 0.5) * ch
                # Max radius is half min(cw, ch) but constrained by neighbors
                # Init with small radius
                r = 0.4 * min(cw, ch)
                xs.append(cx)
                ys.append(cy)
                rs.append(r)
                count += 1
            if count >= N:
                break
        
        x0 = np.array([W] + xs + ys + rs)
        
        # Define objective
        def objective(params):
            W_opt = params[0]
            H_opt = 2.0 - W_opt
            if W_opt <= 0.1 or H_opt <= 0.1:
                return 1e6
            
            xs_opt = params[1:1+N]
            ys_opt = params[1+N:1+2*N]
            rs_opt = params[1+2*N:1+3*N]
            
            # Sum of radii
            sum_r = np.sum(rs_opt)
            
            # Overlap penalty
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs_opt[i] - xs_opt[j]
                    dy = ys_opt[i] - ys_opt[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = rs_opt[i] + rs_opt[j]
                    if dist < min_dist:
                        penalty += (min_dist - dist)**2
            
            # Boundary penalty
            for i in range(N):
                if xs_opt[i] < rs_opt[i]:
                    penalty += (rs_opt[i] - xs_opt[i])**2
                if xs_opt[i] > W_opt - rs_opt[i]:
                    penalty += (xs_opt[i] - (W_opt - rs_opt[i]))**2
                if ys_opt[i] < rs_opt[i]:
                    penalty += (rs_opt[i] - ys_opt[i])**2
                if ys_opt[i] > H_opt - rs_opt[i]:
                    penalty += (ys_opt[i] - (H_opt - rs_opt[i]))**2
            
            # Large penalty for invalid W
            if W_opt <= 0 or W_opt >= 2:
                penalty += 1e6
                
            return -sum_r + 100.0 * penalty
        
        # Bounds
        bounds = [(0.1, 1.9)] + [(0.0, 2.0)]*(2*N) + [(0.0, 1.0)]*N
        
        # Optimize
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12})
            
            if res.success:
                W_opt = res.x[0]
                H_opt = 2.0 - W_opt
                xs_opt = res.x[1:1+N]
                ys_opt = res.x[1+N:1+2*N]
                rs_opt = res.x[1+2*N:1+3*N]
                
                # Check validity strictly
                valid = True
                # Perimeter check on bounding box
                min_x = np.min(xs_opt - rs_opt)
                max_x = np.max(xs_opt + rs_opt)
                min_y = np.min(ys_opt - rs_opt)
                max_y = np.max(ys_opt + rs_opt)
                
                # The rectangle containing the circles must have perimeter <= 4
                # But the task says "in a variable rectangle of perimeter 4".
                # This usually means the container is fixed at perimeter 4.
                # However, the interface says "whose bounding rectangle has perimeter at most 4".
                # This implies we can return circles that fit in a smaller box too.
                # But to maximize sum of radii, we should use the full perimeter 4 space.
                # So we should check if the circles fit in a box of perimeter 4.
                # The box is defined by the bounding box of the circles?
                # "n disjoint circles whose bounding rectangle has perimeter at most 4"
                # Yes, the evaluator computes the bounding box of the returned circles.
                
                w_bb = max_x - min_x
                h_bb = max_y - min_y
                perim = 2 * (w_bb + h_bb)
                
                if perim > 4.0 + 1e-9:
                    valid = False
                else:
                    # Check overlaps
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = xs_opt[i] - xs_opt[j]
                            dy = ys_opt[i] - ys_opt[j]
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < rs_opt[i] + rs_opt[j] - 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    score = np.sum(rs_opt)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack((xs_opt, ys_opt, rs_opt))
                        
        except Exception:
            pass

    if best_result is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
