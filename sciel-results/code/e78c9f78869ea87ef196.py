import numpy as np
import scipy.optimize as so
import time

N = 21

def construct():
    # Deadline guard: 140 seconds to be safe
    start_time = time.time()
    deadline = start_time + 140
    
    best_val = -np.inf
    best_x = None
    
    # Try different aspect ratios (width w, height 2-w)
    # w in (0, 2)
    aspect_ratios = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8]
    
    for w_init in aspect_ratios:
        if time.time() > deadline - 20: # Stop if less than 20s left
            break
            
        h_init = 2.0 - w_init
        if h_init <= 0.1: continue
        
        # Create a grid initialization
        # Choose grid dimensions roughly matching aspect ratio
        # If w > h, more columns
        # Let's try a few grid shapes for each aspect ratio
        # For simplicity, let's use a fixed grid shape that works well for 21, e.g., 3x7 or 4x6
        # We'll just place 21 circles in a 5x5 grid (25 slots) and let optimizer prune/adjust?
        # No, better to match the grid to the count.
        
        # Let's use a 3x7 grid (21 cells) if aspect ratio is wide, 7x3 if tall, else 4x6 (drop 3) or 5x5 (drop 4)
        # Actually, SLSQP can handle variable radii, so we can just use a dense grid.
        
        # Let's try a 4x6 grid (24 circles) and drop 3? Or just 21 in a 5x5?
        # Let's use a 3x7 grid for wide, 7x3 for tall, and 5x5 for square-ish.
        
        if w_init > 1.2:
            nx, ny = 7, 3
        elif w_init < 0.8:
            nx, ny = 3, 7
        else:
            nx, ny = 5, 5 # 25 cells, we will use 21 of them? Or just 4x6=24?
            # Let's stick to 21 cells exactly to avoid confusion.
            # If 5x5, we have 4 empty. Which ones? Corner ones are bad for packing.
            # Let's just use a 3x7 grid always? It might be bad for square aspect ratio.
            # Let's try 4x6 grid (24 cells) and select 21?
            # Simpler: Just use a 5x5 grid and place 21 circles in the first 21 cells (row-major).
            # This leaves the last 4 cells (bottom right) empty. This is asymmetric.
            
        # Let's refine initialization:
        # We want 21 circles.
        # If we use a 3x7 grid, it's exactly 21.
        # If we use a 4x6 grid, it's 24. We can drop 3.
        # Let's just use 3x7 for wide, 7x3 for tall, and for square, let's use 5x5 and drop 4 corners?
        # Actually, a 4x5 grid (20) is too small.
        # Let's just use a 3x7 grid for all, but adjust the cell sizes to fit the rectangle?
        # No, the cell sizes are determined by w and h.
        
        # Let's define the initial positions based on a 3x7 grid (21 cells)
        # This assumes the rectangle can be divided into 3 rows and 7 cols.
        # This works for any aspect ratio, the cells just have different aspect ratios.
        
        x_coords = []
        y_coords = []
        r_coords = []
        
        # 3 rows, 7 cols
        rows, cols = 3, 7
        for i in range(rows):
            for j in range(cols):
                cx = (j + 0.5) * (w_init / cols)
                cy = (i + 0.5) * (h_init / rows)
                # Initial radius: half the smaller dimension of the cell, with margin
                cell_w = w_init / cols
                cell_h = h_init / rows
                r_init = 0.5 * min(cell_w, cell_h) * 0.9 # 10% margin
                
                x_coords.append(cx)
                y_coords.append(cy)
                r_coords.append(r_init)
                
        x0 = np.array(x_coords + y_coords + r_coords + [w_init])
        
        # Define objective
        def objective(v):
            xs = v[0:N]
            ys = v[N:2*N]
            rs = v[2*N:3*N]
            w = v[3*N]
            h = 2.0 - w
            return -np.sum(rs)
            
        # Define constraints
        # SLSQP expects g(v) >= 0
        def constraints(v):
            xs = v[0:N]
            ys = v[N:2*N]
            rs = v[2*N:3*N]
            w = v[3*N]
            h = 2.0 - w
            
            cons = []
            
            # Bounds for x, y, r
            for i in range(N):
                cons.append(xs[i] - rs[i])
                cons.append(w - xs[i] - rs[i])
                cons.append(ys[i] - rs[i])
                cons.append(h - ys[i] - rs[i])
                
            # Pairwise distances
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist_sq = dx*dx + dy*dy
                    sum_r = rs[i] + rs[j]
                    cons.append(dist_sq - sum_r*sum_r)
                    
            return cons
            
        # Bounds
        # x: [0, 2] (max width is 2 if h=0, but h>0 so w<2)
        # y: [0, 2]
        # r: [0, 1] (max radius is 1 if 1 circle in 2x2 square? No, perimeter 4 square is 1x1, r=0.5. Max r is 1 if 2x2? No, 2(w+h)=4. If w=2, h=0. Impossible. Max r is 0.5 for square. For very thin, r is small. 1.0 is safe upper bound)
        # w: [0.1, 1.9]
        
        bounds = []
        for i in range(N):
            bounds.append((0, 2)) # x
        for i in range(N):
            bounds.append((0, 2)) # y
        for i in range(N):
            bounds.append((1e-6, 1.0)) # r
        bounds.append((0.1, 1.9)) # w
        
        # Run SLSQP
        try:
            res = so.minimize(objective, x0, method='SLSQP', bounds=bounds, 
                              constraints=[{'type': 'ineq', 'fun': constraints}],
                              options={'maxiter': 200, 'ftol': 1e-8})
            
            if res.success or res.fun < best_val:
                # Check feasibility manually to be safe
                v = res.x
                xs = v[0:N]
                ys = v[N:2*N]
                rs = v[2*N:3*N]
                w = v[3*N]
                h = 2.0 - w
                
                # Check bounds
                if np.any(xs < -1e-5) or np.any(xs > w + 1e-5) or \
                   np.any(ys < -1e-5) or np.any(ys > h + 1e-5) or \
                   np.any(rs < 1e-6) or w <= 0.1 or w >= 1.9:
                    continue
                    
                # Check overlaps
                valid = True
                for i in range(N):
                    if xs[i] - rs[i] < -1e-5 or xs[i] + rs[i] - w > 1e-5:
                        valid = False
                        break
                    if ys[i] - rs[i] < -1e-5 or ys[i] + rs[i] - h > 1e-5:
                        valid = False
                        break
                    for j in range(i+1, N):
                        dx = xs[i] - xs[j]
                        dy = ys[i] - ys[j]
                        if dx*dx + dy*dy < (rs[i] + rs[j])**2 - 1e-5:
                            valid = False
                            break
                    if not valid:
                        break
                            
                if valid:
                    val = -res.fun
                    if val > best_val:
                        best_val = val
                        best_x = v
                        
        except Exception as e:
            pass # Ignore errors and try next aspect ratio
            
    # Format output
    if best_x is None:
        # Fallback: simple grid
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N: break
                cells.append([(i+0.5)/side, (j+0.5)/side, 0.5/side - 1e-3])
        return np.array(cells)
        
    xs = best_x[0:N]
    ys = best_x[N:2*N]
    rs = best_x[2*N:3*N]
    
    result = np.column_stack((xs, ys, rs))
    return result
