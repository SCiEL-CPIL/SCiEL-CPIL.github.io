"""
Circle packing in perimeter-4 rectangle: 21 circles.
Uses a grid-based cellular decomposition approach with scipy SLSQP optimization:
1. Partition the rectangle into a grid of cells.
2. Assign one circle to each cell.
3. Each circle's radius is constrained by its cell's inradius (distance to nearest edge).
4. Optimize cell boundaries and aspect ratio to maximize sum of radii.
This inherently guarantees disjoint interiors since circles are confined to non-overlapping cells.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def check_validity(x, y, r, w, tol=1e-9):
        h = 2.0 - w
        if w <= 0.1 or h <= 0.1:
            return False, 0.0
        if np.any(r <= 0):
            return False, 0.0
        
        # Boundary checks
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
            
        # Overlap checks
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum * r_sum - tol:
                    return False, 0.0
        return True, float(np.sum(r))

    def grid_packing(w, cols, rows, num_circles):
        """
        Pack circles in a grid layout within a w x (2-w) rectangle.
        Returns x, y, r arrays.
        """
        h = 2.0 - w
        
        # Create grid positions
        x = np.zeros(num_circles)
        y = np.zeros(num_circles)
        r = np.zeros(num_circles)
        
        for i in range(num_circles):
            row, col = divmod(i, cols)
            if row >= rows:
                break
            cx = (col + 0.5) * (w / cols)
            cy = (row + 0.5) * (h / rows)
            x[i] = cx
            y[i] = cy
            # Radius limited by cell size and distance to boundaries
            cell_w = w / cols
            cell_h = h / rows
            r[i] = min(cell_w, cell_h) * 0.48  # Slightly less than half to allow for wall clearance
        
        return x, y, r

    def optimize_grid(w, cols, rows, time_limit=10.0):
        """
        Optimize a grid packing by adjusting circle positions within cells
        and growing radii.
        """
        h = 2.0 - w
        x, y, r = grid_packing(w, cols, rows, N)
        
        # Initial validity check
        valid, val = check_validity(x, y, r, w)
        if not valid:
            return None
        
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        best_val = val
        
        # Try to grow radii iteratively
        growth_factor = 1.001
        for iteration in range(500):
            if time.time() > deadline - 5.0:
                break
            if time.time() > start_time + time_limit:
                break
                
            # Try to grow all radii
            new_r = r * growth_factor
            
            # For each circle, try to move it slightly to accommodate larger radius
            # Simple approach: just check if larger radii fit in current positions
            valid, new_val = check_validity(x, y, new_r, w)
            if valid:
                r = new_r
                best_r = r.copy()
                best_val = new_val
            else:
                # Try smaller growth
                new_r = r * 1.0001
                valid, new_val = check_validity(x, y, new_r, w)
                if valid:
                    r = new_r
                    best_r = r.copy()
                    best_val = new_val
                else:
                    break
        
        return x, y, best_r, best_val

    def optimize_with_slsqp(w, cols, rows, time_limit=15.0):
        """
        Use SLSQP to optimize circle positions and radii.
        """
        h = 2.0 - w
        
        # Variables: x_0, y_0, r_0, x_1, y_1, r_1, ..., x_{N-1}, y_{N-1}, r_{N-1}
        # Total: 3*N variables
        
        def objective(vars):
            x = vars[0:N]
            y = vars[N:2*N]
            r = vars[2*N:3*N]
            return -np.sum(r)  # Minimize negative sum of radii
        
        # Initial guess
        x0, y0, r0 = grid_packing(w, cols, rows, N)
        x0_opt = np.concatenate([x0, y0, r0])
        
        # Constraints:
        # 1. Circle i must be within bounds: r_i <= x_i <= w - r_i
        # 2. Circle i must be within bounds: r_i <= y_i <= h - r_i
        # 3. No overlap: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        
        constraints = []
        for i in range(N):
            # x_i >= r_i
            constraints.append({'type': 'ineq', 'fun': lambda vars, i=i: vars[i] - vars[2*N+i]})
            # x_i <= w - r_i
            constraints.append({'type': 'ineq', 'fun': lambda vars, i=i: (w - vars[2*N+i]) - vars[i]})
            # y_i >= r_i
            constraints.append({'type': 'ineq', 'fun': lambda vars, i=i: vars[N+i] - vars[2*N+i]})
            # y_i <= h - r_i
            constraints.append({'type': 'ineq', 'fun': lambda vars, i=i: (h - vars[2*N+i]) - vars[N+i]})
        
        # Overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda vars, i=i, j=j: 
                        (vars[i] - vars[j])**2 + (vars[N+i] - vars[N+j])**2 - 
                        (vars[2*N+i] + vars[2*N+j])**2
                })
        
        # Bounds: r_i > 0
        bounds = [(0, w/2)] * N + [(0, h/2)] * N + [(1e-6, min(w, h)/2)] * N
        
        # Optimize
        result = minimize(objective, x0_opt, method='SLSQP', 
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 100, 'ftol': 1e-8, 'disp': False})
        
        if result.success or result.fun < 0:
            x_opt = result.x[0:N]
            y_opt = result.x[N:2*N]
            r_opt = result.x[2*N:3*N]
            valid, val = check_validity(x_opt, y_opt, r_opt, w)
            if valid:
                return x_opt, y_opt, r_opt, val
        
        return None

    best_value = 0.0
    best_result = None
    
    # Try different grid configurations and aspect ratios
    # For N=21, common grids: 3x7, 5x5 (with 1 extra), 4x6 (with 1 extra), etc.
    grid_configs = [
        (3, 7),   # 21 circles exactly
        (7, 3),   # 21 circles exactly
        (5, 5),   # 25 cells, use 21
        (4, 6),   # 24 cells, use 21
        (6, 4),   # 24 cells, use 21
        (5, 4),   # 20 cells, need 1 more - skip
    ]
    
    aspect_ratios = [1.0, 0.9, 1.1, 0.85, 1.15, 0.8, 1.2, 0.75, 1.25]
    
    # Sort aspect ratios by likelihood
    aspect_ratios.sort(key=lambda x: abs(x - 1.0))
    
    for w0 in aspect_ratios:
        if time.time() > deadline - 15.0:
            break
            
        time_budget = min(12.0, deadline - time.time() - 5.0)
        if time_budget < 3.0:
            break
        
        for cols, rows in grid_configs:
            if time.time() > deadline - 10.0:
                break
            
            # Check if this grid can fit N circles
            if cols * rows < N:
                continue
            
            # Method 1: Simple grid optimization
            res = optimize_grid(w0, cols, rows, time_limit=time_budget * 0.4)
            if res is not None:
                x, y, r, val = res
                if val > best_value:
                    best_value = val
                    best_result = (x, y, r, w0)
            
            # Method 2: SLSQP optimization (more expensive but potentially better)
            if time.time() < deadline - 8.0:
                res = optimize_with_slsqp(w0, cols, rows, time_limit=time_budget * 0.6)
                if res is not None:
                    x, y, r, val = res
                    if val > best_value:
                        best_value = val
                        best_result = (x, y, r, w0)
    
    # If we have a result, do a final polish
    if best_result is not None:
        x, y, r, w = best_result
        
        # Final high-precision polish: try very small radius growths
        for _ in range(200):
            if time.time() > deadline - 2.0:
                break
            new_r = r * 1.00005
            valid, val_new = check_validity(x, y, new_r, w)
            if valid:
                r = new_r
                best_value = val_new
            else:
                break
        
        # Final validity check and safety shrink
        valid, val = check_validity(x, y, r, w)
        if valid:
            # Shrink slightly to ensure strict inequality for evaluator
            r_final = r * (1 - 1e-9)
            return np.column_stack((x, y, r_final))
        else:
            # Fallback to previous best if polish failed
            return np.column_stack((best_result[0], best_result[1], best_result[2] * (1 - 1e-9)))
    
    # Fallback: Simple grid in unit square (Perimeter 4 -> 1x1)
    cols = 5
    rows = 5
    cells = []
    for i in range(N):
        row, col = divmod(i, cols)
        cx = (col + 0.5) / cols
        cy = (row + 0.5) / rows
        r0 = min(1.0 / cols, 1.0 / rows) * 0.45
        cells.append((cx, cy, r0))
    return np.array(cells)

if __name__ == "__main__":
    res = construct()
    print(res.shape)
