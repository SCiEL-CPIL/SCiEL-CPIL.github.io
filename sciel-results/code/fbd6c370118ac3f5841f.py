import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit

    def check_validity(x, y, r, w, h):
        """Check if the solution is valid (non-overlapping, within bounds)."""
        if np.any(x < r) or np.any(x > w - r) or np.any(y < r) or np.any(y > h - r):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - 1e-9:
                    return False
        return True

    def optimize_for_aspect(ar, max_time_per_run=10.0):
        """
        Optimize for a specific aspect ratio using SLSQP.
        We maximize sum(r) subject to non-overlap and boundary constraints.
        """
        h = 2.0 / (ar + 1.0)
        w = 2.0 * ar / (ar + 1.0)
        
        # Initial guess: Grid-like distribution
        # Arrange N circles in a grid pattern
        cols = int(np.ceil(np.sqrt(N * ar)))
        rows = int(np.ceil(N / cols))
        
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        r0 = np.zeros(N)
        
        # Distribute points
        for i in range(N):
            row = i // cols
            col = i % cols
            # Center in cell
            x0[i] = (col + 0.5) * (w / cols)
            y0[i] = (row + 0.5) * (h / rows)
            # Initial radius: half the minimum dimension of a cell, slightly shrunk
            r0[i] = 0.4 * min(w / cols, h / rows)
        
        # Ensure initial feasibility by shrinking if needed
        for _ in range(50):
            if check_validity(x0, y0, r0, w, h):
                break
            r0 *= 0.95
        
        if not check_validity(x0, y0, r0, w, h):
            return None, 0.0

        # Variables: x, y, r (3N variables)
        def objective(params):
            r = params[2::3]
            return -np.sum(r)  # Minimize negative sum

        def constraints(params):
            x = params[0::3]
            y = params[1::3]
            r = params[2::3]
            
            cons = []
            
            # Boundary constraints:
            # x[i] >= r[i]  => x[i] - r[i] >= 0
            # x[i] <= w - r[i] => w - x[i] - r[i] >= 0
            # y[i] >= r[i]  => y[i] - r[i] >= 0
            # y[i] <= h - r[i] => h - y[i] - r[i] >= 0
            
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
                cons.append({'type': 'ineq', 'fun': lambda p, i=i, w=w: w - p[3*i] - p[3*i+2]})
                cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
                cons.append({'type': 'ineq', 'fun': lambda p, i=i, h=h: h - p[3*i+1] - p[3*i+2]})
                
            # Non-overlap constraints:
            # (x[i] - x[j])^2 + (y[i] - y[j])^2 >= (r[i] + r[j])^2
            # Equivalent to: (x[i] - x[j])^2 + (y[i] - y[j])^2 - (r[i] + r[j])^2 >= 0
            
            for i in range(N):
                for j in range(i + 1, N):
                    idx_i, idx_j = 3*i, 3*j
                    def non_overlap(p, i=idx_i, j=idx_j):
                        dx = p[i] - p[j]
                        dy = p[i+1] - p[j+1]
                        dr = p[i+2] + p[j+2]
                        return dx*dx + dy*dy - dr*dr
                    cons.append({'type': 'ineq', 'fun': non_overlap})
                    
            return cons

        # Initial parameters
        p0 = np.concatenate([x0, y0, r0])
        
        # Bounds:
        # x in [0, w], y in [0, h], r in [0, min(w,h)/2]
        bounds = []
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((0, min(w, h) / 2))
            
        # Run SLSQP
        try:
            res = minimize(
                objective, p0, method='SLSQP', bounds=bounds, constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            
            if res.success:
                p_res = res.x
                x_res = p_res[0::3]
                y_res = p_res[1::3]
                r_res = p_res[2::3]
                
                if check_validity(x_res, y_res, r_res, w, h):
                    score = np.sum(r_res)
                    return np.column_stack((x_res, y_res, r_res)), score
            else:
                # Even if not "success", check if the result is valid and good
                p_res = res.x
                x_res = p_res[0::3]
                y_res = p_res[1::3]
                r_res = p_res[2::3]
                
                if check_validity(x_res, y_res, r_res, w, h):
                    score = np.sum(r_res)
                    return np.column_stack((x_res, y_res, r_res)), score
                    
        except Exception as e:
            pass
            
        return None, 0.0

    best_solution = None
    best_score = 0.0

    # Try a range of aspect ratios
    # Square is often good, but elongated might pack better for N=21
    aspect_ratios = [1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 0.8, 0.7, 0.5]
    
    # Shuffle to avoid bias if time runs out
    random.shuffle(aspect_ratios)
    
    for ar in aspect_ratios:
        if time.time() > deadline - 15:
            break
            
        sol, score = optimize_for_aspect(ar)
        
        if sol is not None and score > best_score:
            best_score = score
            best_solution = sol.copy()
            
    # If we have a solution, try a final polish with a different initial guess
    # or just return the best found.
    # The SLSQP is already quite strong. Let's do a quick local search on the best solution
    # to squeeze out a bit more.
    
    if best_solution is not None:
        x, y, r = best_solution[:, 0], best_solution[:, 1], best_solution[:, 2]
        w = 2.0 / (1.0 + 1.0) # Wait, we need to recover w,h from the aspect ratio used
        # Actually, we didn't store w,h in best_solution. We need to reconstruct or store them.
        # Let's modify the loop to store w,h.
        
    # Let's restart the loop logic to store w,h properly
    best_solution = None
    best_score = 0.0
    best_wh = None
    
    for ar in aspect_ratios:
        if time.time() > deadline - 15:
            break
            
        h = 2.0 / (ar + 1.0)
        w = 2.0 * ar / (ar + 1.0)
        
        sol, score = optimize_for_aspect(ar)
        
        if sol is not None and score > best_score:
            best_score = score
            best_solution = sol.copy()
            best_wh = (w, h)
            
    # Final Polish: Local search on the best solution
    if best_solution is not None and best_wh is not None:
        x = best_solution[:, 0].copy()
        y = best_solution[:, 1].copy()
        r = best_solution[:, 2].copy()
        w, h = best_wh
        
        # Try to grow each radius slightly
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(N):
                old_r = r[i]
                # Try to grow by 0.1%
                new_r = old_r * 1.001
                r[i] = new_r
                
                if check_validity(x, y, r, w, h):
                    improved = True
                    best_score = np.sum(r)
                else:
                    r[i] = old_r
                    
            if not improved:
                break
                
        best_solution = np.column_stack((x, y, r))

    # Fallback if optimization failed completely
    if best_solution is None:
        # Simple grid fallback
        side = 5
        x_list, y_list, r_list = [], [], []
        # 21 circles in 5x5 grid (25 spots, use 21)
        for i in range(5):
            for j in range(5):
                if len(x_list) >= N:
                    break
                x_list.append((j + 0.5) / 5.0 * 1.0) # Assuming w=1, h=1 for square
                y_list.append((i + 0.5) / 5.0 * 1.0)
                r_list.append(0.4 / 5.0)
        # Scale to perimeter 4 (w=1, h=1)
        best_solution = np.column_stack((x_list, y_list, r_list))
        
    return best_solution
