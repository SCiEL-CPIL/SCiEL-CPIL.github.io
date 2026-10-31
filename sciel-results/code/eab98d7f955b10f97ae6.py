"""Circle packing in perimeter-4 rectangle: SLSQP + Greedy Expansion Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-7):
    """Check if packing is valid."""
    if W <= 0.01 or H <= 0.01:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _max_radius_for_circle(i, x, y, r, W, H):
    """Calculate the maximum radius for circle i given current positions of others."""
    # Wall constraints
    max_r = min(x[i], W - x[i], y[i], H - y[i])
    
    # Neighbor constraints
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx * dx + dy * dy)
        # To not overlap, r_i + r_j <= d  =>  r_i <= d - r_j
        allowed = d - r[j]
        if allowed < max_r:
            max_r = allowed
            
    return max(0.0, max_r)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    def run_slsqp(x0, y0, r0, w0, h0, time_budget):
        """
        Run SLSQP optimization for a fixed aspect ratio (w0, h0).
        Variables: [x_0..x_N-1, y_0..y_N-1, r_0..r_N-1]
        Objective: -sum(r)
        Constraints: 
          1. Wall constraints: x_i >= r_i, x_i <= w - r_i, etc.
          2. Non-overlap: (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2
        """
        def objective(z):
            return -np.sum(z[2*N:])

        # Constraints are functions that return >= 0 for valid states
        # SLSQP requires c(z) >= 0
        cons = []
        
        # Wall constraints
        # x_i - r_i >= 0
        # x_i - (w - r_i) <= 0  =>  w - r_i - x_i >= 0
        # y_i - r_i >= 0
        # y_i - (h - r_i) <= 0  =>  h - r_i - y_i >= 0
        
        # We can vectorize these, but SLSQP takes a list of dicts or a single dict with 'fun' returning array.
        # Let's use a single 'eq'/'ineq' dict structure or list.
        # For efficiency, we define constraint functions inline.
        
        # Actually, SLSQP can take a list of constraint dicts.
        # But for 21 circles, we have 4*21 = 84 wall constraints and 210 overlap constraints.
        # Total 294 constraints. This might be slow for SLSQP if not vectorized well.
        
        # Let's define a single constraint function that returns a vector of values.
        def constraint_fun(z):
            x = z[:N]
            y = z[N:2*N]
            r = z[2*N:]
            
            c_vals = np.zeros(4*N + N*(N-1)//2)
            idx = 0
            
            # Walls
            for i in range(N):
                c_vals[idx] = x[i] - r[i]          # Left
                idx += 1
                c_vals[idx] = (w0 - r[i]) - x[i]   # Right
                idx += 1
                c_vals[idx] = y[i] - r[i]          # Bottom
                idx += 1
                c_vals[idx] = (h0 - r[i]) - y[i]   # Top
                idx += 1
            
            # Overlaps
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    # d^2 - (r_i+r_j)^2 >= 0
                    c_vals[idx] = dx*dx + dy*dy - (r[i] + r[j])**2
                    idx += 1
                    
            return c_vals

        cons_dict = {'type': 'ineq', 'fun': constraint_fun}
        
        # Bounds: r > 0, x,y within reasonable range
        bounds = []
        for i in range(N):
            bounds.append((0.0, w0)) # x
        for i in range(N):
            bounds.append((0.0, h0)) # y
        for i in range(N):
            bounds.append((1e-4, min(w0, h0)/2)) # r
            
        # Initial guess
        z0 = np.concatenate([x0, y0, r0])
        
        try:
            res = minimize(
                objective, 
                z0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons_dict,
                options={'maxiter': 500, 'ftol': 1e-8, 'disp': False}
            )
            
            if res.success or res.fun < 0: # Even if not fully successful, if valid and better, keep
                x_opt = res.x[:N]
                y_opt = res.x[N:2*N]
                r_opt = res.x[2*N:]
                
                # Check validity strictly
                if _is_valid(x_opt, y_opt, r_opt, w0, h0):
                    val = float(np.sum(r_opt))
                    return x_opt, y_opt, r_opt, w0, h0, val
        except Exception:
            pass
            
        return None

    def polish(x, y, r, w, h, time_budget):
        """
        Greedy radial expansion polish.
        Iteratively increase each radius to its max possible value given neighbors.
        """
        improved = True
        iterations = 0
        max_iters = 1000
        
        while improved and iterations < max_iters:
            improved = False
            iterations += 1
            if time.time() > time_budget:
                break
                
            for i in range(N):
                # Calculate max possible radius for circle i
                max_r = _max_radius_for_circle(i, x, y, r, w, h)
                
                # If we can increase, do it
                if max_r > r[i] + 1e-8:
                    r[i] = max_r
                    improved = True
                    
        return x, y, r, w, h, float(np.sum(r))

    def get_initial_guesses():
        guesses = []
        # Aspect ratios to try
        aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.9, 1.1]
        
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3 or w0 < 0.3:
                continue
            
            # Grid initialization
            side = int(np.ceil(np.sqrt(N)))
            x0 = np.zeros(N)
            y0 = np.zeros(N)
            r0 = np.zeros(N)
            
            for i in range(N):
                row, col = divmod(i, side)
                x0[i] = (col + 0.5) / side * w0
                y0[i] = (row + 0.5) / side * h0
                # Initial radius estimate: half the cell size, slightly smaller
                cell_w = w0 / side
                cell_h = h0 / side
                r0[i] = 0.4 * min(cell_w, cell_h)
                
            guesses.append((x0, y0, r0, w0, h0))
            
        return guesses

    guesses = get_initial_guesses()
    
    # Allocate time
    # SLSQP can be slow. Let's give each aspect ratio a decent chunk of time.
    # We have ~140s. 7 guesses. ~20s each.
    time_per_guess = (deadline - start_time) / len(guesses)
    time_per_guess = min(time_per_guess, 25.0)
    
    for i, (x0, y0, r0, w0, h0) in enumerate(guesses):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
            
        remaining = len(guesses) - i
        budget = (deadline - 2.0 - elapsed) / remaining
        budget = max(3.0, min(budget, 30.0))
        
        try:
            res = run_slsqp(x0, y0, r0, w0, h0, budget)
            if res is not None:
                x_opt, y_opt, r_opt, w_opt, h_opt, val = res
                
                # Polish this result
                if time.time() < deadline - 5:
                    x_pol, y_pol, r_pol, w_pol, h_pol, val_pol = polish(
                        x_opt, y_opt, r_opt, w_opt, h_opt, deadline - 2.0
                    )
                    if _is_valid(x_pol, y_pol, r_pol, w_pol, h_pol):
                        if val_pol > best_value:
                            best_value = val_pol
                            best_result = (x_pol.copy(), y_pol.copy(), r_pol.copy(), w_pol)
                        # Update best_value with polished value even if not global best yet?
                        # No, we only update best_result if it's better.
                else:
                    if val > best_value:
                        best_value = val
                        best_result = (x_opt.copy(), y_opt.copy(), r_opt.copy(), w_opt)
                        
        except Exception:
            continue

    # If no result found via SLSQP, fallback to simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        w0 = 1.0
        h0 = 1.0
        cells = []
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side * w0
            cy = (row + 0.5) / side * h0
            r0 = 0.4 / side
            cells.append((cx, cy, r0))
        return np.array(cells)

    x, y, r, w = best_result
    return np.column_stack((x, y, r))
