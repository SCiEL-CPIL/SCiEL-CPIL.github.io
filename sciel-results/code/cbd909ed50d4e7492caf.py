"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def _perimeter(w, h):
    return 2.0 * (w + h)

def _constraints_for_r(w, h, r):
    """Compute constraints: circle i must be inside [0,w]x[0,h] and disjoint from others.
    Returns array of constraint values (>=0 means satisfied)."""
    n = len(r)
    cons = []
    for i in range(n):
        # x >= r_i
        cons.append(r[i] - 0)  # placeholder, we'll do this in the objective/constraints properly
    # Actually, let's structure this differently.
    # Variables: x0, y0, r0, x1, y1, r1, ..., x20, y20, r20
    # Constraints:
    # 1) r_i >= 0 (handled by bounds)
    # 2) x_i >= r_i  => x_i - r_i >= 0
    # 3) y_i >= r_i  => y_i - r_i >= 0
    # 4) x_i + r_i <= w => w - x_i - r_i >= 0
    # 5) y_i + r_i <= h => h - y_i - r_i >= 0
    # 6) For i < j: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    #    => (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    def constraint_func(vars):
        vals = []
        for i in range(n):
            xi = vars[3*i]
            yi = vars[3*i+1]
            ri = vars[3*i+2]
            vals.append(xi - ri)
            vals.append(yi - ri)
            vals.append(w - xi - ri)
            vals.append(h - yi - ri)
        for i in range(n):
            for j in range(i+1, n):
                xi = vars[3*i]
                yi = vars[3*i+1]
                ri = vars[3*i+2]
                xj = vars[3*j]
                yj = vars[3*j+1]
                rj = vars[3*j+2]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                sum_r = ri + rj
                vals.append(dist_sq - sum_r**2)
        return np.array(vals)
    
    return constraint_func

def _objective(vars, w, h):
    """Negative sum of radii."""
    r = vars[2::3]
    return -np.sum(r)

def _refine(w, h, x0, y0, r0, time_budget):
    """Use scipy minimize with constraints to maximize sum of radii."""
    n = len(r0)
    vars0 = np.zeros(3*n)
    for i in range(n):
        vars0[3*i] = x0[i]
        vars0[3*i+1] = y0[i]
        vars0[3*i+2] = r0[i]
    
    # Bounds: r_i >= 0, x_i, y_i in [0, w+h] (loose)
    bounds = []
    for i in range(n):
        bounds.append((0, w + h))  # x
        bounds.append((0, w + h))  # y
        bounds.append((1e-10, max(w, h)/2))  # r
    
    constraint_func = _constraints_for_r(w, h, r0)
    
    constraints = {'type': 'ineq', 'fun': constraint_func}
    
    start_time = time.time()
    timeout = time_budget
    
    def _callback(xk):
        if time.time() - start_time > timeout:
            raise Exception("timeout")
    
    try:
        result = minimize(
            _objective, vars0, args=(w, h),
            method='SLSQP',
            bounds=bounds,
            constraints=[constraints],
            options={'maxiter': 200, 'ftol': 1e-12, 'iprint': 0}
        )
        if result.success:
            new_vars = result.x
            new_x = new_vars[0::3]
            new_y = new_vars[1::3]
            new_r = new_vars[2::3]
            # Verify constraints are satisfied
            cvals = constraint_func(new_vars)
            if np.min(cvals) >= -1e-8:
                return new_x, new_y, new_r
    except Exception:
        pass
    
    return x0, y0, r0

def _grid_initial(w, h, n):
    """Create an initial grid packing of n circles."""
    # Try to find a grid layout
    best = None
    best_sum = 0
    
    for cols in range(1, int(np.sqrt(n)) + 3):
        rows = int(np.ceil(n / cols))
        if rows * cols < n:
            continue
        
        # Cell dimensions
        cw = w / cols
        ch = h / rows
        
        x = np.zeros(n)
        y = np.zeros(n)
        r = np.zeros(n)
        
        for i in range(n):
            row = i // cols
            col = i % cols
            x[i] = (col + 0.5) * cw
            y[i] = (row + 0.5) * ch
            r[i] = min(cw, ch) / 2 - 1e-6
        
        # Check if circles are disjoint
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dist < r[i] + r[j] - 1e-9:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            s = np.sum(r)
            if s > best_sum:
                best_sum = s
                best = (x.copy(), y.copy(), r.copy())
    
    if best is None:
        # Fallback: single row
        x = np.array([(i + 0.5) * w / n for i in range(n)])
        y = np.array([h / 2] * n)
        r = np.array([min(w/n, h) / 2 - 1e-6] * n)
        best = (x, y, r)
    
    return best

def construct():
    start_time = time.time()
    deadline = 150 - 5  # 5 second safety margin
    
    best_score = 0
    best_result = None
    
    # Try several aspect ratios
    aspect_ratios = [1.0, 1.5, 2.0, 0.75, 1.25, 1.75, 2.5, 0.6, 1.1, 1.4]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
        
        # Perimeter = 2(w+h) = 4 => w+h = 2
        # w/h = ar => h = w/ar
        # w + w/ar = 2 => w(1 + 1/ar) = 2 => w = 2*ar/(ar+1)
        w = 2.0 * ar / (ar + 1.0)
        h = 2.0 / (ar + 1.0)
        
        # Initial grid packing
        x0, y0, r0 = _grid_initial(w, h, N)
        
        # Refine with SLSQP
        time_left = deadline - time.time()
        if time_left < 5:
            continue
        
        x1, y1, r1 = _refine(w, h, x0, y0, r0, time_left - 5)
        
        # Verify the result
        valid = True
        for i in range(N):
            if x1[i] < r1[i] - 1e-7 or x1[i] + r1[i] > w + 1e-7 or \
               y1[i] < r1[i] - 1e-7 or y1[i] + r1[i] > h + 1e-7:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((x1[i]-x1[j])**2 + (y1[i]-y1[j])**2)
                    if dist < r1[i] + r1[j] - 1e-7:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            score = np.sum(r1)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([x1, y1, r1])
        
        # Also try the initial grid if it's valid and better
        valid0 = True
        for i in range(N):
            if x0[i] < r0[i] - 1e-7 or x0[i] + r0[i] > w + 1e-7 or \
               y0[i] < r0[i] - 1e-7 or y0[i] + r0[i] > h + 1e-7:
                valid0 = False
                break
        if valid0:
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((x0[i]-x0[j])**2 + (y0[i]-y0[j])**2)
                    if dist < r0[i] + r0[j] - 1e-7:
                        valid0 = False
                        break
                if not valid0:
                    break
        if valid0:
            score0 = np.sum(r0)
            if score0 > best_score:
                best_score = score0
                best_result = np.column_stack([x0, y0, r0])
    
    if best_result is None:
        # Fallback: simple grid in a square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
