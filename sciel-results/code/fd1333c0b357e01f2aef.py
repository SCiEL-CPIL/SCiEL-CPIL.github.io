"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint
import itertools

def _objective_and_grad(x):
    """
    x = [r_0, ..., r_20, w, h]  (23 variables)
    Objective: -sum(r) - 0.01*sum(r^2)  (maximize sum radii, penalize extreme values)
    """
    r = x[:N]
    w, h = x[N], x[N+1]
    
    # Perimeter constraint: 2*(w+h) <= 4 => w+h <= 2
    # We handle this via bounds or penalty. Let's use a large penalty if violated.
    # Actually, better to enforce w+h=2 in the optimization by fixing h=2-w?
    # No, aspect ratio is a variable. Let's keep w, h as variables and add penalty.
    # Or, simpler: parameterize by aspect ratio.
    # Let's stick to w, h and add a quadratic penalty for w+h > 2.
    
    # Circles centers:
    # c_i = (r_i, h - r_i)  for i=0..10 (bottom row)
    # c_i = (w - r_i, r_i)  for i=11..20 (top row)
    # Wait, this assumes specific positions. This is a very constrained topology.
    # Bottom row: y = r_i. x positions need to be determined.
    # Top row: y = h - r_i. x positions need to be determined.
    
    # This "fixed topology" assumption (10 bottom, 11 top, touching boundaries) is too restrictive
    # and might not be the optimal topology.
    # Let's try a more general approach: 
    # Variables: x_i, y_i, r_i for all i.
    # Objective: -sum(r_i)
    # Constraints:
    # 1. 2*(max(x_i+r_i) - min(x_i-r_i) + max(y_i+r_i) - min(y_i-r_i)) <= 4
    # 2. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    # 3. r_i >= 0
    
    # This is a general non-convex problem.
    # To make it tractable, we can fix the rectangle to be [0, w] x [0, h] with 2(w+h)=4 => h=2-w.
    # So w is the only rectangle variable.
    # Variables: x_0..x_20, y_0..y_20, r_0..r_20, w. Total 63 variables.
    # This is large but solvable with SLSQP or similar if we have a good start.
    
    # However, the directive suggests a "rigid body mechanics" or "contact graph" approach.
    # That is complex to implement from scratch correctly in a short time.
    
    # Alternative: Use a simpler heuristic + local refinement.
    # 1. Start with a grid-like packing.
    # 2. Use a gradient-based optimizer to adjust positions and radii to maximize sum radii
    #    subject to non-overlap and boundary constraints.
    
    # Let's define the variables as:
    # x = [x_0, ..., x_20, y_0, ..., y_20, r_0, ..., r_20, w]
    # h = 2 - w
    
    r = x[:N]
    # Wait, I need to define the variable layout clearly.
    # Let's use:
    # x[0:N] = x coords
    # x[N:2N] = y coords
    # x[2N:3N] = radii
    # x[3N] = w
    
    cx = x[0:N]
    cy = x[N:2*N]
    cr = x[2*N:3*N]
    w = x[3*N]
    h = 2.0 - w
    
    # Objective: maximize sum(cr)
    # Gradient: -1 for radii, 0 for others (initially, ignoring constraint gradients)
    # We need to handle constraints via the optimizer.
    
    # For the objective function, just return -sum(cr)
    return -np.sum(cr)

def _constraints(x):
    """
    Returns a list of constraints.
    """
    cx = x[0:N]
    cy = x[N:2*N]
    cr = x[2*N:3*N]
    w = x[3*N]
    h = 2.0 - w
    
    cons = []
    
    # Boundary constraints:
    # x_i >= r_i, x_i + r_i <= w
    # y_i >= r_i, y_i + r_i <= h
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*N+i]}) # x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*N] - (x[i] + x[2*N+i])}) # w - (x_i + r_i) >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[N+i] - x[2*N+i]}) # y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: (2.0 - x[3*N]) - (x[N+i] + x[2*N+i])}) # h - (y_i + r_i) >= 0
    
    # Non-overlap constraints:
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(N):
        for j in range(i+1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[i]-x[j])**2 + (x[N+i]-x[N+j])**2 - (x[2*N+i]+x[2*N+j])**2
            })
    
    return cons

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial guess: Grid packing
    # 5x5 grid = 25 cells, we need 21.
    # Let's use a 5x4 grid (20 circles) + 1 extra? Or 5x5 with 4 removed?
    # Better: 4x5 grid = 20 circles. Add 1 in the middle of a gap?
    # Or just 5x5 grid, take first 21.
    
    # Let's try a 5x4 arrangement (20 circles) and see if we can squeeze a 21st.
    # Or 4x5.
    # Let's start with a 5x5 grid of equal circles, then optimize.
    
    side = 5
    # Perimeter 4 => w+h=2.
    # If square, w=h=1.
    # Grid spacing 1/5 = 0.2. Radius 0.1.
    # Sum radii = 25 * 0.1 = 2.5.
    # But we only have 21 circles.
    
    # Let's initialize with 21 circles in a 5x5 grid (top-left 21 positions).
    x0 = np.zeros(3*N + 1)
    
    # Rectangle
    w0 = 1.0
    h0 = 1.0
    x0[3*N] = w0
    
    # Grid positions
    idx = 0
    for i in range(5):
        for j in range(5):
            if idx >= N:
                break
            cx = (i + 0.5) / 5.0
            cy = (j + 0.5) / 5.0
            r = 0.1
            x0[idx] = cx
            x0[N+idx] = cy
            x0[2*N+idx] = r
            idx += 1
    
    # Bounds
    # w in [0.5, 1.5] (aspect ratio 1:3 to 3:1)
    # r in [0.01, 0.5]
    # x, y in [0, 2]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 2.0)) # x
    for i in range(N):
        bounds.append((0.0, 2.0)) # y
    for i in range(N):
        bounds.append((0.01, 0.5)) # r
    bounds.append((0.5, 1.5)) # w
    
    # Use SLSQP
    # This might be slow. Let's try.
    
    try:
        res = minimize(
            _objective_and_grad,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=_constraints(x0),
            options={'maxiter': 1000, 'ftol': 1e-8}
        )
        
        if res.success:
            best_x = res.x
        else:
            best_x = x0
            
    except Exception as e:
        print(f"Optimization failed: {e}")
        best_x = x0
        
    # Check time
    if time.time() > deadline:
        # Return best found so far (which is just the initial grid if opt failed or timed out)
        # We need to construct the output array from best_x
        pass
        
    # Construct output
    cx = best_x[0:N]
    cy = best_x[N:2*N]
    cr = best_x[2*N:3*N]
    
    # Ensure validity
    # If any radius is too small or negative, clamp
    cr = np.maximum(cr, 0.001)
    
    # Check if circles are disjoint and within bounds
    # If not, fall back to a known valid configuration (grid)
    
    valid = True
    w = best_x[3*N]
    h = 2.0 - w
    
    # Check bounds
    for i in range(N):
        if cx[i] < cr[i] - 1e-6 or cx[i] + cr[i] > w + 1e-6:
            valid = False
            break
        if cy[i] < cr[i] - 1e-6 or cy[i] + cr[i] > h + 1e-6:
            valid = False
            break
    
    # Check overlaps
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                dist2 = (cx[i]-cx[j])**2 + (cy[i]-cy[j])**2
                rad_sum = cr[i] + cr[j]
                if dist2 < rad_sum**2 - 1e-6:
                    valid = False
                    break
            if not valid:
                break
        
    if not valid:
        # Fallback to simple grid
        side = 5
        cells = []
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx >= N:
                    break
                cx = (i + 0.5) / 5.0
                cy = (j + 0.5) / 5.0
                r = 0.5 / 5.0 - 1e-3
                cells.append([cx, cy, r])
                idx += 1
        return np.array(cells)
        
    return np.column_stack((cx, cy, cr))

# EVOLVE-BLOCK-END
