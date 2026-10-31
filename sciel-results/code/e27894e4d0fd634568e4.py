"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 21

# EVOLVE-BLOCK-START

def _validate_and_clamp(x, n):
    """Clamp coordinates/radii to be valid (positive radii, circles inside rectangle)."""
    # x layout: [W, H, (cx, cy, r) * n]
    W = x[0]
    H = x[1]
    coords = x[2:].reshape(n, 3)
    # Enforce perimeter constraint: 2*(W+H) <= 4 => W+H <= 2
    s = W + H
    if s > 2.0 + 1e-9:
        # Scale W and H down proportionally
        scale = 2.0 / s
        W *= scale
        H *= scale
    # Clamp radii to be positive and circles inside [0,W]x[0,H]
    for i in range(n):
        cx, cy, r = coords[i]
        r = max(r, 1e-8)
        cx = min(max(cx, r), W - r)
        cy = min(max(cy, r), H - r)
        coords[i] = [cx, cy, r]
    return np.array([W, H]) , coords


def _objective_and_grad(x, n):
    """Objective: -sum(r). Returns (f, grad)."""
    W, H = x[0], x[1]
    coords = x[2:].reshape(n, 3)
    # Clamp for evaluation
    s = W + H
    if s > 2.0 + 1e-9:
        scale = 2.0 / s
        W *= scale
        H *= scale
    # Objective: maximize sum of radii => minimize -sum(r)
    radii = coords[:, 2]
    f = -np.sum(radii)
    # Gradient
    grad = np.zeros_like(x)
    grad[0] = 0.0
    grad[1] = 0.0
    # d(-sum r)/d(cx_i) = 0, d(-sum r)/d(cy_i) = 0, d(-sum r)/d(r_i) = -1
    for i in range(n):
        grad[2 + 3*i + 2] = -1.0
    return f, grad


def _constraints(n):
    """Build constraint list for SLSQP: non-overlap and boundary."""
    # Non-overlap: (cx_i - cx_j)^2 + (cy_i - cy_j)^2 >= (r_i + r_j)^2
    # Boundary: r_i <= cx_i <= W - r_i  =>  cx_i - r_i >= 0 and W - cx_i - r_i >= 0
    # Similarly for y.
    cons = []
    
    # Boundary constraints (inequality: g(x) >= 0)
    # For each circle i:
    #   cx_i - r_i >= 0
    #   W - cx_i - r_i >= 0
    #   cy_i - r_i >= 0
    #   H - cy_i - r_i >= 0
    # We'll handle these via bounds in the optimizer instead, but SLSQP doesn't have per-variable 
    # bounds that depend on other variables. So we use inequality constraints.
    
    # Actually, let's use a different approach: use bounds for cx in [0, W], cy in [0, H], r > 0
    # But W and H are variables. This is tricky.
    # Better: use inequality constraints for everything.
    
    # Non-overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            def non_overlap(args, i=i, j=j):
                W, H = args[0], args[1]
                coords = args[2:].reshape(n, 3)
                dx = coords[i, 0] - coords[j, 0]
                dy = coords[i, 1] - coords[j, 1]
                r_sum = coords[i, 2] + coords[j, 2]
                return dx*dx + dy*dy - r_sum*r_sum
            cons.append({'type': 'ineq', 'fun': non_overlap})
    
    # Boundary constraints: circle inside rectangle
    # cx_i - r_i >= 0
    for i in range(n):
        def left_bound(args, i=i):
            coords = args[2:].reshape(n, 3)
            return coords[i, 0] - coords[i, 2]
        cons.append({'type': 'ineq', 'fun': left_bound})
        
        def right_bound(args, i=i):
            W = args[0]
            coords = args[2:].reshape(n, 3)
            return W - coords[i, 0] - coords[i, 2]
        cons.append({'type': 'ineq', 'fun': right_bound})
        
        def bottom_bound(args, i=i):
            coords = args[2:].reshape(n, 3)
            return coords[i, 1] - coords[i, 2]
        cons.append({'type': 'ineq', 'fun': bottom_bound})
        
        def top_bound(args, i=i):
            H = args[1]
            coords = args[2:].reshape(n, 3)
            return H - coords[i, 1] - coords[i, 2]
        cons.append({'type': 'ineq', 'fun': top_bound})
    
    # Perimeter constraint: 2*(W+H) <= 4 => W+H <= 2
    def perimeter(args):
        return 2.0 - args[0] - args[1]
    cons.append({'type': 'ineq', 'fun': perimeter})
    
    return cons


def _initial_guess(W, H, n, rng):
    """Generate initial guess for n circles in rectangle W x H."""
    x = np.zeros(2 + 3*n)
    x[0] = W
    x[1] = H
    for i in range(n):
        cx = rng.uniform(0.1*W, 0.9*W)
        cy = rng.uniform(0.1*H, 0.9*H)
        r = rng.uniform(0.05*min(W,H), 0.15*min(W,H))
        x[2+3*i] = cx
        x[2+3*i+1] = cy
        x[2+3*i+2] = r
    return x


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_sum_r = -np.inf
    best_x = None
    
    rng = np.random.default_rng(42)
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 1.5, 2.0, 0.75, 1.25, 0.6, 1.8, 0.5, 2.5, 0.4]
    
    # Precompute constraints (they don't depend on W, H)
    cons = _constraints(N)
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        # W + H = 2 (use full perimeter)
        # H = W / ar, so W + W/ar = 2 => W(1 + 1/ar) = 2 => W = 2*ar/(1+ar)
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 / (1.0 + ar)
        
        # Multiple random restarts
        for restart in range(8):
            if time.time() > deadline:
                break
            
            x0 = _initial_guess(W, H, N, rng)
            
            # Bounds: W > 0, H > 0, cx, cy, r > 0
            # We'll use loose bounds and rely on constraints
            bounds = [(1e-6, 2.0), (1e-6, 2.0)] + [(1e-6, None)] * (3*N)
            
            try:
                result = minimize(
                    _objective_and_grad,
                    x0,
                    args=(N,),
                    method='SLSQP',
                    jac=True,
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 500, 'ftol': 1e-8}
                )
                
                if result.success or result.fun < -best_sum_r:
                    # Validate
                    x = result.x
                    W_r, H_r = x[0], x[1]
                    coords = x[2:].reshape(N, 3)
                    
                    # Check validity
                    valid = True
                    if W_r <= 0 or H_r <= 0 or 2*(W_r+H_r) > 4.0 + 1e-6:
                        valid = False
                    
                    if valid:
                        # Check non-overlap
                        for i in range(N):
                            for j in range(i+1, N):
                                dx = coords[i,0] - coords[j,0]
                                dy = coords[i,1] - coords[j,1]
                                dist = np.sqrt(dx*dx + dy*dy)
                                if dist < coords[i,2] + coords[j,2] - 1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break
                    
                    if valid:
                        # Check boundary
                        for i in range(N):
                            cx, cy, r = coords[i]
                            if cx - r < -1e-6 or W_r - cx - r < -1e-6 or cy - r < -1e-6 or H_r - cy - r < -1e-6:
                                valid = False
                                break
                    
                    if valid:
                        sum_r = np.sum(coords[:, 2])
                        if sum_r > best_sum_r:
                            best_sum_r = sum_r
                            best_x = x.copy()
                
            except Exception:
                pass
    
    # If no valid solution found, return a simple grid
    if best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        W = 2.0 / (1.0 + 1.0)  # square
        H = W
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side * W
                cy = (j + 0.5) / side * H
                r = 0.5 / side * min(W, H) * 0.99
                cells.append([cx, cy, r])
        return np.array(cells[:N])
    
    # Extract final result
    W, H = best_x[0], best_x[1]
    coords = best_x[2:].reshape(N, 3)
    
    # Final clamp to ensure validity
    for i in range(N):
        cx, cy, r = coords[i]
        r = max(r, 1e-8)
        cx = min(max(cx, r), W - r)
        cy = min(max(cy, r), H - r)
        coords[i] = [cx, cy, r]
    
    return coords

# EVOLVE-BLOCK-END
