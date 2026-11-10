import numpy as np
from scipy.optimize import minimize
import time

def _union_area_at_c(x, y, c):
    """
    Compute the exact 2D union area of the 64 squares at height c.
    x: array of 8 x-base positions
    y: array of 8 y-base positions
    c: height in [0, 1]
    """
    N = 8
    frac = 1.0 / N
    
    # x0_i = x[i] + c*i/N
    # y0_j = y[j] + c*j/N
    x0s = x + c * np.arange(N) * frac
    y0s = y + c * np.arange(N) * frac
    
    # Get all unique x and y boundaries
    # Boundaries are x0s and x0s + frac
    xs = np.sort(np.unique(np.concatenate([x0s, x0s + frac])))
    ys = np.sort(np.unique(np.concatenate([y0s, y0s + frac])))
    
    area = 0.0
    # Iterate over grid cells
    # Number of cells is (len(xs)-1) * (len(ys)-1)
    # Max 15 * 15 = 225 cells
    for k in range(len(xs) - 1):
        cx0 = xs[k]
        cx1 = xs[k+1]
        for l in range(len(ys) - 1):
            cy0 = ys[l]
            cy1 = ys[l+1]
            # Check if this cell is covered by any square
            # A square (i,j) covers the cell if:
            # x0s[i] <= cx0 and x0s[i] + frac >= cx1
            # y0s[j] <= cy0 and y0s[j] + frac >= cy1
            # We can check this efficiently.
            # For a fixed cell, we need to see if there exists i,j such that the above holds.
            # i must satisfy: cx1 - frac <= x0s[i] <= cx0
            # j must satisfy: cy1 - frac <= y0s[j] <= cy0
            
            # Since N is small, a simple loop is fine.
            covered = False
            for i in range(N):
                if x0s[i] <= cx0 + 1e-12 and x0s[i] + frac >= cx1 - 1e-12:
                    for j in range(N):
                        if y0s[j] <= cy0 + 1e-12 and y0s[j] + frac >= cy1 - 1e-12:
                            covered = True
                            break
                if covered:
                    break
            if covered:
                area += (cx1 - cx0) * (cy1 - cy0)
    return area

def _exact_volume(x, y):
    """
    Compute the exact 3D union volume by integrating the 2D area over c.
    """
    N = 8
    frac = 1.0 / N
    
    # Find all breakpoints in c where the order of boundaries changes.
    # Boundaries are x[i] + c*i/N and x[i] + c*i/N + 1/N
    # Breakpoints occur when two boundaries are equal.
    # x[i] + c*i/N = x[j] + c*j/N  => c = N*(x[i]-x[j])/(i-j)
    # x[i] + c*i/N = x[j] + c*j/N + 1/N => c = N*(x[i]-x[j]-1/N)/(i-j)
    # etc.
    
    breakpoints = set([0.0, 1.0])
    
    # Generate all possible c values where x-boundaries cross
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # x[i] + c*i/N = x[j] + c*j/N
            denom = (i - j) / N
            if abs(denom) > 1e-12:
                c1 = (x[j] - x[i]) / denom
                if 0 <= c1 <= 1:
                    breakpoints.add(c1)
                
                # x[i] + c*i/N = x[j] + c*j/N + 1/N
                c2 = (x[j] - x[i] - 1/N) / denom
                if 0 <= c2 <= 1:
                    breakpoints.add(c2)
                
                # x[i] + c*i/N + 1/N = x[j] + c*j/N
                c3 = (x[j] - x[i] + 1/N) / denom
                if 0 <= c3 <= 1:
                    breakpoints.add(c3)
                
                # x[i] + c*i/N + 1/N = x[j] + c*j/N + 1/N
                c4 = (x[j] - x[i]) / denom # Same as c1
                if 0 <= c4 <= 1:
                    breakpoints.add(c4)
    
    # Generate all possible c values where y-boundaries cross
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            denom = (i - j) / N
            if abs(denom) > 1e-12:
                c1 = (y[j] - y[i]) / denom
                if 0 <= c1 <= 1:
                    breakpoints.add(c1)
                
                c2 = (y[j] - y[i] - 1/N) / denom
                if 0 <= c2 <= 1:
                    breakpoints.add(c2)
                
                c3 = (y[j] - y[i] + 1/N) / denom
                if 0 <= c3 <= 1:
                    breakpoints.add(c3)
                
                c4 = (y[j] - y[i]) / denom
                if 0 <= c4 <= 1:
                    breakpoints.add(c4)
    
    # Sort breakpoints
    bps = sorted(list(breakpoints))
    
    # Integrate area over each interval
    total_vol = 0.0
    for k in range(len(bps) - 1):
        c_start = bps[k]
        c_end = bps[k+1]
        if c_end <= c_start:
            continue
        # The area is linear in c on this interval, so we can use the trapezoidal rule
        # which is exact for linear functions.
        a_start = _union_area_at_c(x, y, c_start)
        a_end = _union_area_at_c(x, y, c_end)
        total_vol += 0.5 * (a_start + a_end) * (c_end - c_start)
        
    return total_vol

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    N = 8
    deadline = 590.0 # Leave 10s margin
    
    best_params = None
    best_vol = float('inf')
    
    # Use multiple random restarts
    n_restarts = 10
    np.random.seed(42)
    
    for restart in range(n_restarts):
        if time.time() - start_time > deadline:
            break
            
        # Random initial guess
        # We want the tubes to be spread out to minimize overlap.
        # A good initial guess is to space them out.
        x0 = np.sort(np.random.uniform(0, 1, N))
        y0 = np.sort(np.random.uniform(0, 1, N))
        
        # We can also try a deterministic spread
        if restart == 0:
            x0 = np.linspace(0, 1-1/N, N)
            y0 = np.linspace(0, 1-1/N, N)
            
        params0 = np.concatenate([x0, y0])
        
        # Objective function
        def obj(params):
            x = params[:N]
            y = params[N:]
            return _exact_volume(x, y)
        
        # Optimize
        try:
            res = minimize(obj, params0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success and res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x
        except Exception as e:
            pass
            
        # Check time
        if time.time() - start_time > deadline:
            break
            
    # If no valid params found, return a default
    if best_params is None:
        x0 = np.linspace(0, 1-1/N, N)
        y0 = np.linspace(0, 1-1/N, N)
        best_params = np.concatenate([x0, y0])
        
    x_opt = best_params[:N]
    y_opt = best_params[N:]
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions

