import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

N = 21

def construct():
    deadline = time.time() + 140  # safety margin
    
    def evaluate(w, centers, radii):
        """Check validity and return sum of radii if valid."""
        h = 2.0 - w
        if h <= 0 or w <= 0:
            return -np.inf
        # Check bounds
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            if r < 0:
                return -np.inf
            if x < r - 1e-9 or x > w - r + 1e-9 or y < r - 1e-9 or y > h - r + 1e-9:
                return -np.inf
        # Check non-overlap
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                dist2 = dx*dx + dy*dy
                if dist2 < (radii[i]+radii[j])**2 - 1e-9:
                    return -np.inf
        return np.sum(radii)
    
    def make_initial(w_init):
        h_init = 2.0 - w_init
        # Grid initial placement
        cols = 5
        rows = 5
        x_positions = np.linspace(0.5*w_init/cols, 0.5*(w_init - w_init/cols), cols)
        y_positions = np.linspace(0.5*h_init/rows, 0.5*(h_init - h_init/rows), rows)
        centers = []
        for i in range(cols):
            for j in range(rows):
                centers.append([x_positions[i], y_positions[j]])
        centers = np.array(centers[:N])
        # Initial radii: small fraction of cell size
        cell_w = w_init / cols
        cell_h = h_init / rows
        r0 = 0.3 * min(cell_w, cell_h)
        radii = np.full(N, r0)
        return centers, radii
    
    def objective(params, w):
        """Penalty-based objective for fixed w."""
        h = 2.0 - w
        centers = params[:2*N].reshape(N, 2)
        radii = params[2*N:]
        
        # Objective: maximize sum of radii
        obj = -np.sum(radii)
        
        # Penalty for out-of-bounds
        penalty = 0.0
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            # x bounds
            if x < r:
                penalty += (r - x)**2
            if x > w - r:
                penalty += (x - (w-r))**2
            # y bounds
            if y < r:
                penalty += (r - y)**2
            if y > h - r:
                penalty += (y - (h-r))**2
            # radius non-negative
            if r < 0:
                penalty += r**2
        
        # Penalty for overlap
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                dist2 = dx*dx + dy*dy
                min_dist = radii[i] + radii[j]
                if dist2 < min_dist**2:
                    penalty += (min_dist**2 - dist2)**2
        
        return obj + 1000.0 * penalty
    
    def objective_full(params):
        """Objective with w as first parameter."""
        w = params[0]
        rest = params[1:]
        return objective(rest, w)
    
    best_score = -np.inf
    best_result = None
    
    # Try different initial aspect ratios
    w_inits = [1.0, 0.8, 1.2, 0.7, 1.3, 0.9, 1.1]
    
    for w_init in w_inits:
        if time.time() > deadline:
            break
        h_init = 2.0 - w_init
        if h_init <= 0.1:
            continue
            
        centers, radii = make_initial(w_init)
        params0 = np.concatenate([[w_init], centers.flatten(), radii])
        
        bounds = [(0.1, 1.9)] + [(0, 2.0)]*(2*N) + [(0.001, 1.0)]*N
        
        result = minimize(objective_full, params0, method='L-BFGS-B', 
                         bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if result.success or result.fun < best_score:
            w_opt = result.x[0]
            c_opt = result.x[1:1+2*N].reshape(N, 2)
            r_opt = result.x[1+2*N:]
            score = evaluate(w_opt, c_opt, r_opt)
            if score > best_score:
                best_score = score
                best_result = (w_opt, c_opt, r_opt)
    
    # Polish with tighter constraints
    if best_result is not None and time.time() < deadline:
        w_opt, c_opt, r_opt = best_result
        h_opt = 2.0 - w_opt
        
        def objective_polish(params, w, h):
            centers = params[:2*N].reshape(N, 2)
            radii = params[2*N:]
            obj = -np.sum(radii)
            penalty = 0.0
            for i in range(N):
                x, y = centers[i]
                r = radii[i]
                if x < r: penalty += (r-x)**2
                if x > w-r: penalty += (x-(w-r))**2
                if y < r: penalty += (r-y)**2
                if y > h-r: penalty += (y-(h-r))**2
                if r < 0: penalty += r**2
            for i in range(N):
                for j in range(i+1, N):
                    dx = centers[i,0]-centers[j,0]
                    dy = centers[i,1]-centers[j,1]
                    dist2 = dx*dx+dy*dy
                    md = radii[i]+radii[j]
                    if dist2 < md**2:
                        penalty += (md**2-dist2)**2
            return obj + 5000.0*penalty
        
        params0 = np.concatenate([c_opt.flatten(), r_opt])
        bounds = [(0, w_opt)]*N + [(0, h_opt)]*N + [(0.001, 1.0)]*N
        # Actually need separate bounds for x and y
        bounds = []
        for i in range(N):
            bounds.append((0, w_opt))  # x
        for i in range(N):
            bounds.append((0, h_opt))  # y
        for i in range(N):
            bounds.append((0.001, 1.0))  # r
        
        result2 = minimize(objective_polish, params0, args=(w_opt, h_opt), method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
        
        if result2.success:
            c_polish = result2.x[:2*N].reshape(N, 2)
            r_polish = result2.x[2*N:]
            score2 = evaluate(w_opt, c_polish, r_polish)
            if score2 > best_score:
                best_score = score2
                best_result = (w_opt, c_polish, r_polish)
    
    if best_result is None:
        # Fallback: simple grid
        w = 1.0
        h = 1.0
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                x = (i+0.5)*w/side
                y = (j+0.5)*h/side
                r = 0.5*min(w,h)/side - 1e-4
                cells.append([x, y, r])
        cells = cells[:N]
        return np.array(cells)
    
    w_opt, c_opt, r_opt = best_result
    result = np.column_stack([c_opt, r_opt])
    return result
