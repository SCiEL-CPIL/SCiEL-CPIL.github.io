import numpy as np
import time
from scipy.optimize import minimize

N = 21
TIME_LIMIT = 140  # seconds

def _hex_lattice_positions(n, w, h):
    """
    Generate positions for n circles on a hexagonal lattice scaled to fit in w x h.
    Returns (n, 2) array of positions.
    """
    # Hexagonal lattice parameters
    # Distance between centers in same row
    dx = 1.0
    # Distance between centers in adjacent rows
    dy = dx * np.sqrt(3) / 2
    
    # We want to fit n circles.
    # Let's estimate the number of rows and columns.
    # For hex packing, density is ~0.9069.
    # Area of rectangle = w*h.
    # Area per circle ~ pi * r^2.
    # But let's just generate a grid and trim.
    
    # Estimate rows
    # If aspect ratio is w/h, and hex packing...
    # Let's try to generate enough points and pick the best n?
    # Or just generate a grid of roughly sqrt(n) by sqrt(n).
    
    # Simple heuristic:
    # Number of columns ~ sqrt(n * w / h)
    cols = max(1, int(np.sqrt(n * w / h)))
    rows = max(1, int(np.ceil(n / cols)))
    
    # Generate points
    points = []
    for r in range(rows + 1):
        for c in range(cols + 1):
            x = c * dx
            y = r * dy
            if r % 2 != 0:
                x += dx / 2
            points.append([x, y])
            
    points = np.array(points)
    
    if len(points) < n:
        # Need more points, increase resolution
        # Just return grid for now if hex fails to generate enough
        # But let's try to scale
        pass
    
    # We have a set of candidate points. We need to select n of them that maximize the minimum distance?
    # Or just take the first n?
    # Taking the first n from a hex lattice is usually a good compact cluster.
    
    # Sort by distance from center of the cluster to keep them compact?
    # Or just take the first n generated (which fills row by row).
    
    if len(points) >= n:
        pts = points[:n]
    else:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(n)))
        xs = (np.arange(n) % side + 0.5) / side
        ys = (np.arange(n) // side + 0.5) / side
        pts = np.column_stack([xs, ys])
        # Scale to w, h
        pts[:, 0] *= w
        pts[:, 1] *= h
        return pts

    # Normalize pts to fit in [0, w] x [0, h]
    # Current extent:
    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)
    
    # If extent is 0, handle it
    if max_x - min_x < 1e-9:
        max_x = min_x + 1.0
    if max_y - min_y < 1e-9:
        max_y = min_y + 1.0
        
    scale_x = w / (max_x - min_x)
    scale_y = h / (max_y - min_y)
    
    # To preserve hexagonal shape somewhat, we might want to scale uniformly?
    # But the task allows any rectangle with perimeter 4.
    # The optimization will adjust w and h.
    # For the initial guess, let's scale to fit the box.
    # Actually, if we scale non-uniformly, the hex lattice becomes a rhombic lattice.
    # That's fine.
    
    pts = (pts - min_x) * scale_x
    pts[:, 1] = (pts[:, 1] - min_y) * scale_y
    
    # Center in the box?
    # The optimizer will move them.
    # Let's just return them.
    
    return pts

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_config = None
    
    # We will optimize w, h, and positions.
    # Perimeter constraint: 2*(w+h) <= 4  =>  w+h <= 2.
    # We can parameterize w and h such that w+h = 2 (assuming filling the rectangle is optimal).
    # Let w = 2 * t, h = 2 * (1-t) for t in (0, 1).
    # Or just optimize w and h with a penalty if w+h > 2.
    # Given the small N, the optimal rectangle is likely close to square or slightly rectangular.
    
    # Strategy:
    # 1. Try a few aspect ratios (t values).
    # 2. For each, initialize with a hex lattice scaled to that aspect ratio.
    # 3. Optimize positions (and maybe fine-tune w, h if we include them in vars).
    #    Let's include w, h in the variables for the final polish, but fix them for the main search
    #    to reduce dimensionality and help convergence.
    #    Actually, let's optimize w, h, x, y, r all together?
    #    Vars: [w, h, x1..xN, y1..yN, r1..rN] -> 2 + 3*N = 65 variables.
    #    Nelder-Mead in 65D is slow.
    #    Better: Fix w, h for the position/radius optimization, then do a small 2D optimization of w, h afterwards?
    #    Or just loop over w, h.
    
    # Let's loop over aspect ratios.
    # t = w / (w+h). Since w+h=2, w = 2t, h = 2(1-t).
    # t ranges from 0.3 to 0.7 (symmetric).
    
    aspect_ratios = np.linspace(0.3, 0.7, 15)
    
    for t in aspect_ratios:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        w = 2 * t
        h = 2 * (1 - t)
        
        # Initialize positions using hex lattice
        x0_pos = _hex_lattice_positions(N, w, h)
        
        # Initial radii: estimate based on grid spacing
        # For hex lattice with spacing dx, dy, the max radius is roughly min(dx, dy*sqrt(3)/2) / 2?
        # Actually, for hex packing, r = dx / 2.
        # Let's just use a small initial radius.
        r_init = np.full(N, 0.1)
        
        # Flatten variables: [x1, y1, ..., xN, yN, r1, ..., rN]
        # We fix w, h here.
        x0 = np.concatenate([x0_pos.flatten(), r_init])
        
        def objective(vars):
            x = vars[0:N]
            y = vars[N:2*N]
            r = vars[2*N:3*N]
            
            # Ensure radii are positive
            r = np.maximum(r, 0)
            
            # Sum of radii
            sum_r = np.sum(r)
            
            # Overlap Penalty
            overlap_pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    min_d = r[i] + r[j]
                    if d < min_d:
                        overlap_pen += (min_d - d)**2
            
            # Boundary Penalty
            bound_pen = 0.0
            for i in range(N):
                if x[i] < r[i]:
                    bound_pen += (r[i] - x[i])**2
                if x[i] > w - r[i]:
                    bound_pen += (x[i] - (w - r[i]))**2
                if y[i] < r[i]:
                    bound_pen += (r[i] - y[i])**2
                if y[i] > h - r[i]:
                    bound_pen += (y[i] - (h - r[i]))**2
            
            # Perimeter penalty (should be 0 if w+h=2, but just in case)
            # We fixed w, h so this is constant.
            
            penalty = 100.0 * (overlap_pen + bound_pen)
            
            return -sum_r + penalty
        
        # Run optimization
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7})
        except Exception:
            continue
            
        x_opt = res.x
        x = x_opt[0:N]
        y = x_opt[N:2*N]
        r = x_opt[2*N:3*N]
        
        # Post-processing: Ensure validity
        # Check constraints and shrink if necessary
        
        # 1. Boundary check
        for i in range(N):
            r[i] = min(r[i], x[i], w-x[i], y[i], h-y[i])
            if r[i] < 0:
                r[i] = 0
        
        # 2. Overlap check and shrink
        # Iterate a few times to resolve overlaps
        for _ in range(10):
            max_viol = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    min_d = r[i] + r[j]
                    if d < min_d:
                        viol = (min_d - d) / 2
                        if viol > max_viol:
                            max_viol = viol
            if max_viol <= 1e-8:
                break
            r = r - max_viol
            r = np.maximum(r, 0)
            
        if not np.all(r > 1e-6):
            continue
            
        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_config = np.column_stack((x, y, r))
            
    if best_config is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        w, h = 1.0, 1.0
        xs = (np.arange(N) % side + 0.5) / side * w
        ys = (np.arange(N) // side + 0.5) / side * h
        r = np.full(N, 0.5 / side * 0.9)
        return np.column_stack((xs, ys, r))
        
    return best_config
