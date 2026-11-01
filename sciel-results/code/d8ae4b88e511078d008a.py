"""
Simulated Annealing with Soft-Min Objective for Circle Packing.
Strategy:
1. Initialize with a hexagonal grid.
2. Use Simulated Annealing to maximize the packing radius.
   Objective: Minimize a smooth approximation of the "inverse minimum distance".
   f(x) = sum( exp(-alpha * d_ij) ) + wall_penalties
   Minimizing this pushes all distances d_ij to be large.
3. Polish the best found configuration with L-BFGS-B on the same objective.
"""

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(cents):
    """Compute the packing radius (the score)."""
    n = len(cents)
    # Wall distances
    # Distance to wall is min(0.5 - |x|, 0.5 - |y|)
    # We want to maximize the minimum of these.
    wall_dists = np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1]))
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    if n > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
    else:
        min_pair = 0.5
        
    return min(min_wall, min_pair)

def _construct_hexagonal():
    """Create a hexagonal packing arrangement for 50 circles."""
    best_cents = None
    best_r = 0

    for n_rows in [5, 6, 7, 8, 9, 10]:
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]

        max_rc = max(row_counts)
        # Estimate radius for this layout
        # Horizontal: 2r * max_rc <= 1  => r <= 1/(2*max_rc)
        # Vertical: r + (n_rows-1)*r*sqrt(3) + r <= 1 => r(2 + (n_rows-1)sqrt(3)) <= 1
        r_horiz = 1.0 / (2.0 * max_rc)
        r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
        r_est = min(r_horiz, r_vert)
        
        if r_est <= best_r:
            continue

        centers = []
        # Calculate total height to center vertically
        # y_spacing = r * sqrt(3)
        # total_height = r + (n_rows-1)*r*sqrt(3) + r = r(2 + (n_rows-1)sqrt(3))
        # We want to center this in [-0.5, 0.5]
        
        # Use r_est for layout
        r = r_est
        y_spacing = r * np.sqrt(3.0)
        total_height = 2*r + (n_rows - 1) * y_spacing
        y_start = -0.5 + r + (1.0 - total_height) / 2.0 # Center vertically
        
        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * y_spacing
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + (1.0 - x_span) / 2.0 # Center horizontally
            
            # Offset odd rows
            offset = y_spacing * 0.5 if row_idx % 2 == 1 else 0.0
            
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                centers.append([x, y])

        centers = np.array(centers[:N])
        # Clip to ensure validity (though hex should be inside)
        centers = np.clip(centers, -0.5, 0.5)
        
        r_actual = _score(centers)
        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()

    if best_cents is None:
        # Fallback grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
        best_r = _score(best_cents)

    return best_cents, best_r

def _soft_min_objective(x, alpha):
    """
    Objective function for Simulated Annealing.
    Minimizes a sum of exponential penalties for close pairs and wall violations.
    This is smooth and differentiable.
    """
    c = x.reshape(N, 2)
    
    # Wall penalties: 
    # We want distance to wall >= r. 
    # But we are maximizing r. 
    # A common trick: minimize sum of exp(-alpha * d_wall) 
    # where d_wall is distance to nearest wall.
    # If d_wall is small, exp is large -> penalty.
    # If d_wall is large, exp is small -> low cost.
    
    wall_dists = np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1]))
    # To avoid issues with negative distances (if out of bounds), we penalize out-of-bounds heavily
    # But SA moves are usually small. Let's assume mostly in bounds.
    # Actually, let's just use the raw distance. If it goes negative, exp(-alpha*neg) = exp(pos) -> huge penalty.
    wall_cost = np.sum(np.exp(-alpha * wall_dists))
    
    # Pairwise penalties
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # We only care about the closest pairs. 
    # Summing exp(-alpha * d) over all pairs works.
    pair_cost = np.sum(np.exp(-alpha * dists))
    
    return wall_cost + pair_cost

def _sa_optimize(x0, start_time, deadline):
    """
    Simulated Annealing to maximize the packing radius.
    """
    x = x0.copy()
    best_x = x.copy()
    best_score = _score(x.reshape(N, 2))
    
    # Parameters
    # Alpha controls the "sharpness" of the soft-min. 
    # Higher alpha focuses more on the minimum distance.
    # We can anneal alpha as well, or keep it fixed.
    # Let's start with a moderate alpha and increase it.
    alpha = 50.0 
    
    T = 1.0 # Initial temperature
    T_end = 1e-8
    cooling_rate = 0.999
    
    # Step size
    step_scale = 0.05
    
    current_cost = _soft_min_objective(x, alpha)
    
    while time.time() < deadline - 10:
        if T < T_end:
            break
            
        # Generate a neighbor
        # Random move in 2D
        move = np.random.normal(0, step_scale, size=(N, 2))
        # Occasionally do a larger "jump" to escape local minima
        if np.random.random() < 0.05:
            move *= 5.0
            
        x_new = x + move
        # Keep within bounds [-0.5, 0.5] roughly? 
        # The objective function penalizes going out of bounds, so we don't strictly need to clip,
        # but clipping helps stability.
        x_new = np.clip(x_new, -0.6, 0.6)
        
        cost_new = _soft_min_objective(x_new, alpha)
        
        delta_cost = cost_new - current_cost
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta_cost < 0 or np.random.random() < np.exp(-delta_cost / T):
            x = x_new
            current_cost = cost_new
            
            # Check actual score (expensive, so do it periodically or if cost improved significantly)
            # To save time, we only compute exact score if the soft-min cost improved by a margin
            # or every K iterations.
            # Let's compute score every 100 accepted moves or if cost < best_cost_recorded
            if not hasattr(_sa_optimize, '_count'):
                _sa_optimize._count = 0
            _sa_optimize._count += 1
            
            if _sa_optimize._count % 50 == 0:
                score_new = _score(x.reshape(N, 2))
                if score_new > best_score:
                    best_score = score_new
                    best_x = x.copy()
        
        # Cool down
        T *= cooling_rate
        # Reduce step size slightly to refine
        step_scale *= 0.9995
        
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # 1. Initial Hexagonal Layout
    cents, r0 = _construct_hexagonal()
    best_cents = cents.copy()
    best_r = r0
    
    # 2. Simulated Annealing
    # We run SA for a significant portion of the time
    sa_deadline = start_time + 100.0 # Allow 100s for SA
    # If we have less time, adjust
    if time.time() + 20 > deadline:
         sa_deadline = deadline - 20
         
    x_flat = best_cents.reshape(-1)
    sa_cents, sa_r = _sa_optimize(x_flat, start_time, sa_deadline)
    
    if sa_r > best_r:
        best_r = sa_r
        best_cents = sa_cents.copy()
        
    # 3. Final Polish with L-BFGS-B
    # Use the soft-min objective with a high alpha to sharpen the minimum
    # Or use the standard violation loss with a target r slightly above best_r?
    # Better: Minimize the soft-min objective directly with L-BFGS-B.
    # This is smooth and L-BFGS-B handles it well.
    
    if time.time() < deadline - 10:
        alpha_final = 100.0
        x0 = best_cents.reshape(-1)
        
        def obj(x):
            return _soft_min_objective(x, alpha_final)
            
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            res = minimize(
                obj, x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            polish_cents = res.x.reshape(N, 2)
            polish_r = _score(polish_cents)
            
            if polish_r > best_r:
                best_r = polish_r
                best_cents = polish_cents.copy()
        except Exception:
            pass # If optimization fails, keep SA result

    # Final safety check
    best_cents = np.clip(best_cents, -0.5, 0.5)
    
    return best_cents

# EVOLVE-BLOCK-END
