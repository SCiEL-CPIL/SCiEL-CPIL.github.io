"""Program for 200 equal circles in a unit square."""
import numpy as np
import time

N = 200
TIME_LIMIT = 145.0  # seconds, leaving margin before 150s kill


def _compute_radius(centres):
    """Compute the packing radius for a set of centres."""
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check bounds: centres must be inside [-0.5, 0.5]^2
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    # Distance to walls
    wall_dists = np.min(np.stack([
        0.5 - centres[:, 0],
        0.5 + centres[:, 0],
        0.5 - centres[:, 1],
        0.5 + centres[:, 1]
    ], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    if min_wall <= 0:
        return 0.0
    # Minimum pairwise distance
    # Use scipy if available for speed, else fallback
    try:
        from scipy.spatial.distance import cdist
        dists = cdist(centres, centres)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
    except ImportError:
        # Fallback: compute all pairwise distances
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centres[i] - centres[j])
                if d < min_dist:
                    min_dist = d
    return min(min_dist / 2.0, min_wall)


def _grid_packing():
    """Create a square grid packing with optimal dimensions for N points."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _hexagonal_packing():
    """Create a hexagonal packing pattern, optimized for N points."""
    best_centres = None
    best_r = -1
    
    # Try various numbers of rows
    for n_rows in range(10, 25):
        # For hexagonal packing, alternating rows have offset
        # Row 0: n_cols points, Row 1: n_cols - 1 points (or vice versa)
        # Let's use a pattern where we can fit N points
        
        # Try both orientations: start with full row or offset row
        for start_offset in [0, 1]:
            # Estimate columns needed
            # If start_offset=0: rows 0,2,4,... have n_cols, rows 1,3,5,... have n_cols-1
            # If start_offset=1: rows 0,2,4,... have n_cols-1, rows 1,3,5,... have n_cols
            
            # Let's try to find n_cols such that total points >= N
            # For simplicity, let's try a range of n_cols
            for n_cols in range(10, 25):
                # Calculate how many points fit in this configuration
                total_points = 0
                for row in range(n_rows):
                    if (row + start_offset) % 2 == 0:
                        total_points += n_cols
                    else:
                        total_points += n_cols - 1
                
                if total_points < N:
                    continue
                
                # Calculate spacing s
                # Horizontal: n_cols * s <= 1 (for full rows)
                # Vertical: (n_rows - 1) * s * sqrt(3)/2 + s <= 1 (approx)
                # Actually, vertical extent is (n_rows - 1) * s * sqrt(3)/2
                # And we need to center it
                
                # Width needed: n_cols * s for full rows
                # But offset rows are shifted by s/2, so they fit in same width if n_cols-1 points
                # Let's say s = 1.0 / n_cols (horizontal constraint)
                # Vertical: (n_rows - 1) * s * sqrt(3)/2 <= 1
                # So s <= 2 / ((n_rows - 1) * sqrt(3)) if n_rows > 1
                
                s_horiz = 1.0 / n_cols
                if n_rows > 1:
                    s_vert = 2.0 / ((n_rows - 1) * np.sqrt(3))
                else:
                    s_vert = 1.0
                
                s = min(s_horiz, s_vert)
                r = s / 2.0
                
                if r <= best_r:
                    continue
                
                # Create the grid
                centres = []
                y_start = -0.5 + s * np.sqrt(3) / 2  # Center vertically
                for row in range(n_rows):
                    y = y_start + row * s * np.sqrt(3) / 2
                    if (row + start_offset) % 2 == 0:
                        # Full row: n_cols points
                        x_start = -0.5 + s / 2
                        for col in range(n_cols):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centres.append((x, y))
                    else:
                        # Offset row: n_cols - 1 points
                        x_start = -0.5 + s  # Shift by s/2 from full row start
                        for col in range(n_cols - 1):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centres.append((x, y))
                
                centres = np.array(centres)
                if len(centres) < N:
                    continue
                
                centres = centres[:N]
                actual_r = _compute_radius(centres)
                if actual_r > best_r:
                    best_r = actual_r
                    best_centres = centres
    
    if best_centres is not None:
        return best_centres
    
    # Fallback to grid
    return _grid_packing()


def _local_optimization(centres, max_iters=2000, step_size=0.001):
    """Local optimization using gradient ascent on a smooth surrogate objective."""
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 5)  # Leave 5s buffer
    
    try:
        from scipy.spatial.distance import cdist
    except ImportError:
        cdist = None

    # We maximize a smooth surrogate: sum of squared distances to neighbors + sum of squared distances to walls
    # This encourages spreading out.
    # Gradient of sum_{j != i} ||x_i - x_j||^2 wrt x_i is 2 * sum_{j != i} (x_i - x_j)
    # Gradient of min_wall_i^2 is approx 2 * min_wall_i * grad(min_wall_i)
    
    current_step = step_size
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Calculate gradients
        # Pairwise repulsion
        if cdist is not None:
            # Compute pairwise distances and vectors
            # dists[i,j] is distance. We need sum of (x_i - x_j) for each i.
            # This is equivalent to N * x_i - sum_j x_j if we consider all j, but we exclude i.
            # Sum_{j != i} (x_i - x_j) = (N-1)*x_i - (sum_all_x - x_i) = N*x_i - sum_all_x.
            # Wait, this is the gradient of sum_{i,j} ||x_i - x_j||^2.
            # We want sum_{i<j} ||x_i - x_j||^2.
            # The gradient for x_i in sum_{i<j} is sum_{j != i} 2(x_i - x_j).
            # Using the identity: sum_{j != i} (x_i - x_j) = N x_i - sum_{all k} x_k.
            # So grad_pairwise_i = 2 * (N * x_i - sum_all_x).
            
            sum_all_x = np.sum(best_centres, axis=0)
            grad_pair = 2.0 * (N * best_centres - np.tile(sum_all_x, (N, 1)))
            
        else:
            grad_pair = np.zeros_like(best_centres)
            for i in range(N):
                for j in range(N):
                    if i != j:
                        grad_pair[i] += 2.0 * (best_centres[i] - best_centres[j])

        # Wall repulsion
        # For each point, find the nearest wall distance and direction.
        # w_i = min(0.5-x, 0.5+x, 0.5-y, 0.5+y)
        # If 0.5-x is min, grad is -2(0.5-x) in x, 0 in y.
        # If 0.5+x is min, grad is +2(0.5+x) in x, 0 in y.
        # If 0.5-y is min, grad is 0 in x, -2(0.5-y) in y.
        # If 0.5+y is min, grad is 0 in x, +2(0.5+y) in y.
        
        grad_wall = np.zeros_like(best_centres)
        dx_left = 0.5 - best_centres[:, 0]
        dx_right = 0.5 + best_centres[:, 0]
        dy_bottom = 0.5 - best_centres[:, 1]
        dy_top = 0.5 + best_centres[:, 1]
        
        # Determine which wall is closest for each point
        # We want to push away from the closest wall.
        # dist = min(dx_left, dx_right, dy_bottom, dy_top)
        # If dx_left is min, push right (+x). Grad of (0.5-x)^2 is -2(0.5-x). So force is +2(0.5-x) in x.
        # Wait, we are maximizing the distance. So we want to increase the distance.
        # d/dx (0.5-x)^2 = -2(0.5-x). This is negative if x < 0.5.
        # So to increase the objective, we should move in the direction of the gradient?
        # If we maximize f, we move in direction of grad f.
        # If f = (0.5-x)^2, grad = -2(0.5-x). If x=0, grad=-1. Move in -x direction? That decreases distance.
        # Ah, the objective is the distance. We want to maximize distance.
        # If we are at x=0, distance to left wall is 0.5. Distance to right is 0.5.
        # If we move to x=0.1, dist left=0.4, dist right=0.6. Min is 0.4. Decreased.
        # So we should move towards the center?
        # The "potential" is the distance to the *nearest* wall.
        # To maximize the minimum distance to walls, we should push points towards the center of the square?
        # No, we want to push them away from the walls.
        # If a point is close to the left wall, we should push it to the right.
        # The gradient of the distance to the left wall $d_L = 0.5 - x$ is $-1$.
        # If we maximize $d_L$, we should move in direction $- \nabla d_L = -(-1) = 1$ (positive x).
        # So the force should be away from the wall.
        
        # Let's construct the gradient of the min-wall-distance squared.
        # It's non-smooth at the boundaries where two walls are equidistant, but we can handle it by taking the gradient of the active constraint.
        
        # For each point i:
        # candidates: 
        # left: val = 0.5 - x, grad_vec = [-1, 0]
        # right: val = 0.5 + x, grad_vec = [1, 0]
        # bottom: val = 0.5 - y, grad_vec = [0, -1]
        # top: val = 0.5 + y, grad_vec = [0, 1]
        
        # We pick the one with min val.
        # Then grad_obj = 2 * val * grad_vec.
        # And we add this to the total gradient.
        
        vals = np.stack([dx_left, dx_right, dy_bottom, dy_top], axis=1)
        min_idx = np.argmin(vals, axis=1)
        min_vals = np.min(vals, axis=1)
        
        for i in range(N):
            idx = min_idx[i]
            v = min_vals[i]
            if idx == 0: # left
                grad_wall[i, 0] += 2.0 * v * (-1.0) # Wait. 
                # If we maximize (0.5-x)^2, grad is -2(0.5-x). 
                # If x=0, grad=-1. We move in direction of grad? 
                # If we move in direction -1 (left), x becomes negative, dist becomes > 0.5? No, x=-0.1, dist=0.6. 
                # But we are constrained to [-0.5, 0.5]. 
                # Actually, if we are at x=0, we can move left or right. 
                # The "min distance" is 0.5. 
                # If we move left, min dist becomes min(0.6, 0.4) = 0.4. 
                # So moving left *decreases* the min distance. 
                # So the gradient of the min-distance function should point towards the center?
                # Let's re-evaluate.
                # f(x) = min(0.5-x, 0.5+x). 
                # For x in [-0.5, 0.5], f(x) = 0.5 - |x|.
                # We want to maximize f(x). 
                # Max is at x=0. 
                # So the gradient should point towards 0.
                # If x > 0, f(x) = 0.5 - x. grad = -1. Move in direction -1 (towards 0). Correct.
                # If x < 0, f(x) = 0.5 + x. grad = 1. Move in direction 1 (towards 0). Correct.
                # So for left wall (x < 0 usually, or close to it), the force should be towards +x?
                # If x is close to -0.5, x is negative. 
                # f(x) = 0.5 - |x| = 0.5 + x. Grad = 1.
                # So we should move in +x direction.
                # My previous logic: "If left wall is min, push right". Yes.
                # So force is +1 in x.
                # Formula: 2 * val * grad_vec.
                # grad_vec for left wall distance $0.5-x$ is $[-1, 0]$.
                # So term is $2(0.5-x)(-1) = -2(0.5-x)$.
                # If x=-0.4, term is $-2(0.9) = -1.8$. This points to -x. WRONG.
                # The issue is that we are maximizing the distance.
                # The gradient of the distance $d$ is the direction to increase $d$.
                # $d_L = 0.5 - x$. $\nabla d_L = [-1, 0]$.
                # To increase $d_L$, we move in direction $-\nabla d_L = [1, 0]$.
                # So the force should be $- \nabla d_{min}$.
                # Let's use this.
                
                pass # Logic inside loop below
            
            # Re-implementing inside loop for clarity
            if idx == 0: # left
                # d = 0.5 - x. grad_d = [-1, 0]. Force = -grad_d = [1, 0].
                # Weighted by derivative of d^2? No, let's just use force proportional to distance?
                # Or just constant force?
                # Let's use force = 2 * d * (-grad_d) ? No.
                # Let's maximize sum of squared distances to walls.
                # Obj = sum w_i^2. Grad = 2 w_i grad_w_i.
                # w_i = min(...).
                # If left is min, w_i = 0.5 - x. grad_w_i = [-1, 0].
                # Grad Obj = 2(0.5-x)[-1, 0] = [-2(0.5-x), 0].
                # If x=-0.4, Grad = [-1.8, 0]. Move in -x direction.
                # This moves x to -0.5, decreasing distance.
                # Why? Because we are maximizing the *square* of the distance to the *left* wall.
                # If we are at x=-0.4, dist to left is 0.9. Dist to right is 0.1.
                # Min dist is 0.1 (right wall).
                # So the active constraint is the RIGHT wall.
                # So we should push away from the right wall (towards left).
                # My code picks min_idx.
                # If x=-0.4, dx_left=0.9, dx_right=0.1. Min is right.
                # So idx=1.
                # w = 0.1. grad_w = [1, 0].
                # Grad Obj = 2(0.1)[1, 0] = [0.2, 0]. Move in +x direction.
                # This moves x to -0.3. Dist to right becomes 0.2. Min dist increases. Correct.
                
                pass

        # Corrected Wall Gradient Calculation
        for i in range(N):
            idx = min_idx[i]
            v = min_vals[i]
            if idx == 0: # left
                # w = 0.5 - x. grad_w = [-1, 0].
                # We maximize w^2. Grad = 2w grad_w.
                # Force = 2 * v * [-1, 0].
                # This pushes in -x direction if v > 0.
                # If left wall is min, x is likely close to -0.5? No, if left is min, x is close to -0.5?
                # If x=-0.4, left=0.9, right=0.1. Right is min.
                # If x=0.4, left=0.1, right=0.9. Left is min.
                # So if left is min, x > 0.
                # w = 0.5 - x. grad_w = [-1, 0].
                # Force = 2(0.5-x)[-1, 0] = [-2(0.5-x), 0].
                # If x=0.4, force = [-0.2, 0]. Move in -x direction.
                # x becomes 0.3. Dist to left becomes 0.2. Min dist increases. Correct.
                grad_wall[i, 0] += -2.0 * v
            elif idx == 1: # right
                # w = 0.5 + x. grad_w = [1, 0].
                # Force = 2(0.5+x)[1, 0] = [2(0.5+x), 0].
                # If x=-0.4, force = [0.2, 0]. Move in +x direction.
                # x becomes -0.3. Dist to right becomes 0.2. Min dist increases. Correct.
                grad_wall[i, 0] += 2.0 * v
            elif idx == 2: # bottom
                # w = 0.5 - y. grad_w = [0, -1].
                # Force = 2(0.5-y)[0, -1] = [0, -2(0.5-y)].
                grad_wall[i, 1] += -2.0 * v
            elif idx == 3: # top
                # w = 0.5 + y. grad_w = [0, 1].
                # Force = 2(0.5+y)[0, 1] = [0, 2(0.5+y)].
                grad_wall[i, 1] += 2.0 * v

        total_grad = grad_pair + grad_wall
        
        # Normalize gradient to prevent huge steps
        grad_norm = np.linalg.norm(total_grad)
        if grad_norm > 0:
            total_grad = total_grad / grad_norm
        
        # Update
        new_centres = best_centres + current_step * total_grad
        
        # Clip to bounds (slightly inside to avoid numerical issues)
        new_centres = np.clip(new_centres, -0.4999, 0.4999)
        
        new_r = _compute_radius(new_centres)
        
        if new_r > best_r + 1e-12:
            best_r = new_r
            best_centres = new_centres
            no_improve_count = 0
            # Increase step size slightly if improving? Or keep stable.
            # current_step = min(0.01, current_step * 1.05)
        else:
            no_improve_count += 1
            current_step *= 0.95
            if current_step < 1e-8:
                break
            
            if no_improve_count > 50:
                # Try a random kick to escape local minima
                kick = np.random.normal(0, 0.001, size=(N, 2))
                new_centres = best_centres + kick
                new_centres = np.clip(new_centres, -0.4999, 0.4999)
                new_r = _compute_radius(new_centres)
                if new_r > best_r:
                    best_r = new_r
                    best_centres = new_centres
                    no_improve_count = 0
                    current_step = 0.001
                else:
                    break

    return best_centres, best_r


def construct():
    start_time = time.time()
    
    # Start with hexagonal packing as initial configuration
    centres = _hexagonal_packing()
    current_r = _compute_radius(centres)
    
    # If hexagonal didn't give a good result, try grid
    if current_r < 0.03:
        centres = _grid_packing()
        current_r = _compute_radius(centres)
    
    # Local optimization
    if time.time() - start_time < TIME_LIMIT - 10:
        centres, current_r = _local_optimization(centres)
    
    # Final check
    final_r = _compute_radius(centres)
    if final_r <= 0:
        # Fallback to a valid grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return centres
