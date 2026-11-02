"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 32

# EVOLVE-BLOCK-START
def _hex_init(n):
    """Initialize with hexagonal packing."""
    centers = np.zeros((n, 2))
    # Hexagonal packing in a square-ish region
    # Side length of hexagon grid
    side = int(math.ceil(math.sqrt(n)))
    # Adjust spacing to fit in [0,1]
    # Approximate radius for hex packing
    r = 0.1
    dx = 2 * r * math.sqrt(3) / 2
    dy = 2 * r * 1.5
    x_offset = 0
    y_offset = 0
    
    # Generate points
    pts = []
    for i in range(side + 1):
        for j in range(side + 1):
            x = (i * dx) + (j % 2) * (dx / 2)
            y = j * dy
            pts.append([x, y])
            
    pts = np.array(pts)
    # Normalize to fit in [0.1, 0.9] roughly
    if len(pts) > 0:
        min_x, min_y = pts.min(axis=0)
        max_x, max_y = pts.max(axis=0)
        span_x = max_x - min_x
        span_y = max_y - min_y
        if span_x > 0:
            pts[:, 0] = (pts[:, 0] - min_x) / span_x * 0.8 + 0.1
        if span_y > 0:
            pts[:, 1] = (pts[:, 1] - min_y) / span_y * 0.8 + 0.1
            
    # Take first n
    if len(pts) >= n:
        return pts[:n]
    else:
        # Fallback to grid
        side_g = int(math.ceil(math.sqrt(n)))
        centers = np.array([(i + 0.5) / side_g, (j + 0.5) / side_g] for i in range(side_g) for j in range(side_g) if i*side_g+j < n)
        return centers

def _compute_hard_radii(centers):
    """Compute the true maximum radii for given centers."""
    n = len(centers)
    radii = np.zeros(n)
    for i in range(n):
        c = centers[i]
        r_wall = min(c[0], 1 - c[0], c[1], 1 - c[1])
        min_dist = r_wall
        for j in range(n):
            if i != j:
                d = np.linalg.norm(centers[i] - centers[j])
                min_dist = min(min_dist, d / 2)
        radii[i] = min_dist
    return radii

def _soft_min(values, beta=50):
    """
    Smooth approximation of min.
    Returns (soft_min_value, weights)
    weights[i] is the gradient contribution of values[i] to the soft_min.
    soft_min = -1/beta * log(sum(exp(-beta * values)))
    d(soft_min)/d(values[i]) = exp(-beta * values[i]) / sum(exp(-beta * values))
    """
    # Shift for numerical stability
    m = np.min(values)
    shifted = values - m
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    val = m - np.log(sum_exp) / beta
    return val, weights

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = N
    best_centers = None
    best_sum = -1.0
    
    # Initializations
    inits = [
        _hex_init(n),
        np.array([(i + 0.5) / 6, (j + 0.5) / 6] for i in range(6) for j in range(6) if i*6+j < n),
    ]
    
    # Add some random inits if time permits
    np.random.seed(42)
    for _ in range(2):
        inits.append(np.random.rand(n, 2) * 0.8 + 0.1)

    beta = 50.0 # Sharpness of soft-min
    
    for c0 in inits:
        if time.time() > deadline - 10:
            break
            
        c0 = np.clip(c0, 0.01, 0.99)
        
        def objective_and_grad(x):
            c = x.reshape(n, 2)
            # We don't strictly need to clip here if we use bounds in minimize, 
            # but it helps with gradient stability near boundaries.
            # L-BFGS-B handles bounds, so we can assume c is within bounds.
            
            total_val = 0.0
            grad = np.zeros_like(x)
            
            for i in range(n):
                # Distances to walls
                dists_wall = np.array([c[i][0], 1 - c[i][0], c[i][1], 1 - c[i][1]])
                # Distances to other circles / 2
                dists_circ = np.zeros(n-1)
                count = 0
                for j in range(n):
                    if i != j:
                        d = np.linalg.norm(c[i] - c[j])
                        dists_circ[count] = d / 2.0
                        count += 1
                
                # Combine all constraints for circle i
                # We want to maximize min(dists_wall, dists_circ)
                # This is equivalent to minimizing -min(...)
                # So we compute soft_min of these distances.
                all_dists = np.concatenate([dists_wall, dists_circ])
                
                val, weights = _soft_min(all_dists, beta)
                total_val += val
                
                # Gradient of soft_min w.r.t all_dists is weights
                # Chain rule: d(val)/d(x) = sum(weights * d(all_dists)/d(x))
                
                # d(dists_wall)/dx
                # dists_wall[0] = x, d/dx=1, d/dy=0
                # dists_wall[1] = 1-x, d/dx=-1, d/dy=0
                # dists_wall[2] = y, d/dx=0, d/dy=1
                # dists_wall[3] = 1-y, d/dx=0, d/dy=-1
                
                gx = weights[0] - weights[1]
                gy = weights[2] - weights[3]
                
                # d(dists_circ)/dx
                # dists_circ[k] = 0.5 * ||c_i - c_j||
                # d/d(c_i) = 0.5 * (c_i - c_j) / ||c_i - c_j||
                count = 0
                for j in range(n):
                    if i != j:
                        diff = c[i] - c[j]
                        d = np.linalg.norm(diff)
                        if d > 1e-9:
                            # Gradient of distance w.r.t c_i
                            grad_vec = 0.5 * diff / d
                            gx += weights[4 + count] * grad_vec[0]
                            gy += weights[4 + count] * grad_vec[1]
                        count += 1
                        
                grad[2*i] = gx
                grad[2*i+1] = gy
            
            # We want to MAXIMIZE sum(radii), so we MINIMIZE -sum(radii)
            return -total_val, grad

        try:
            from scipy.optimize import minimize
            result = minimize(
                lambda x: objective_and_grad(x)[0],
                c0.flatten(),
                jac=lambda x: objective_and_grad(x)[1],
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * (2*n),
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            
            c_new = result.x.reshape(n, 2)
            # Compute hard radii for this configuration
            r_new = _compute_hard_radii(c_new)
            sum_new = np.sum(r_new)
            
            if sum_new > best_sum:
                best_sum = sum_new
                best_centers = c_new.copy()
                
        except Exception:
            pass

    # Final Polish: Simple local search to fix any minor overlaps or squeeze more
    if best_centers is not None and time.time() < deadline - 5:
        # Iterative refinement
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(n):
                c_i = best_centers[i]
                # Compute current radius
                r_i = _compute_hard_radii(best_centers)[i]
                
                # Try small perturbations
                best_pos = c_i.copy()
                best_r = r_i
                
                # Adaptive step size based on radius
                step = max(r_i * 0.1, 0.001)
                directions = [
                    [step, 0], [-step, 0], [0, step], [0, -step],
                    [step, step], [-step, -step], [step, -step], [-step, step]
                ]
                
                for dx, dy in directions:
                    new_pos = c_i + np.array([dx, dy])
                    if new_pos[0] < 0 or new_pos[0] > 1 or new_pos[1] < 0 or new_pos[1] > 1:
                        continue
                    
                    new_centers = best_centers.copy()
                    new_centers[i] = new_pos
                    new_r = _compute_hard_radii(new_centers)[i]
                    
                    # Also check if we violated any other constraints?
                    # _compute_hard_radii handles this by taking min with neighbors.
                    # If new_r > best_r, it's an improvement for this circle.
                    # However, moving circle i might reduce radius of neighbors.
                    # For a simple polish, we just maximize the sum.
                    # Let's compute sum for the whole configuration? That's expensive.
                    # Instead, we assume that if local radius increases and we don't push too hard, it's good.
                    # Better: check if the move is valid (no overlaps) and increases sum.
                    
                    # Quick validity check:
                    valid = True
                    for j in range(n):
                        if i == j: continue
                        d = np.linalg.norm(new_pos - best_centers[j])
                        # If d < r_i + r_j, they overlap.
                        # But r_i is changing.
                        # This simple heuristic is risky.
                        # Let's stick to the gradient result mostly.
                        pass
                    
                    if new_r > best_r + 1e-9:
                        # Check if this move causes global sum to decrease?
                        # To be safe, only accept if we don't overlap with others at their current radii.
                        # Actually, if we move i, r_i increases. Other radii might decrease.
                        # A safer polish is to just use the gradient result and skip this heuristic if it's complex.
                        # Let's try a simpler approach: just re-run a tiny gradient step?
                        # No, let's just accept if it increases the local radius and doesn't obviously overlap.
                        # Overlap check:
                        overlap = False
                        for j in range(n):
                            if i == j: continue
                            d = np.linalg.norm(new_pos - best_centers[j])
                            r_j = _compute_hard_radii(best_centers)[j] # Expensive, but n=32
                            # If d < new_r + r_j, potential overlap.
                            # But r_j might also change.
                            # Let's just accept the move if new_r > best_r.
                            # The next iteration of the main loop (if we had one) would fix neighbors.
                            # Since this is a one-shot polish, we might degrade the solution.
                            # Let's skip this specific heuristic to avoid bugs and rely on the optimizer.
                            pass
                        
                        # Actually, let's just do a "push apart" if overlaps exist.
                        # The optimizer should have handled this.
                        # Let's just break out of this polish loop if it's too risky.
                        pass
                        
            # Given the risk of degrading the solution with a simple heuristic, 
            # and the fact that the smooth optimizer should be good, 
            # I will skip the complex heuristic polish and just return the best found.
            break 

    # Ensure validity: if any overlaps, shrink radii.
    # The problem asks for disjoint interiors.
    # Our radii are computed as min(dist/2, wall_dist).
    # This guarantees disjoint interiors IF we use these radii.
    # So the result is always valid by construction.
    
    result = np.zeros((n, 3))
    if best_centers is not None:
        radii = _compute_hard_radii(best_centers)
        for i in range(n):
            result[i] = [best_centers[i][0], best_centers[i][1], radii[i]]
    else:
        # Fallback
        c = _hex_init(n)
        radii = _compute_hard_radii(c)
        for i in range(n):
            result[i] = [c[i][0], c[i][1], radii[i]]

    return result
# EVOLVE-BLOCK-END
