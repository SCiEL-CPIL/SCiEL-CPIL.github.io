"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import scipy.optimize as opt
import time

N = 32

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Initialization: Hexagonal lattice pattern
    # We want to place N circles. A hexagonal lattice is denser than a square grid.
    # Let's try to fit a hexagonal pattern.
    # Approximate radius for hexagonal packing in unit square:
    # Area of square = 1. Area of circles = N * pi * r^2.
    # Packing fraction ~ 0.9069 for hex.
    # N * pi * r^2 * (1/0.9069) <= 1  => r <= sqrt(1 / (N * pi / 0.9069))
    # For N=32: r <= sqrt(0.9069 / (32 * pi)) ~ sqrt(0.9069 / 100.53) ~ sqrt(0.009) ~ 0.095
    # Let's use a slightly smaller initial radius to allow movement.
    
    # Create a hexagonal grid of points
    points = []
    r_hex = 0.1 # Initial spacing parameter
    # Generate points in a hex lattice
    for i in range(10):
        for j in range(10):
            x = i * r_hex * 1.732 + (j % 2) * r_hex * 0.866
            y = j * r_hex * 1.5
            if 0 <= x <= 1 and 0 <= y <= 1:
                points.append((x, y))
    
    # If we don't have enough points, add more or adjust
    # Actually, simpler: just use a jittered grid for initialization which is robust
    side = int(np.ceil(np.sqrt(N)))
    init_points = []
    for i in range(side):
        for j in range(side):
            # Center of cell
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            # Add small random jitter to break symmetry
            x += np.random.uniform(-0.1, 0.1) / side
            y += np.random.uniform(-0.1, 0.1) / side
            init_points.append((x, y))
            if len(init_points) >= N:
                break
        if len(init_points) >= N:
            break
            
    # If we still don't have N points (shouldn't happen with side=ceil(sqrt(N))), fill with random
    while len(init_points) < N:
        init_points.append((np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)))
        
    init_points = np.array(init_points[:N])
    
    # Initial radii: small enough to not overlap
    # Max possible radius is roughly 1/side
    init_radii = np.full(N, 0.5 / side * 0.8)
    
    # Flatten variables: x0, y0, r0, x1, y1, r1, ...
    x0 = np.concatenate([init_points[:, 0], init_points[:, 1], init_radii])
    
    # Constraints
    def objective(x):
        radii = x[2*N:]
        # Maximize sum of radii => Minimize negative sum
        return -np.sum(radii)
        
    def constraints(x):
        cons = []
        centers = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        
        # Boundary constraints:
        # x_i >= r_i  => x_i - r_i >= 0
        # y_i >= r_i  => y_i - r_i >= 0
        # 1 - x_i >= r_i => 1 - x_i - r_i >= 0
        # 1 - y_i >= r_i => 1 - y_i - r_i >= 0
        
        cons.append(centers[:, 0] - radii)
        cons.append(centers[:, 1] - radii)
        cons.append(1 - centers[:, 0] - radii)
        cons.append(1 - centers[:, 1] - radii)
        
        # Non-overlap constraints:
        # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        # We can use (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        # To be safe with SLSQP, we might want to ensure this is strictly positive or handle the non-smoothness at 0.
        # SLSQP handles inequality constraints.
        
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx*dx + dy*dy
                sum_r = radii[i] + radii[j]
                # Constraint: dist_sq - sum_r^2 >= 0
                cons.append(np.array([dist_sq - sum_r*sum_r]))
                
        return cons

    # SLSQP requires constraints to be in a specific format or a list of dicts.
    # Let's use the list of dicts format for clarity, or just return a list of values if using 'type': 'ineq'.
    # Actually, SLSQP accepts a list of constraints.
    
    # Let's restructure for SLSQP efficiency.
    # SLSQP can be slow with many constraints. N=32 => 32*4 + 32*31/2 = 128 + 496 = 624 constraints.
    # This is manageable.
    
    def make_constraints(x):
        centers = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        
        # Boundary
        c_bound = np.concatenate([
            centers[:, 0] - radii,
            centers[:, 1] - radii,
            1 - centers[:, 0] - radii,
            1 - centers[:, 1] - radii
        ])
        
        # Non-overlap
        c_non_overlap = []
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx*dx + dy*dy
                sum_r = radii[i] + radii[j]
                c_non_overlap.append(dist_sq - sum_r*sum_r)
        c_non_overlap = np.array(c_non_overlap)
        
        return np.concatenate([c_bound, c_non_overlap])
        
    # SLSQP constraint definition
    # We need to define them as a list of dictionaries or a single function returning an array?
    # If we return an array, we must specify 'type': 'ineq' for the whole array?
    # No, SLSQP expects a list of constraint dicts.
    # Alternatively, we can define a single constraint function that returns the vector of inequalities.
    # But standard SLSQP usage with a single 'ineq' constraint function returning a vector works.
    
    def constraint_func(x):
        return make_constraints(x)
        
    constraints = [
        {'type': 'ineq', 'fun': constraint_func}
    ]
    
    # Bounds
    # x, y in [0, 1]
    # r in [0, 0.5] (max radius in unit square is 0.5)
    bounds = []
    for i in range(N):
        bounds.append((0, 1)) # x_i
        bounds.append((0, 1)) # y_i
        bounds.append((0, 0.5)) # r_i
        
    # Run optimization
    # Max iterations to keep it fast
    max_iter = 500
    
    try:
        result = opt.minimize(
            objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints, 
            options={'maxiter': max_iter, 'ftol': 1e-9}
        )
        
        if result.success:
            x_opt = result.x
        else:
            # If it fails, try to use the best point found?
            # result.x is the last point. We should check validity.
            x_opt = result.x
            
        # Validate result
        centers = x_opt[:2*N].reshape(N, 2)
        radii = x_opt[2*N:]
        
        # Check if any circle is out of bounds or overlapping significantly
        valid = True
        for i in range(N):
            if centers[i, 0] < radii[i] - 1e-5 or centers[i, 1] < radii[i] - 1e-5 or \
               centers[i, 0] > 1 - radii[i] + 1e-5 or centers[i, 1] > 1 - radii[i] + 1e-5:
                valid = False
                break
            if radii[i] < 1e-6:
                valid = False
                break
                
        if valid:
            # Double check overlaps
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    if dist < radii[i] + radii[j] - 1e-5:
                        valid = False
                        break
                if not valid:
                    break
            
        if valid:
            # Format output
            result_arr = np.column_stack((centers, radii))
            return result_arr
        else:
            # Fallback to initial grid if optimization failed validity
            pass
            
    except Exception as e:
        pass

    # Fallback: Return the initial grid (or a slightly improved version)
    # Let's just return the grid from the seed to be safe
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
