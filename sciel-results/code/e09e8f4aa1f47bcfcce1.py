"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.spatial.distance import cdist
import itertools

def construct():
    """Pack 26 circles with disjoint interiors inside [0,1]^2 and maximize the sum of their radii."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Initial guess: grid arrangement
    side = int(np.ceil(np.sqrt(N)))
    initial_positions = []
    for i in range(side):
        for j in range(side):
            initial_positions.append([(i + 0.5) / side, (j + 0.5) / side])
    initial_positions = initial_positions[:N]
    x0 = np.array(initial_positions)
    
    # Initial radii: small uniform
    r0 = np.full(N, 0.02)
    
    # Decision variables: [x_1, y_1, ..., x_N, y_N, r_1, ..., r_N]
    n_vars = 2 * N + N  # 2*N positions + N radii
    
    def objective(vars):
        # Maximize sum of radii => minimize negative sum
        positions = vars[:2*N].reshape(N, 2)
        radii = vars[2*N:]
        return -np.sum(radii)
    
    # Constraints:
    # 1. Boundary: x_i >= r_i, x_i <= 1 - r_i, y_i >= r_i, y_i <= 1 - r_i
    # 2. Non-overlap: ||p_i - p_j|| >= r_i + r_j for all i < j
    # 3. Radii >= 0
    
    # We'll use nonlinear constraints for non-overlap and boundary
    # For efficiency, we'll use a penalty approach or SLSQP with constraints
    
    def boundary_constraints(vars):
        positions = vars[:2*N].reshape(N, 2)
        radii = vars[2*N:]
        # x_i - r_i >= 0
        # 1 - x_i - r_i >= 0
        # y_i - r_i >= 0
        # 1 - y_i - r_i >= 0
        cons = []
        for i in range(N):
            cons.append(positions[i, 0] - radii[i])
            cons.append(1 - positions[i, 0] - radii[i])
            cons.append(positions[i, 1] - radii[i])
            cons.append(1 - positions[i, 1] - radii[i])
        return np.array(cons)
    
    def nonoverlap_constraints(vars):
        positions = vars[:2*N].reshape(N, 2)
        radii = vars[2*N:]
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(positions[i] - positions[j])
                cons.append(dist - radii[i] - radii[j])
        return np.array(cons)
    
    # Bounds: positions in [0,1], radii in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0, 1))  # x_i
        bounds.append((0, 1))  # y_i
    for i in range(N):
        bounds.append((1e-6, 0.5))  # r_i
    
    # Use SLSQP with nonlinear constraints
    # This might be slow, so let's try a different approach: iterative local optimization
    
    # Alternative: Use a penalty method with gradient descent
    # Or use scipy.optimize.minimize with 'SLSQP' but it might be slow for 26 circles
    
    # Let's try a smarter approach: start with grid, then optimize radii and positions
    # using a local search with gradient-based methods
    
    # Actually, let's use a different strategy:
    # 1. Start with a reasonable initial configuration
    # 2. Use scipy.optimize.minimize with method='SLSQP' and constraints
    # 3. If too slow, fall back to a simpler approach
    
    # Given the time limit, let's try a hybrid approach:
    # Use a penalty-based gradient descent with Nelder-Mead or L-BFGS-B
    
    def penalty_objective(vars, penalty_coeff=100):
        positions = vars[:2*N].reshape(N, 2)
        radii = vars[2*N:]
        
        # Main objective: maximize sum of radii
        obj = -np.sum(radii)
        
        # Penalty for boundary violations
        for i in range(N):
            if positions[i, 0] < radii[i]:
                obj += penalty_coeff * (radii[i] - positions[i, 0])**2
            if positions[i, 0] > 1 - radii[i]:
                obj += penalty_coeff * (positions[i, 0] + radii[i] - 1)**2
            if positions[i, 1] < radii[i]:
                obj += penalty_coeff * (radii[i] - positions[i, 1])**2
            if positions[i, 1] > 1 - radii[i]:
                obj += penalty_coeff * (positions[i, 1] + radii[i] - 1)**2
        
        # Penalty for overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(positions[i] - positions[j])
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    obj += penalty_coeff * (min_dist - dist)**2
        
        return obj
    
    # Use L-BFGS-B with penalty method
    # Start with a good initial guess
    x_init = np.concatenate([x0.flatten(), r0])
    
    best_vars = x_init.copy()
    best_obj = penalty_objective(best_vars)
    
    # Try multiple restarts with different initial configurations
    configs = [
        # Grid
        x0.copy(),
        # Hexagonal packing
        None,
        # Random
        None
    ]
    
    # Generate hexagonal initial configuration
    hex_positions = []
    rows = 0
    cols = 0
    # For hexagonal packing, alternate rows offset by half
    spacing = 0.15
    y = spacing
    while len(hex_positions) < N:
        offset = 0 if rows % 2 == 0 else spacing / 2
        x = spacing + offset
        while x <= 1 - spacing and len(hex_positions) < N:
            hex_positions.append([x, y])
            x += spacing * np.sqrt(3)
        y += spacing * 0.5
        rows += 1
    hex_positions = np.array(hex_positions[:N])
    
    # Random configurations
    random_configs = []
    np.random.seed(42)
    for _ in range(5):
        rand_pos = np.random.uniform(0.1, 0.9, (N, 2))
        random_configs.append(rand_pos)
    
    all_configs = [x0, hex_positions] + random_configs
    
    for config_idx, config in enumerate(all_configs):
        if time.time() > deadline - 10:
            break
        
        r_init = np.full(N, 0.02)
        x_init = np.concatenate([config.flatten(), r_init])
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                penalty_objective,
                x_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_vars = result.x.copy()
        except Exception:
            continue
    
    # Extract best solution
    best_positions = best_vars[:2*N].reshape(N, 2)
    best_radii = best_vars[2*N:]
    
    # Ensure validity: clip positions and radii
    for i in range(N):
        # Clip radii to be non-negative
        best_radii[i] = max(best_radii[i], 1e-6)
        
        # Ensure circle is inside the unit square
        best_radii[i] = min(best_radii[i], 
                           best_positions[i, 0], 
                           1 - best_positions[i, 0],
                           best_positions[i, 1],
                           1 - best_positions[i, 1])
        best_radii[i] = max(best_radii[i], 1e-6)
    
    # Final check and adjustment for overlaps
    # If there are still overlaps, reduce radii
    for _ in range(10):
        overlap_found = False
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(best_positions[i] - best_positions[j])
                min_dist = best_radii[i] + best_radii[j]
                if dist < min_dist:
                    # Reduce the larger radius
                    if best_radii[i] > best_radii[j]:
                        best_radii[i] = max((dist - best_radii[j]) / 2, 1e-6)
                    else:
                        best_radii[j] = max((dist - best_radii[i]) / 2, 1e-6)
                    overlap_found = True
        if not overlap_found:
            break
    
    # Create output array
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [best_positions[i, 0], best_positions[i, 1], best_radii[i]]
    
    return result

# EVOLVE-BLOCK-END
