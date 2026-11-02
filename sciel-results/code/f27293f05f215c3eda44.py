"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _penalty_objective(params, radii):
    """Objective: negative sum of radii + penalty for overlaps and boundary violations."""
    n = len(radii)
    positions = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Objective: maximize sum of radii
    obj = -np.sum(radii)
    
    # Boundary penalty: circles must be inside [0,1]^2
    # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    left_viol = np.maximum(0, -positions[:, 0] + radii)
    right_viol = np.maximum(0, positions[:, 0] + radii - 1)
    bottom_viol = np.maximum(0, -positions[:, 1] + radii)
    top_viol = np.maximum(0, positions[:, 1] + radii - 1)
    boundary_penalty = np.sum(left_viol**2 + right_viol**2 + bottom_viol**2 + top_viol**2)
    
    # Overlap penalty: for each pair, distance >= r_i + r_j
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                penalty += (min_dist - dist)**2
    
    return obj + 10.0 * (boundary_penalty + penalty)

def _hexagonal_initial(n):
    """Create a hexagonal packing initial configuration."""
    # For 26 circles, use a hexagonal grid pattern
    # Arrange in rows with offset
    # Try different arrangements
    
    best_params = None
    best_obj = np.inf
    
    for rows in range(4, 8):
        # Distribute n circles into rows
        per_row = [n // rows] * rows
        for i in range(n % rows):
            per_row[i] += 1
        
        positions = []
        radii = []
        
        for row_idx, count in enumerate(per_row):
            y = (row_idx + 0.5) / rows
            for col_idx in range(count):
                x = (col_idx + 0.5) / count
                # Add small random perturbation
                x += np.random.uniform(-0.01, 0.01)
                y += np.random.uniform(-0.01, 0.01)
                positions.append([x, y])
                radii.append(0.08)
        
        positions = np.array(positions)
        radii = np.array(radii)
        
        # Clip to valid range
        positions[:, 0] = np.clip(positions[:, 0], 0.05, 0.95)
        positions[:, 1] = np.clip(positions[:, 1], 0.05, 0.95)
        radii = np.clip(radii, 0.02, 0.15)
        
        params = np.concatenate([positions.flatten(), radii])
        obj = _penalty_objective(params, radii)
        
        if obj < best_obj:
            best_obj = obj
            best_params = params
    
    return best_params

def _grid_initial(n):
    """Create a grid initial configuration."""
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    radii = []
    
    for i in range(side):
        for j in range(side):
            if len(positions) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y])
            radii.append(0.5 / side - 0.005)
    
    positions = np.array(positions[:n])
    radii = np.array(radii[:n])
    
    # Add small random perturbation
    positions += np.random.uniform(-0.01, 0.01, positions.shape)
    positions[:, 0] = np.clip(positions[:, 0], 0.05, 0.95)
    positions[:, 1] = np.clip(positions[:, 1], 0.05, 0.95)
    
    params = np.concatenate([positions.flatten(), radii])
    return params

def _random_initial(n):
    """Create a random initial configuration."""
    positions = np.random.uniform(0.1, 0.9, (n, 2))
    radii = np.random.uniform(0.03, 0.12, n)
    params = np.concatenate([positions.flatten(), radii])
    return params

def _optimize_from_initial(params0, radii0, max_time=10):
    """Optimize from a given initial configuration."""
    start_time = time.time()
    
    bounds = []
    for i in range(len(radii0)):
        bounds.append((0.02, 0.98))  # x
        bounds.append((0.02, 0.98))  # y
        bounds.append((0.02, 0.15))  # radius
    
    # Use L-BFGS-B
    result = minimize(
        _penalty_objective,
        params0,
        args=(radii0,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
    )
    
    elapsed = time.time() - start_time
    if elapsed > max_time:
        return None
    
    return result.x

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin before 120s limit
    
    np.random.seed(42)
    
    best_params = None
    best_score = -np.inf
    
    # Try multiple initializations
    initializations = []
    
    # Grid
    initializations.append(_grid_initial(N))
    
    # Hexagonal
    initializations.append(_hexagonal_initial(N))
    
    # Random (multiple)
    for _ in range(5):
        initializations.append(_random_initial(N))
    
    for init_params in initializations:
        if time.time() > deadline - 15:
            break
        
        radii0 = init_params[2*N:]
        result_params = _optimize_from_initial(init_params, radii0, max_time=15)
        
        if result_params is not None:
            # Validate and compute score
            positions = result_params[:2*N].reshape(N, 2)
            radii = result_params[2*N:]
            
            # Check validity
            valid = True
            for i in range(N):
                if positions[i, 0] - radii[i] < -1e-6 or positions[i, 0] + radii[i] > 1 + 1e-6:
                    valid = False
                    break
                if positions[i, 1] - radii[i] < -1e-6 or positions[i, 1] + radii[i] > 1 + 1e-6:
                    valid = False
                    break
            
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((positions[i,0]-positions[j,0])**2 + (positions[i,1]-positions[j,1])**2)
                        if dist < radii[i] + radii[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_params = result_params.copy()
    
    # If no valid solution found, use the grid as fallback
    if best_params is None:
        init_params = _grid_initial(N)
        positions = init_params[:2*N].reshape(N, 2)
        radii = init_params[2*N:]
        # Ensure validity by shrinking
        for i in range(N):
            min_dist_to_boundary = min(positions[i,0], 1-positions[i,0], positions[i,1], 1-positions[i,1])
            radii[i] = min(radii[i], min_dist_to_boundary * 0.99)
        best_params = np.concatenate([positions.flatten(), radii])
    
    positions = best_params[:2*N].reshape(N, 2)
    radii = best_params[2*N:]
    
    # Final validation and clipping
    for i in range(N):
        positions[i, 0] = np.clip(positions[i, 0], radii[i] + 1e-6, 1 - radii[i] - 1e-6)
        positions[i, 1] = np.clip(positions[i, 1], radii[i] + 1e-6, 1 - radii[i] - 1e-6)
    
    result = np.column_stack([positions[:, 0], positions[:, 1], radii])
    return result
# EVOLVE-BLOCK-END
