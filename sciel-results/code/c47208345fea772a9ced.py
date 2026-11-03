"""Improved circle packing in unit square using L-BFGS-B with soft penalties."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START

def _penalized_objective(params, n, penalty_weight):
    """
    Objective: Minimize negative sum of radii + penalty for violations.
    params: [x0, y0, r0, x1, y1, r1, ...]
    """
    positions = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Base objective: maximize sum of radii
    obj = -np.sum(radii)
    
    # Penalty for boundary violations
    # x - r >= 0, 1 - x - r >= 0, etc.
    # Penalty is sum of max(0, violation)^2
    boundary_violations = np.maximum(0, radii - positions[:, 0])
    boundary_violations += np.maximum(0, radii - (1 - positions[:, 0]))
    boundary_violations += np.maximum(0, radii - positions[:, 1])
    boundary_violations += np.maximum(0, radii - (1 - positions[:, 1]))
    obj += penalty_weight * np.sum(boundary_violations**2)
    
    # Penalty for overlap violations
    # dist(i,j) >= r_i + r_j
    # violation = max(0, r_i + r_j - dist)
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            violation = np.maximum(0, radii[i] + radii[j] - dist)
            obj += penalty_weight * violation**2
            
    return obj

def _is_valid(params, n, tol=1e-6):
    """Check if configuration is valid (no overlaps, within bounds)."""
    positions = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Check bounds
    for i in range(n):
        x, y = positions[i]
        r = radii[i]
        if r < -tol:
            return False
        if x - r < -tol or y - r < -tol or x + r > 1 + tol or y + r > 1 + tol:
            return False
    
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    
    return True

def _initial_layout(n, seed):
    """Generate a random or grid-like initial layout."""
    np.random.seed(seed)
    # Use a jittered grid to ensure good spread
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            # Add jitter
            x += np.random.uniform(-0.5/side, 0.5/side)
            y += np.random.uniform(-0.5/side, 0.5/side)
            positions.append([x, y])
    
    positions = np.array(positions[:n])
    # Clip to [0, 1]
    positions = np.clip(positions, 0.0, 1.0)
    
    # Initial radius estimate: small enough to fit
    r0 = 0.5 / side * 0.8
    return positions, r0

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 115s limit, keep 5s margin for safety
    
    best_result = None
    best_score = -np.inf
    
    # Bounds for L-BFGS-B
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = [(0, 1)] * (2*N) + [(0, 0.5)] * N
    
    # Penalty weight for soft constraints
    # Start with a moderate penalty and increase it if needed, 
    # or just use a high one. L-BFGS-B is fast, so we can try a few restarts.
    penalty_weight = 100.0
    
    # Try multiple random restarts
    max_restarts = 20 # We will break based on time
    for i in range(max_restarts):
        if time.time() > deadline:
            break
        
        positions, r0 = _initial_layout(N, seed=42 + i)
        
        # Initial parameters
        params0 = np.zeros(3*N)
        params0[:2*N] = positions.flatten()
        params0[2*N:] = r0
        
        # Optimize
        try:
            result = minimize(
                _penalized_objective,
                params0,
                args=(N, penalty_weight),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            # Verify validity strictly
            if _is_valid(result.x, N):
                score = np.sum(result.x[2*N:])
                if score > best_score:
                    best_score = score
                    best_result = result.x.reshape(N, 3)
                    
        except Exception:
            continue
    
    # If no valid result found, return the grid seed
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result

# EVOLVE-BLOCK-END
