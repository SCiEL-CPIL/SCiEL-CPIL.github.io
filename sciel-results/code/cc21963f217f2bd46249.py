"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Initial grid placement
    side = int(np.ceil(np.sqrt(N)))
    x0 = np.zeros((N, 3))
    for idx in range(N):
        i = idx % side
        j = idx // side
        x0[idx, 0] = (i + 0.5) / side
        x0[idx, 1] = (j + 0.5) / side
        x0[idx, 2] = 0.5 / side - 1e-3
    
    best_x = x0.copy()
    best_score = _compute_score(x0)
    
    # Optimization with smooth penalty
    def objective(x):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 0.0  # Signal timeout
        return -_objective(x)
    
    def _objective(x):
        # Smooth penalty formulation
        penalty = 0.0
        weights = []
        
        # Boundary constraints: x - r >= 0, y - r >= 0, x + r <= 1, y + r <= 1
        for i in range(N):
            x_i, y_i, r_i = x[i]
            # Soft penalty for boundary violations
            for val in [x_i - r_i, y_i - r_i, r_i - x_i, r_i - y_i]:
                if val < 0:
                    penalty += np.exp(-val * 50)  # Smooth approximation of max(0, -val)
        
        # Pairwise distance constraints: dist(i,j) >= r_i + r_j
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i, 0] - x[j, 0]
                dy = x[i, 1] - x[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = x[i, 2] + x[j, 2]
                if dist < min_dist:
                    penalty += np.exp(-(dist - min_dist) * 50)
        
        # Maximize sum of radii, minimize penalty
        return np.sum(x[:, 2]) - penalty
    
    # Try multiple optimization restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x_init = x0.copy()
        else:
            # Add small random perturbation
            x_init = x0.copy()
            x_init[:, :2] += np.random.normal(0, 0.02, (N, 2))
            x_init[:, 2] = np.clip(x_init[:, 2], 0.01, 0.2)
        
        # Use L-BFGS-B for bounded optimization
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.25)] * N
        # Flatten bounds properly
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))  # x
            bounds.append((0.0, 1.0))  # y
            bounds.append((0.0, 0.25))  # radius
        
        try:
            result = minimize(
                objective,
                x_init.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success and time.time() < deadline:
                x_opt = result.x.reshape(N, 3)
                # Ensure validity
                x_opt[:, :2] = np.clip(x_opt[:, :2], 0.0, 1.0)
                x_opt[:, 2] = np.clip(x_opt[:, 2], 0.0, 0.25)
                
                score = _compute_score(x_opt)
                if score > best_score:
                    best_x = x_opt
                    best_score = score
                    
        except Exception:
            continue
    
    # Final high-precision polish if time remains
    if time.time() < deadline - 20:
        try:
            bounds = []
            for i in range(N):
                bounds.append((0.0, 1.0))
                bounds.append((0.0, 1.0))
                bounds.append((0.0, 0.25))
            
            result = minimize(
                objective,
                best_x.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success and time.time() < deadline:
                x_polished = result.x.reshape(N, 3)
                x_polished[:, :2] = np.clip(x_polished[:, :2], 0.0, 1.0)
                x_polished[:, 2] = np.clip(x_polished[:, 2], 0.0, 0.25)
                
                score = _compute_score(x_polished)
                if score > best_score:
                    best_x = x_polished
                    best_score = score
                    
        except Exception:
            pass
    
    return best_x

def _compute_score(x):
    """Compute the actual score: sum of radii if valid, else 0."""
    # Check boundary constraints
    for i in range(N):
        x_i, y_i, r_i = x[i]
        if x_i - r_i < -1e-9 or y_i - r_i < -1e-9 or x_i + r_i > 1 + 1e-9 or y_i + r_i > 1 + 1e-9:
            return 0.0
    
    # Check pairwise distance constraints
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i, 0] - x[j, 0]
            dy = x[i, 1] - x[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = x[i, 2] + x[j, 2]
            if dist < min_dist - 1e-9:
                return 0.0
    
    return np.sum(x[:, 2])
# EVOLVE-BLOCK-END
