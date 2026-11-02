"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time

def _construct():
    """
    Pack 32 circles in the unit square maximizing sum of radii.
    Uses a multi-start local optimization with gradient-free Nelder-Mead
    on the combined objective: sum of radii, with penalties for overlap
    and boundary violations.
    """
    deadline = time.time() + 110  # 110s safety margin out of 120s
    
    best_circles = None
    best_score = -np.inf
    
    # Initial grid-based solution as fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
             for i in range(side) for j in range(side)]
    grid_circles = np.array(cells[:N])
    best_circles = grid_circles.copy()
    best_score = _evaluate(grid_circles)
    
    rng = np.random.default_rng(42)
    
    # Try multiple random initializations with local optimization
    n_starts = 20
    
    for start_idx in range(n_starts):
        if time.time() > deadline:
            break
        
        # Generate random initial positions
        x = rng.uniform(0.05, 0.95, N)
        y = rng.uniform(0.05, 0.95, N)
        r = rng.uniform(0.02, 0.08, N)
        
        circles = np.column_stack([x, y, r])
        
        # Local optimization using scipy's minimize
        try:
            from scipy.optimize import minimize
            
            def objective(params):
                x = params[:N]
                y = params[N:2*N]
                r = params[2*N:]
                
                # Penalty for going out of bounds
                penalty = 0.0
                for i in range(N):
                    if x[i] < r[i]:
                        penalty += (r[i] - x[i]) ** 2
                    if x[i] > 1 - r[i]:
                        penalty += (x[i] - (1 - r[i])) ** 2
                    if y[i] < r[i]:
                        penalty += (r[i] - y[i]) ** 2
                    if y[i] > 1 - r[i]:
                        penalty += (y[i] - (1 - r[i])) ** 2
                
                # Penalty for overlapping circles
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        min_dist = r[i] + r[j]
                        if dist < min_dist:
                            penalty += (min_dist - dist) ** 2
                
                # Maximize sum of radii (minimize negative)
                return -np.sum(r) + 100 * penalty
            
            initial_params = circles.flatten()
            
            result = minimize(
                objective,
                initial_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if result.success or result.fun < best_score:
                opt_circles = result.x.reshape(N, 3)
                # Clamp to valid range
                opt_circles[:, 0] = np.clip(opt_circles[:, 0], 0.01, 0.99)
                opt_circles[:, 1] = np.clip(opt_circles[:, 1], 0.01, 0.99)
                opt_circles[:, 2] = np.clip(opt_circles[:, 2], 0.005, 0.3)
                
                score = _evaluate(opt_circles)
                if score > best_score:
                    best_score = score
                    best_circles = opt_circles.copy()
                    
        except Exception:
            continue
    
    # Final refinement on best solution
    if best_circles is not None and time.time() < deadline:
        try:
            from scipy.optimize import minimize
            
            def objective(params):
                x = params[:N]
                y = params[N:2*N]
                r = params[2*N:]
                
                penalty = 0.0
                for i in range(N):
                    if x[i] < r[i]:
                        penalty += (r[i] - x[i]) ** 2
                    if x[i] > 1 - r[i]:
                        penalty += (x[i] - (1 - r[i])) ** 2
                    if y[i] < r[i]:
                        penalty += (r[i] - y[i]) ** 2
                    if y[i] > 1 - r[i]:
                        penalty += (y[i] - (1 - r[i])) ** 2
                
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        min_dist = r[i] + r[j]
                        if dist < min_dist:
                            penalty += (min_dist - dist) ** 2
                
                return -np.sum(r) + 100 * penalty
            
            initial_params = best_circles.flatten()
            
            result = minimize(
                objective,
                initial_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if result.fun < best_score:
                opt_circles = result.x.reshape(N, 3)
                opt_circles[:, 0] = np.clip(opt_circles[:, 0], 0.01, 0.99)
                opt_circles[:, 1] = np.clip(opt_circles[:, 1], 0.01, 0.99)
                opt_circles[:, 2] = np.clip(opt_circles[:, 2], 0.005, 0.3)
                
                score = _evaluate(opt_circles)
                if score > best_score:
                    best_circles = opt_circles.copy()
        
        except Exception:
            pass
    
    return best_circles


def _evaluate(circles):
    """Evaluate a packing: return sum of radii if valid, else negative."""
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    
    # Check boundaries
    for i in range(N):
        if x[i] < r[i] - 1e-9 or x[i] > 1 - r[i] + 1e-9:
            return -1.0
        if y[i] < r[i] - 1e-9 or y[i] > 1 - r[i] + 1e-9:
            return -1.0
    
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if dist < r[i] + r[j] - 1e-9:
                return -1.0
    
    return np.sum(r)


def construct():
    return _construct()
# EVOLVE-BLOCK-END
