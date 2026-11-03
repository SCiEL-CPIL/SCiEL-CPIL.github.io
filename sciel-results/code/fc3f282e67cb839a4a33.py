"""Circle packing in unit square using physics-inspired L-BFGS-B optimization."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from math import sqrt

_DEADLINE = 140.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START_TIME


def _penalized(params, penalty=1000.0):
    """
    Objective function: minimize negative sum of radii plus penalty for violations.
    params: array of shape (3*N,) where each triple is (x, y, r)
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Objective: maximize sum of radii => minimize -sum(r)
    obj = -np.sum(r)
    
    # Penalty for out-of-bounds
    for i in range(N):
        if x[i] - r[i] < 0:
            obj += penalty * (x[i] - r[i])**2
        if x[i] + r[i] > 1:
            obj += penalty * (x[i] + r[i] - 1)**2
        if y[i] - r[i] < 0:
            obj += penalty * (y[i] - r[i])**2
        if y[i] + r[i] > 1:
            obj += penalty * (y[i] + r[i] - 1)**2
        if r[i] < 0:
            obj += penalty * r[i]**2
    
    # Penalty for overlaps
    for i in range(N):
        for j in range(i+1, N):
            d = sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            overlap = r[i] + r[j] - d
            if overlap > 0:
                obj += penalty * overlap**2
    
    return obj


def _gradient(params, penalty=1000.0):
    """
    Numerical gradient of the penalized objective.
    """
    eps = 1e-7
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _penalized(params_plus, penalty)
        
        params_minus = params.copy()
        params_minus[i] -= eps
        f_minus = _penalized(params_minus, penalty)
        
        grad[i] = (f_plus - f_minus) / (2 * eps)
    
    return grad


def _is_valid(params, tol=1e-6):
    """Check if the configuration is valid."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    for i in range(N):
        if x[i] - r[i] < -tol or x[i] + r[i] > 1 + tol:
            return False
        if y[i] - r[i] < -tol or y[i] + r[i] > 1 + tol:
            return False
        if r[i] < -tol:
            return False
    
    for i in range(N):
        for j in range(i+1, N):
            d = sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    return True


def _grid_init(seed=42):
    """Create a grid initialization with small random perturbations."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(N)))
    params = np.zeros(3 * N)
    
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            x = (i + 0.5) / side + rng.uniform(-0.02, 0.02)
            y = (j + 0.5) / side + rng.uniform(-0.02, 0.02)
            r = 0.45 / side + rng.uniform(-0.01, 0.01)
            
            # Clamp to valid range
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            r = np.clip(r, 0.01, 0.25)
            
            params[3*count] = x
            params[3*count+1] = y
            params[3*count+2] = r
            count += 1
    
    return params


def _fix_to_boundary(params):
    """Fix any circles that are out of bounds by projecting them back."""
    x = params[0::3].copy()
    y = params[1::3].copy()
    r = params[2::3].copy()
    
    for i in range(N):
        # Ensure radius is positive
        r[i] = max(r[i], 0.001)
        
        # Ensure circle is within bounds
        if x[i] - r[i] < 0:
            x[i] = r[i]
        if x[i] + r[i] > 1:
            x[i] = 1 - r[i]
        if y[i] - r[i] < 0:
            y[i] = r[i]
        if y[i] + r[i] > 1:
            y[i] = 1 - r[i]
        
        # Ensure radius doesn't exceed bounds
        r[i] = min(r[i], min(x[i], 1-x[i], y[i], 1-y[i]))
        r[i] = max(r[i], 0.001)
    
    params[0::3] = x
    params[1::3] = y
    params[2::3] = r
    
    return params


def _optimize_single(params_init, max_iter=200, time_limit=20.0):
    """
    Run L-BFGS-B optimization from a single initial point.
    """
    global _START_TIME
    
    start = time.time()
    
    # Bounds for each parameter
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))  # x
        bounds.append((0.001, 0.999))  # y
        bounds.append((0.001, 0.45))   # r
    
    # Run optimization
    try:
        result = minimize(
            _penalized,
            params_init,
            method='L-BFGS-B',
            jac=_gradient,
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'ftol': 1e-10,
                'gtol': 1e-8,
                'maxls': 20
            }
        )
        
        # Check time
        if time.time() - start > time_limit:
            return params_init  # Return initial if timed out
        
        return result.x
    except Exception:
        return params_init


def _construct_optimized():
    """
    Main construction using physics-inspired optimization.
    Multiple random restarts with L-BFGS-B.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    best_params = None
    best_score = -np.inf
    
    # Try multiple random initializations
    n_restarts = 5
    rng = np.random.default_rng(2024)
    
    for restart in range(n_restarts):
        if _elapsed() > _DEADLINE - 10:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Grid initialization
            params_init = _grid_init(seed=42 + restart)
        elif restart == 1:
            # Corner-heavy initialization
            params_init = np.zeros(3 * N)
            corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
            for i, (cx, cy) in enumerate(corners):
                params_init[3*i] = cx
                params_init[3*i+1] = cy
                params_init[3*i+2] = 0.15
            # Fill rest with random
            for i in range(4, N):
                params_init[3*i] = rng.uniform(0.1, 0.9)
                params_init[3*i+1] = rng.uniform(0.1, 0.9)
                params_init[3*i+2] = rng.uniform(0.02, 0.1)
        elif restart == 2:
            # Hexagonal-ish packing
            params_init = np.zeros(3 * N)
            row = 0
            col = 0
            spacing = 0.25
            for i in range(N):
                if col == 4:
                    col = 0
                    row += 1
                x = 0.15 + col * spacing + (0.05 if row % 2 == 1 else 0)
                y = 0.15 + row * spacing * 0.866
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                params_init[3*i] = x
                params_init[3*i+1] = y
                params_init[3*i+2] = 0.1
                col += 1
        else:
            # Pure random
            params_init = np.zeros(3 * N)
            for i in range(N):
                params_init[3*i] = rng.uniform(0.1, 0.9)
                params_init[3*i+1] = rng.uniform(0.1, 0.9)
                params_init[3*i+2] = rng.uniform(0.02, 0.12)
        
        # Fix to boundary
        params_init = _fix_to_boundary(params_init)
        
        # Run optimization with time limit
        time_remaining = _DEADLINE - _elapsed() - 5
        time_limit = min(25.0, time_remaining / (n_restarts - restart))
        
        if time_limit < 2:
            break
        
        params_opt = _optimize_single(params_init, max_iter=150, time_limit=time_limit)
        params_opt = _fix_to_boundary(params_opt)
        
        # Evaluate
        score = -_penalized(params_opt, penalty=1000.0)
        
        if score > best_score:
            best_score = score
            best_params = params_opt.copy()
    
    return best_params


def _construct_grid_fallback():
    """Fallback grid construction."""
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.45 / side) 
             for i in range(side) for j in range(side)]
    return np.array(cells[:N], dtype=np.float64)


def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    try:
        params = _construct_optimized()
        
        if params is None or not _is_valid(params, tol=1e-4):
            # Try to fix
            params = _fix_to_boundary(params)
            if not _is_valid(params, tol=1e-4):
                # Reduce radii slightly
                r = params[2::3]
                r = r * 0.9
                params[2::3] = r
                params = _fix_to_boundary(params)
                if not _is_valid(params, tol=1e-4):
                    raise ValueError("Invalid after fixing")
        
        result = params.reshape(N, 3)
        
        # Final check
        if not _is_valid(result.flatten(), tol=1e-5):
            raise ValueError("Final validation failed")
            
    except Exception:
        result = _construct_grid_fallback()
    
    return result
# EVOLVE-BLOCK-END
