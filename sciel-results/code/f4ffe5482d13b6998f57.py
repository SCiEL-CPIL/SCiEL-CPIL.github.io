"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _validate(arr):
    """Check that circles are inside [0,1]^2 and disjoint."""
    if arr.shape != (N, 3):
        return False
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r <= 0):
        return False
    if np.any(x - r < -1e-9) or np.any(y - r < -1e-9) or np.any(x + r > 1 + 1e-9) or np.any(y + r > 1 + 1e-9):
        return False
    # Check pairwise distances
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (x[i] - x[j])**2 + (y[i] - y[j])**2
            if d2 < (r[i] + r[j])**2 - 1e-9:
                return False
    return True


def _score(arr):
    """Sum of radii."""
    return float(np.sum(arr[:, 2]))


def _softplus(x, beta=100.0):
    """Numerically stable softplus."""
    # For large positive x, softplus(x) ≈ x
    # For large negative x, softplus(x) ≈ 0
    return np.where(x > 20, x, np.log1p(np.exp(-x)) if x > -20 else 0.0)


def _energy(params, beta=100.0):
    """
    Energy function: negative sum of radii + penalty for violations.
    params: array of shape (3*N,) = [x0,y0,r0, x1,y1,r1, ...]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Objective: maximize sum(r) -> minimize -sum(r)
    obj = -np.sum(r)
    
    # Boundary penalties: x - r >= 0, y - r >= 0, x + r <= 1, y + r <= 1
    # Penalty for violation of x - r >= 0: softplus(-(x - r))
    pen_boundary = np.sum(_softplus(r - x, beta))
    pen_boundary += np.sum(_softplus(r - y, beta))
    pen_boundary += np.sum(_softplus(x + r - 1, beta))
    pen_boundary += np.sum(_softplus(y + r - 1, beta))
    
    # Pairwise non-overlap: distance >= r_i + r_j
    # Violation: r_i + r_j - dist
    pen_overlap = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy + 1e-12)
            violation = r[i] + r[j] - dist
            pen_overlap += _softplus(violation, beta)
    
    return obj + 10.0 * (pen_boundary + pen_overlap)


def _energy_grad(params, beta=100.0):
    """
    Gradient of energy function.
    """
    x = params[0::3].copy()
    y = params[1::3].copy()
    r = params[2::3].copy()
    
    grad = np.zeros_like(params)
    
    # d(-sum(r))/dr = -1
    grad[2::3] -= 1.0
    
    # Boundary penalties
    # softplus(r - x): d/dx = -sigma(beta*(r-x)), d/dr = sigma(beta*(r-x))
    # softplus(r - y): d/dy = -sigma(beta*(r-y)), d/dr = sigma(beta*(r-y))
    # softplus(x + r - 1): d/dx = sigma(beta*(x+r-1)), d/dr = sigma(beta*(x+r-1))
    # softplus(y + r - 1): d/dy = sigma(beta*(y+r-1)), d/dr = sigma(beta*(y+r-1))
    
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    
    s1 = sigmoid(beta * (r - x))
    s2 = sigmoid(beta * (r - y))
    s3 = sigmoid(beta * (x + r - 1))
    s4 = sigmoid(beta * (y + r - 1))
    
    grad[0::3] += 10.0 * (-s1 + s3)
    grad[1::3] += 10.0 * (-s2 + s4)
    grad[2::3] += 10.0 * (s1 + s2 + s3 + s4)
    
    # Pairwise overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy + 1e-12)
            violation = r[i] + r[j] - dist
            s = sigmoid(beta * violation)
            
            # d(softplus(violation))/d(dist) = -s
            # d(dist)/dx_i = dx/dist, d(dist)/dx_j = -dx/dist
            # d(violation)/dr_i = 1, d(violation)/dr_j = 1
            
            # Contribution to grad
            # dE/dx_i = 10 * s * (-dx/dist)  [since d(softplus)/d(dist) = -s, d(dist)/dx_i = dx/dist]
            # Wait: softplus(violation), violation = r_i + r_j - dist
            # d(softplus(violation))/d(violation) = s
            # d(violation)/dx_i = -dx/dist
            # So dE/dx_i = 10 * s * (-dx/dist)
            
            # Actually let me redo:
            # E = 10 * softplus(violation)
            # dE/dx_i = 10 * s * d(violation)/dx_i = 10 * s * (-dx/dist)
            # dE/dx_j = 10 * s * d(violation)/dx_j = 10 * s * (dx/dist)
            # dE/dy_i = 10 * s * (-dy/dist)
            # dE/dy_j = 10 * s * (dy/dist)
            # dE/dr_i = 10 * s * 1
            # dE/dr_j = 10 * s * 1
            
            if dist > 1e-10:
                factor = 10.0 * s / dist
                grad[3*i] += factor * (-dx)
                grad[3*j] += factor * dx
                grad[3*i+1] += factor * (-dy)
                grad[3*j+1] += factor * dy
            grad[3*i+2] += 10.0 * s
            grad[3*j+2] += 10.0 * s
    
    return grad


def _initial_config():
    """
    Generate initial configuration: circles placed in a grid with some randomness.
    """
    # Use a 5x5 grid (25 cells) plus one extra
    # Actually, let's use a more organic starting point
    # Place circles in a hexagonal-like pattern
    
    # Start with a grid
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            positions.append([
                (i + 0.5) / side + np.random.uniform(-0.05, 0.05),
                (j + 0.5) / side + np.random.uniform(-0.05, 0.05),
                0.1
            ])
    
    # Add one more circle
    positions.append([0.5 + np.random.uniform(-0.1, 0.1), 
                      0.5 + np.random.uniform(-0.1, 0.1), 
                      0.1])
    
    return np.array(positions[:N])


def _optimize():
    """
    Main optimization routine.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    best_score = -np.inf
    best_params = None
    
    # Try multiple initializations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        np.random.seed(42 + restart)
        params0 = _initial_config().flatten()
        
        # Phase 1: Gradient descent with adaptive step size
        params = params0.copy()
        lr = 0.01
        min_lr = 1e-6
        max_lr = 0.1
        patience = 50
        no_improve = 0
        best_e = _energy(params)
        
        for step in range(2000):
            if time.time() > deadline - 15.0:
                break
            
            grad = _energy_grad(params)
            norm_grad = np.linalg.norm(grad)
            if norm_grad < 1e-10:
                break
            
            # Adaptive step size
            params_new = params - lr * grad / (norm_grad + 1e-10)
            e_new = _energy(params_new)
            
            if e_new < best_e:
                params = params_new
                best_e = e_new
                no_improve = 0
                lr = min(max_lr, lr * 1.1)
            else:
                no_improve += 1
                lr = max(min_lr, lr * 0.7)
            
            if no_improve > patience:
                break
        
        # Phase 2: L-BFGS-B for polishing
        if time.time() < deadline - 10.0:
            try:
                result = minimize(
                    _energy,
                    params,
                    jac=_energy_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.success and result.fun < best_e:
                    params = result.x
                    best_e = result.fun
            except Exception:
                pass
        
        # Extract and validate
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        # Clip to valid range
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        
        # Ensure radii are positive
        r = np.maximum(r, 1e-6)
        
        # Re-clip positions after radius adjustment
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        
        arr = np.column_stack([x, y, r])
        
        if _validate(arr):
            s = _score(arr)
            if s > best_score:
                best_score = s
                best_params = arr.copy()
    
    if best_params is not None:
        return best_params
    
    # Fallback: return a trivial valid configuration
    side = 5
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])


def construct():
    """Construct 26 circles packed in the unit square."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    try:
        result = _optimize()
    except Exception as e:
        # Fallback on any error
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    if time.time() > deadline:
        # Should not happen due to internal checks, but just in case
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return result

# EVOLVE-BLOCK-END
