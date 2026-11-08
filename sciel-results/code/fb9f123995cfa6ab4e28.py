"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _check_validity(pos, radii, tol=1e-8):
    """Check if packing is valid."""
    n = len(pos)
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        if x < -tol or x > 1 + tol or y < -tol or y > 1 + tol:
            return False
        if r < -tol:
            return False
        if x < r - tol or x > 1 - r + tol or y < r - tol or y > 1 - r + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[j][0] - pos[i][0]
            dy = pos[j][1] - pos[i][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _penalty_objective(vars, penalty_weight):
    """
    Objective: minimize -sum(radii) + penalty for constraint violations.
    vars: [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
    """
    n = len(vars) // 3
    pos = vars[:2*n].reshape(n, 2)
    radii = vars[2*n:]
    
    # Main objective: maximize sum of radii
    obj = -np.sum(radii)
    
    penalty = 0.0
    
    # Boundary penalties: circle must be inside [0,1]^2
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        # x boundary
        if x < r:
            penalty += penalty_weight * (r - x) ** 2
        if x > 1 - r:
            penalty += penalty_weight * (x - (1 - r)) ** 2
        # y boundary
        if y < r:
            penalty += penalty_weight * (r - y) ** 2
        if y > 1 - r:
            penalty += penalty_weight * (y - (1 - r)) ** 2
    
    # Overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[j][0] - pos[i][0]
            dy = pos[j][1] - pos[i][1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                penalty += penalty_weight * (min_dist - dist) ** 2
    
    return obj + penalty


def _penalty_gradient(vars, penalty_weight):
    """Analytical gradient of the penalty objective."""
    n = len(vars) // 3
    pos = vars[:2*n].reshape(n, 2)
    radii = vars[2*n:]
    
    grad = np.zeros_like(vars)
    
    # Gradient of -sum(radii)
    grad[2*n:] = -1.0
    
    # Boundary gradients
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        
        # x boundary
        if x < r:
            grad[2*i] += 2 * penalty_weight * (r - x)  # d/dx
            grad[2*n + i] += 2 * penalty_weight * (r - x)  # d/dr
        if x > 1 - r:
            grad[2*i] += 2 * penalty_weight * (x - (1 - r))  # d/dx
            grad[2*n + i] += -2 * penalty_weight * (x - (1 - r))  # d/dr
        
        # y boundary
        if y < r:
            grad[2*i + 1] += 2 * penalty_weight * (r - y)  # d/dy
            grad[2*n + i] += 2 * penalty_weight * (r - y)  # d/dr
        if y > 1 - r:
            grad[2*i + 1] += 2 * penalty_weight * (y - (1 - r))  # d/dy
            grad[2*n + i] += -2 * penalty_weight * (y - (1 - r))  # d/dr
    
    # Overlap gradients
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[j][0] - pos[i][0]
            dy = pos[j][1] - pos[i][1]
            dist_sq = dx * dx + dy * dy
            min_dist = radii[i] + radii[j]
            if dist_sq < min_dist * min_dist:
                dist = np.sqrt(dist_sq)
                if dist < 1e-10:
                    continue
                violation = min_dist - dist
                # d(violation)/dx_i = -dx/dist, d(violation)/dx_j = dx/dist
                # d(violation)/dr_i = 1, d(violation)/dr_j = 1
                # penalty = w * violation^2, d(penalty)/d(var) = 2*w*violation*d(violation)/d(var)
                factor = 2 * penalty_weight * violation / dist
                grad[2*i] += -factor * dx
                grad[2*i + 1] += -factor * dy
                grad[2*j] += factor * dx
                grad[2*j + 1] += factor * dy
                grad[2*n + i] += 2 * penalty_weight * violation
                grad[2*n + j] += 2 * penalty_weight * violation
    
    return grad


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Initialize with grid
    side = int(np.ceil(np.sqrt(N)))
    pos = np.array([[(i + 0.5) / side, (j + 0.5) / side] 
                    for i in range(side) for j in range(side)][:N])
    radii = np.full(N, 0.5 / side - 0.01)
    
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    # L-BFGS-B optimization with increasing penalty weight
    penalty_weights = [10, 50, 100, 500, 1000, 5000]
    
    for pw in penalty_weights:
        if time.time() - start_time > deadline:
            break
        
        # Build variable vector
        vars = np.concatenate([pos.flatten(), radii])
        
        # Bounds: positions in [0,1], radii in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0, 1))  # x
            bounds.append((0, 1))  # y
        for i in range(N):
            bounds.append((0, 0.5))  # radius
        
        try:
            result = minimize(
                _penalty_objective,
                vars,
                jac=_penalty_gradient,
                args=(pw,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 0:
                new_vars = result.x
                new_pos = new_vars[:2*N].reshape(N, 2)
                new_radii = new_vars[2*N:]
                
                # Clamp to valid range
                new_pos = np.clip(new_pos, 0, 1)
                new_radii = np.clip(new_radii, 0, 0.5)
                
                if _check_validity(new_pos, new_radii):
                    score = np.sum(new_radii)
                    if score > best_score:
                        best_score = score
                        best_pos = new_pos.copy()
                        best_radii = new_radii.copy()
                    pos = new_pos
                    radii = new_radii
        except Exception:
            pass
    
    # Final polish: try to improve with more iterations
    if time.time() - start_time < deadline - 5:
        for _ in range(50):
            if time.time() - start_time > deadline:
                break
            # Small random perturbations
            pos_perturbed = best_pos + np.random.normal(0, 0.005, size=(N, 2))
            pos_perturbed = np.clip(pos_perturbed, 0, 1)
            
            vars = np.concatenate([pos_perturbed.flatten(), best_radii])
            bounds = []
            for i in range(N):
                bounds.append((0, 1))
                bounds.append((0, 1))
            for i in range(N):
                bounds.append((0, 0.5))
            
            try:
                result = minimize(
                    _penalty_objective,
                    vars,
                    jac=_penalty_gradient,
                    args=(1000,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < 0:
                    new_vars = result.x
                    new_pos = new_vars[:2*N].reshape(N, 2)
                    new_radii = new_vars[2*N:]
                    new_pos = np.clip(new_pos, 0, 1)
                    new_radii = np.clip(new_radii, 0, 0.5)
                    
                    if _check_validity(new_pos, new_radii):
                        score = np.sum(new_radii)
                        if score > best_score:
                            best_score = score
                            best_pos = new_pos.copy()
                            best_radii = new_radii.copy()
            except Exception:
                pass
    
    # Return best result
    result_arr = np.zeros((N, 3))
    for i in range(N):
        result_arr[i] = [best_pos[i][0], best_pos[i][1], best_radii[i]]
    
    return result_arr
# EVOLVE-BLOCK-END
