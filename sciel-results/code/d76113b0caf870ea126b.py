"""Circle packing in unit square n=32: multi-start phyllotaxis + L-BFGS-B + repair."""
import numpy as np
import time
from scipy.optimize import minimize

N = 32
DEADLINE = 150.0  # seconds, evaluator kills at 150s
SAFETY = 5.0      # return this many seconds before deadline

def _phyllotaxis_positions(n, seed=0):
    """Generate n points in [0,1]^2 using phyllotaxis (low-discrepancy spiral)."""
    rng = np.random.default_rng(seed)
    # Golden angle
    angle = 2.0 * np.pi * (1.0 - 1.0 / ((1.0 + np.sqrt(5.0)) / 2.0))
    # Slight random rotation to break symmetry
    rot = rng.uniform(0, 2 * np.pi)
    idx = np.arange(n)
    r = np.sqrt((idx + 0.5) / n) * 0.48  # keep inside [0,1]^2 with margin
    theta = idx * angle + rot
    x = 0.5 + r * np.cos(theta)
    y = 0.5 + r * np.sin(theta)
    # Clamp to [0.05, 0.95] to keep centers away from boundary initially
    x = np.clip(x, 0.05, 0.95)
    y = np.clip(y, 0.05, 0.95)
    return np.column_stack([x, y])

def _objective_and_grad(params, n, penalty_weight=100.0, boundary_weight=100.0):
    """
    params: (2n + n) = 3n vector: x, y, r
    Objective: minimize -sum(r) + penalty(overlaps) + penalty(boundary)
    """
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:]
    
    # Ensure radii positive (soft)
    r_eff = np.maximum(r, 1e-6)
    
    # Sum of radii
    sum_r = np.sum(r_eff)
    
    # Overlap penalty: for each pair, if dist < r_i + r_j, penalize
    # Use soft min: max(0, r_i + r_j - dist)^2
    overlap_penalty = 0.0
    overlap_grad_x = np.zeros(n)
    overlap_grad_y = np.zeros(n)
    overlap_grad_r = np.zeros(n)
    
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy + 1e-12)
            sep = r_eff[i] + r_eff[j] - dist
            if sep > 0:
                # Gradient of sep^2 w.r.t. positions and radii
                # d(sep^2)/dx_i = 2*sep * d(sep)/dx_i = 2*sep * (dx/dist)
                factor = 2.0 * sep / dist
                overlap_grad_x[i] += factor * dx
                overlap_grad_x[j] -= factor * dx
                overlap_grad_y[i] += factor * dy
                overlap_grad_y[j] -= factor * dy
                # d(sep)/dr_i = 1, so d(sep^2)/dr_i = 2*sep
                overlap_grad_r[i] += 2.0 * sep
                overlap_grad_r[j] += 2.0 * sep
                overlap_penalty += sep * sep
    
    # Boundary penalty: center must be at least r from each wall
    # For left wall: x - r >= 0, penalty if x < r
    # For right wall: 1 - x - r >= 0, penalty if x + r > 1
    # For bottom: y - r >= 0
    # For top: 1 - y - r >= 0
    boundary_penalty = 0.0
    boundary_grad_x = np.zeros(n)
    boundary_grad_y = np.zeros(n)
    boundary_grad_r = np.zeros(n)
    
    for i in range(n):
        # Left
        viol = r_eff[i] - x[i]
        if viol > 0:
            boundary_penalty += viol * viol
            boundary_grad_x[i] += 2.0 * viol * (-1.0)
            boundary_grad_r[i] += 2.0 * viol
        # Right
        viol = x[i] + r_eff[i] - 1.0
        if viol > 0:
            boundary_penalty += viol * viol
            boundary_grad_x[i] += 2.0 * viol
            boundary_grad_r[i] += 2.0 * viol
        # Bottom
        viol = r_eff[i] - y[i]
        if viol > 0:
            boundary_penalty += viol * viol
            boundary_grad_y[i] += 2.0 * viol * (-1.0)
            boundary_grad_r[i] += 2.0 * viol
        # Top
        viol = y[i] + r_eff[i] - 1.0
        if viol > 0:
            boundary_penalty += viol * viol
            boundary_grad_y[i] += 2.0 * viol
            boundary_grad_r[i] += 2.0 * viol
    
    obj = -sum_r + penalty_weight * overlap_penalty + boundary_weight * boundary_penalty
    
    grad = np.zeros(3 * n)
    grad[:n] = -1.0 + penalty_weight * overlap_grad_x + boundary_weight * boundary_grad_x
    grad[n:2*n] = -1.0 + penalty_weight * overlap_grad_y + boundary_weight * boundary_grad_y
    grad[2*n:] = -1.0 + penalty_weight * overlap_grad_r + boundary_weight * boundary_grad_r
    
    return obj, grad

def _is_feasible(coords, tol=1e-4):
    """Check if circles are disjoint and inside unit square."""
    n = len(coords)
    x, y, r = coords[:, 0], coords[:, 1], coords[:, 2]
    # Boundary
    if np.any(x < -tol) or np.any(y < -tol) or np.any(x + r > 1 + tol) or np.any(y + r > 1 + tol):
        return False
    # Overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _repair(coords, max_iter=1000):
    """Iteratively shrink and push apart to ensure feasibility."""
    coords = coords.copy()
    n = len(coords)
    for _ in range(max_iter):
        changed = False
        # Boundary repair: shrink radius if touching wall
        for i in range(n):
            x, y, r = coords[i]
            max_r = min(x, 1 - x, y, 1 - y)
            if r > max_r + 1e-6:
                coords[i, 2] = max_r
                changed = True
        # Overlap repair: for each overlapping pair, shrink the smaller one
        for i in range(n):
            for j in range(i+1, n):
                dx = coords[i, 0] - coords[j, 0]
                dy = coords[i, 1] - coords[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                required = coords[i, 2] + coords[j, 2]
                if dist < required - 1e-6:
                    # Shrink the smaller radius
                    if coords[i, 2] <= coords[j, 2]:
                        coords[i, 2] = max(0.0, (dist - coords[j, 2]) * 0.999)
                    else:
                        coords[j, 2] = max(0.0, (dist - coords[i, 2]) * 0.999)
                    changed = True
        if not changed:
            break
    return coords

def construct():
    start_time = time.time()
    best_coords = None
    best_score = -np.inf
    
    # Initial phyllotaxis-based seed
    initial = _phyllotaxis_positions(N, seed=42)
    # Initial radii: small, will grow
    init_r = np.full(N, 0.05)
    init_params = np.concatenate([initial[:, 0], initial[:, 1], init_r])
    
    # Multiple restarts with different phyllotaxis rotations and random perturbations
    n_restarts = 8
    rng = np.random.default_rng(123)
    
    for restart in range(n_restarts):
        if time.time() - start_time > DEADLINE - SAFETY:
            break
        
        # Create perturbed initial params
        if restart == 0:
            params = init_params.copy()
        else:
            # New phyllotaxis with different rotation
            new_init = _phyllotaxis_positions(N, seed=42 + restart * 7)
            # Add random perturbation
            pert = rng.normal(0, 0.02, size=(N, 2))
            new_init[:, 0] += pert[:, 0]
            new_init[:, 1] += pert[:, 1]
            new_init = np.clip(new_init, 0.05, 0.95)
            params = np.concatenate([new_init[:, 0], new_init[:, 1], init_r])
        
        # Run L-BFGS-B
        try:
            result = minimize(
                _objective_and_grad,
                params,
                args=(N,),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 50}
            )
        except Exception:
            continue
        
        if result.success or result.fun < best_score:
            # Extract coords
            res_params = result.x
            coords = np.column_stack([res_params[:N], res_params[N:2*N], res_params[2*N:]])
            # Ensure radii positive
            coords[:, 2] = np.maximum(coords[:, 2], 0.0)
            # Repair
            coords = _repair(coords)
            # Check feasibility
            if _is_feasible(coords):
                score = np.sum(coords[:, 2])
                if score > best_score:
                    best_score = score
                    best_coords = coords.copy()
    
    # If no feasible solution found, use a safe fallback
    if best_coords is None:
        # Fallback: small grid of circles
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        best_coords = np.array(cells[:N])
    
    # Final check and return
    if not _is_feasible(best_coords):
        best_coords = _repair(best_coords)
    
    return best_coords
