"""
Circle packing in unit square n=32:
- Differentiable contact-graph constraint formulation (active only for near pairs)
- Multi-start from diverse low-discrepancy and grid initializations
- L-BFGS-B with soft penalty + active-set constraint handling
- Repair and polish stages
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 32
DEADLINE = 150.0
SAFETY = 5.0

def _phyllotaxis_positions(n, seed=0):
    rng = np.random.default_rng(seed)
    angle = 2.0 * np.pi * (1.0 - 1.0 / ((1.0 + np.sqrt(5.0)) / 2.0))
    rot = rng.uniform(0, 2 * np.pi)
    idx = np.arange(n)
    r = np.sqrt((idx + 0.5) / n) * 0.48
    theta = idx * angle + rot
    x = 0.5 + r * np.cos(theta)
    y = 0.5 + r * np.sin(theta)
    x = np.clip(x, 0.05, 0.95)
    y = np.clip(y, 0.05, 0.95)
    return np.column_stack([x, y])

def _grid_positions(n, rows=None, cols=None):
    if rows is None or cols is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    xs = np.linspace(0.5 / cols, 1 - 0.5 / cols, cols)
    ys = np.linspace(0.5 / rows, 1 - 0.5 / rows, rows)
    X, Y = np.meshgrid(xs, ys)
    x = X.flatten()[:n]
    y = Y.flatten()[:n]
    return np.column_stack([x, y])

def _hex_positions(n):
    # 4 rows of 8
    x = []
    y = []
    for i in range(4):
        offset = 0 if i % 2 == 0 else 1.0 / 16.0
        for j in range(8):
            x.append(offset + j * 0.125 + 0.0625)
            y.append(0.15 + i * 0.2165)
    x = np.array(x)
    y = np.array(y)
    return np.column_stack([x, y])

def _random_positions(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.1, 0.9, n)
    y = rng.uniform(0.1, 0.9, n)
    return np.column_stack([x, y])

def _objective_and_grad(params, n, penalty_weight=50.0, boundary_weight=50.0, contact_margin=0.02):
    """
    params: (3n) vector: x, y, r
    Objective: minimize -sum(r) + penalty(overlaps) + penalty(boundary)
    Only penalize pairs that are within contact_margin of touching.
    """
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:]
    r_eff = np.maximum(r, 1e-8)
    
    sum_r = np.sum(r_eff)
    
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
            # Only active if close to contact
            if sep > -contact_margin:
                # Smooth penalty: use max(0, sep)^2 but with smooth transition
                if sep > 0:
                    factor = 2.0 * sep / dist
                    overlap_grad_x[i] += factor * dx
                    overlap_grad_x[j] -= factor * dx
                    overlap_grad_y[i] += factor * dy
                    overlap_grad_y[j] -= factor * dy
                    overlap_grad_r[i] += 2.0 * sep
                    overlap_grad_r[j] += 2.0 * sep
                    overlap_penalty += sep * sep
                else:
                    # Small negative sep: use a soft hinge to keep gradient continuous
                    # This helps the optimizer know it's near contact
                    soft = 0.1 * sep * sep
                    overlap_penalty += soft
                    # Gradient of soft penalty is small
                    grad_soft = 0.2 * sep / dist
                    overlap_grad_x[i] += grad_soft * dx
                    overlap_grad_x[j] -= grad_soft * dx
                    overlap_grad_y[i] += grad_soft * dy
                    overlap_grad_y[j] -= grad_soft * dy
                    overlap_grad_r[i] += 0.2 * sep
                    overlap_grad_r[j] += 0.2 * sep
    
    boundary_penalty = 0.0
    boundary_grad_x = np.zeros(n)
    boundary_grad_y = np.zeros(n)
    boundary_grad_r = np.zeros(n)
    
    for i in range(n):
        viol = r_eff[i] - x[i]
        if viol > 0:
            boundary_penalty += viol * viol
            boundary_grad_x[i] += 2.0 * viol * (-1.0)
            boundary_grad_r[i] += 2.0 * viol
        viol = x[i] + r_eff[i] - 1.0
        if viol > 0:
            boundary_penalty += viol * viol
            boundary_grad_x[i] += 2.0 * viol
            boundary_grad_r[i] += 2.0 * viol
        viol = r_eff[i] - y[i]
        if viol > 0:
            boundary_penalty += viol * viol
            boundary_grad_y[i] += 2.0 * viol * (-1.0)
            boundary_grad_r[i] += 2.0 * viol
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
    n = len(coords)
    x, y, r = coords[:, 0], coords[:, 1], coords[:, 2]
    if np.any(x < -tol) or np.any(y < -tol) or np.any(x + r > 1 + tol) or np.any(y + r > 1 + tol):
        return False
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _repair(coords, max_iter=2000):
    coords = coords.copy()
    n = len(coords)
    for _ in range(max_iter):
        changed = False
        for i in range(n):
            x, y, r = coords[i]
            max_r = min(x, 1 - x, y, 1 - y)
            if r > max_r + 1e-7:
                coords[i, 2] = max_r
                changed = True
        for i in range(n):
            for j in range(i+1, n):
                dx = coords[i, 0] - coords[j, 0]
                dy = coords[i, 1] - coords[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                required = coords[i, 2] + coords[j, 2]
                if dist < required - 1e-7:
                    if coords[i, 2] <= coords[j, 2]:
                        coords[i, 2] = max(0.0, (dist - coords[j, 2]) * 0.9999)
                    else:
                        coords[j, 2] = max(0.0, (dist - coords[i, 2]) * 0.9999)
                    changed = True
        if not changed:
            break
    return coords

def _polish(coords, start_time, max_time=10.0):
    """Local polish: small perturbations + re-optimize."""
    n = len(coords)
    best = coords.copy()
    best_score = np.sum(best[:, 2])
    
    rng = np.random.default_rng(int(time.time() * 1000) % 10000)
    
    while time.time() - start_time < DEADLINE - SAFETY - max_time:
        # Try small random perturbations
        perturbed = best.copy()
        perturbed[:, 0] += rng.normal(0, 0.005, n)
        perturbed[:, 1] += rng.normal(0, 0.005, n)
        perturbed[:, 0] = np.clip(perturbed[:, 0], 0.01, 0.99)
        perturbed[:, 1] = np.clip(perturbed[:, 1], 0.01, 0.99)
        
        params = np.concatenate([perturbed[:, 0], perturbed[:, 1], perturbed[:, 2]])
        
        try:
            result = minimize(
                _objective_and_grad,
                params,
                args=(n,),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30}
            )
            res_params = result.x
            coords_test = np.column_stack([res_params[:n], res_params[n:2*n], res_params[2*n:]])
            coords_test[:, 2] = np.maximum(coords_test[:, 2], 0.0)
            coords_test = _repair(coords_test)
            
            if _is_feasible(coords_test):
                score = np.sum(coords_test[:, 2])
                if score > best_score + 1e-6:
                    best = coords_test.copy()
                    best_score = score
        except Exception:
            continue
    
    return best

def construct():
    start_time = time.time()
    best_coords = None
    best_score = -np.inf
    
    # Generate diverse initializations
    initializations = []
    
    # Phyllotaxis with different seeds
    for seed in [42, 123, 456, 789]:
        pos = _phyllotaxis_positions(N, seed=seed)
        r_init = np.full(N, 0.06)
        params = np.concatenate([pos[:, 0], pos[:, 1], r_init])
        initializations.append(params)
    
    # Grid
    pos = _grid_positions(N)
    r_init = np.full(N, 0.08)
    params = np.concatenate([pos[:, 0], pos[:, 1], r_init])
    initializations.append(params)
    
    # Hex
    pos = _hex_positions(N)
    r_init = np.full(N, 0.06)
    params = np.concatenate([pos[:, 0], pos[:, 1], r_init])
    initializations.append(params)
    
    # Random
    rng = np.random.default_rng(12345)
    for _ in range(4):
        pos = _random_positions(N, seed=rng.integers(0, 100000))
        r_init = np.full(N, 0.05)
        params = np.concatenate([pos[:, 0], pos[:, 1], r_init])
        initializations.append(params)
    
    # Perturbed grid
    pos = _grid_positions(N)
    pos[:, 0] += rng.normal(0, 0.02, N)
    pos[:, 1] += rng.normal(0, 0.02, N)
    pos = np.clip(pos, 0.05, 0.95)
    r_init = np.full(N, 0.08)
    params = np.concatenate([pos[:, 0], pos[:, 1], r_init])
    initializations.append(params)
    
    # Run optimizer on each initialization
    for idx, init_params in enumerate(initializations):
        if time.time() - start_time > DEADLINE - SAFETY - 15:
            break
        
        try:
            result = minimize(
                _objective_and_grad,
                init_params,
                args=(N,),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 1500, 'ftol': 1e-9, 'gtol': 1e-7, 'maxls': 40}
            )
        except Exception:
            continue
        
        res_params = result.x
        coords = np.column_stack([res_params[:N], res_params[N:2*N], res_params[2*N:]])
        coords[:, 2] = np.maximum(coords[:, 2], 0.0)
        coords = _repair(coords)
        
        if _is_feasible(coords):
            score = np.sum(coords[:, 2])
            if score > best_score:
                best_score = score
                best_coords = coords.copy()
    
    # Polish the best solution
    if best_coords is not None:
        best_coords = _polish(best_coords, start_time)
        if _is_feasible(best_coords):
            final_score = np.sum(best_coords[:, 2])
            if final_score > best_score:
                best_score = final_score
        else:
            best_coords = _repair(best_coords)
    
    # Fallback if no feasible solution
    if best_coords is None:
        pos = _grid_positions(N)
        r_init = np.full(N, 0.07)
        best_coords = np.column_stack([pos[:, 0], pos[:, 1], r_init])
        best_coords = _repair(best_coords)
    
    return best_coords
