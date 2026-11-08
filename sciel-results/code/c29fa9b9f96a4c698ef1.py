"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _grid_initial():
    """Create a grid-based initial configuration of N circles."""
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95
            positions.append([x, y, r])
    
    while len(positions) < N:
        idx = len(positions)
        i = idx % side
        j = idx // side
        if j >= side:
            break
        x = (i + 0.5) / side
        y = (j + 0.5) / side
        r = 0.5 / side * 0.95
        positions.append([x, y, r])
    
    return np.array(positions[:N])


def _hex_initial():
    """Create a hexagonal packing initial configuration."""
    positions = []
    # Hexagonal packing: rows offset
    # For 26 circles, try to fit in a hexagonal pattern
    # Use a density-based approach
    # Approximate: sqrt(26) ~ 5.1, so about 5x5 grid with hex offset
    
    # Try different configurations
    best_hex = None
    best_score = -1
    
    for rows in range(3, 7):
        cols = int(np.ceil(N / rows))
        if cols < 3:
            continue
        
        spacing_x = 1.0 / (cols + 0.5)  # extra space for offset
        spacing_y = 1.0 / (rows + 0.5)
        
        pos = []
        for row in range(rows):
            for col in range(cols):
                if len(pos) >= N:
                    break
                x = (col + 0.5) * spacing_x + (spacing_x * 0.5 if row % 2 else 0)
                y = (row + 0.5) * spacing_y
                if x < 1.0 and y < 1.0:
                    r = min(spacing_x, spacing_y) * 0.48
                    pos.append([x, y, r])
        
        if len(pos) >= N:
            pos = pos[:N]
            # Score by sum of radii
            score = sum(p[2] for p in pos)
            if score > best_score:
                best_score = score
                best_hex = pos
    
    if best_hex is None:
        return _grid_initial()
    
    return np.array(best_hex)


def _objective_and_gradient(params, N):
    """
    Objective: minimize negative sum of radii with smooth penalty constraints.
    params: flattened array of [x1, y1, r1, x2, y2, r2, ...]
    """
    n = N
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    obj = -np.sum(r)
    
    grad = np.zeros_like(params)
    grad[2::3] = -1.0
    
    penalty = 0.0
    penalty_grad = np.zeros_like(params)
    
    # Boundary penalties (smooth)
    for i in range(n):
        # x constraints
        for val in [x[i], 1 - x[i]]:
            viol = r[i] - val
            if viol > 0:
                penalty += 1000 * viol**2
                penalty_grad[2*i + 2] += 2000 * viol
                if val == x[i]:
                    penalty_grad[i] -= 2000 * viol
                else:
                    penalty_grad[i] += 2000 * viol
        
        # y constraints
        for val in [y[i], 1 - y[i]]:
            viol = r[i] - val
            if viol > 0:
                penalty += 1000 * viol**2
                penalty_grad[2*i + 2] += 2000 * viol
                if val == y[i]:
                    penalty_grad[i + n] -= 2000 * viol
                else:
                    penalty_grad[i + n] += 2000 * viol
    
    # Circle-circle penalties (smooth)
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            dist = np.sqrt(dist_sq + 1e-12)
            if dist < min_dist:
                viol = min_dist - dist
                penalty += 1000 * viol**2
                coeff = 2000 * viol / dist
                penalty_grad[i] += coeff * dx
                penalty_grad[j] -= coeff * dx
                penalty_grad[i + n] += coeff * dy
                penalty_grad[j + n] -= coeff * dy
                penalty_grad[2*i + 2] += 2000 * viol
                penalty_grad[2*j + 2] += 2000 * viol
    
    total_obj = obj + penalty
    total_grad = grad + penalty_grad
    
    return total_obj, total_grad


def _check_validity(pos, radii, tol=1e-7):
    """Check if configuration is valid."""
    n = len(pos)
    if np.any(pos[:, 0] < radii - tol) or np.any(pos[:, 0] > 1 - radii + tol):
        return False
    if np.any(pos[:, 1] < radii - tol) or np.any(pos[:, 1] > 1 - radii + tol):
        return False
    if np.any(radii < -tol):
        return False
    for i in range(n):
        dx = pos[i, 0] - pos[i+1:, 0]
        dy = pos[i, 1] - pos[i+1:, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        if np.any(dist < radii[i] + radii[i+1:] - tol):
            return False
    return True


def _fix_configuration(x, y, r):
    """Fix any invalidities in the configuration."""
    n = len(r)
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    # Clip to boundaries
    for i in range(n):
        r[i] = min(r[i], x[i], 1 - x[i], y[i], 1 - y[i])
        r[i] = max(r[i], 0.001)
        x[i] = max(r[i], min(1 - r[i], x[i]))
        y[i] = max(r[i], min(1 - r[i], y[i]))
    
    # Resolve overlaps by reducing radii
    for _ in range(10):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    # Reduce the smaller radius
                    if r[i] < r[j]:
                        new_r = max(0.001, (dist - r[j]) * 0.99)
                        if new_r < r[i]:
                            r[i] = new_r
                            changed = True
                    else:
                        new_r = max(0.001, (dist - r[i]) * 0.99)
                        if new_r < r[j]:
                            r[j] = new_r
                            changed = True
        if not changed:
            break
    
    # Final boundary check
    for i in range(n):
        r[i] = min(r[i], x[i] - 1e-6, 1 - x[i] - 1e-6, y[i] - 1e-6, 1 - y[i] - 1e-6)
        r[i] = max(r[i], 0.001)
    
    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = N
    total_params = 3 * n
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.001, 0.5))  # r
    
    best_obj = np.inf
    best_params = None
    best_score = 0.0
    
    def _score(params):
        """Compute the actual score (sum of radii) for a parameter set."""
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        # Check validity
        if not _check_validity(np.column_stack([x, y]), r):
            return 0.0
        return np.sum(r)
    
    def _run_optimization(params_init, max_iter=500, time_budget=30.0):
        """Run L-BFGS-B optimization with time budget."""
        start_opt = time.time()
        
        def objective_wrapper(p):
            return _objective_and_gradient(p, n)[0]
        
        def jac_wrapper(p):
            return _objective_and_gradient(p, n)[1]
        
        result = minimize(
            objective_wrapper,
            params_init,
            jac=jac_wrapper,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50,
                     'maxfun': 5000}
        )
        
        elapsed = time.time() - start_opt
        return result
    
    # Try multiple initial configurations
    configs = []
    
    # 1. Grid initial
    init_grid = _grid_initial()
    configs.append(init_grid.flatten())
    
    # 2. Hex initial
    init_hex = _hex_initial()
    configs.append(init_hex.flatten())
    
    # 3. Random restarts
    np.random.seed(42)
    for restart in range(8):
        x_rand = np.random.uniform(0.15, 0.85, size=n)
        y_rand = np.random.uniform(0.15, 0.85, size=n)
        r_rand = np.random.uniform(0.03, 0.08, size=n)
        
        params_rand = np.zeros(3*n)
        params_rand[0::3] = x_rand
        params_rand[1::3] = y_rand
        params_rand[2::3] = r_rand
        configs.append(params_rand)
    
    # Run optimization on each config
    for idx, params_init in enumerate(configs):
        if time.time() > deadline - 15:
            break
        
        remaining = deadline - 15 - time.time()
        if remaining < 5:
            break
        
        # Allocate time budget
        time_budget = min(25.0, remaining * 0.8)
        
        try:
            result = _run_optimization(params_init, max_iter=300, time_budget=time_budget)
        except Exception:
            continue
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x.copy()
        
        # Compute actual score
        score = _score(result.x)
        if score > best_score:
            best_score = score
            best_params = result.x.copy()
        
        # If we have time, do a polish step on the best so far
        if time.time() < deadline - 10 and best_params is not None:
            try:
                result2 = _run_optimization(best_params, max_iter=200, time_budget=10.0)
                score2 = _score(result2.x)
                if score2 > best_score:
                    best_score = score2
                    best_params = result2.x.copy()
            except Exception:
                pass
    
    # Final fix and validation
    if best_params is None:
        # Fallback to grid
        x, y, r = _grid_initial().T
    else:
        x = best_params[0::3]
        y = best_params[1::3]
        r = best_params[2::3]
    
    x, y, r = _fix_configuration(x, y, r)
    
    result_array = np.column_stack([x, y, r])
    
    return result_array

# EVOLVE-BLOCK-END
