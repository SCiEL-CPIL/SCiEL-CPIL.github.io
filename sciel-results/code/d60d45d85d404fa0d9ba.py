"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

def _is_valid(arr, tol=1e-9):
    """Check that all circles are inside [0,1]^2 with disjoint interiors."""
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r <= 0):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    # Check pairwise distances using vectorized computation
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    if np.any(dist < min_dist - tol):
        return False
    return True

def _score(arr):
    """Sum of radii if valid, else -inf."""
    if _is_valid(arr):
        return arr[:, 2].sum()
    return -np.inf

def _make_grid(n):
    """Create a grid of n circles inside the unit square."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append([cx, cy, r])
    return np.array(cells[:n])

def _make_hexagonal(n):
    """Create a hexagonal packing of n circles inside the unit square."""
    side = int(np.ceil(np.sqrt(n)))
    dx = 1.0 / side
    dy = dx * np.sqrt(3) / 2
    cells = []
    for row in range(side + 1):
        offset = (dx / 2) if (row % 2 == 1) else 0
        for col in range(side + 1):
            cx = offset + col * dx
            cy = row * dy
            if 0 <= cx <= 1 and 0 <= cy <= 1:
                r = min(cx, 1 - cx, cy, 1 - cy, dx / 2)
                if r > 0:
                    cells.append([cx, cy, r])
    cells.sort(key=lambda c: -c[2])
    return np.array(cells[:n])

def _make_center_ring(n):
    """Create a configuration with a large center circle and ring around it."""
    center_r = 0.2
    cx, cy = 0.5, 0.5
    cells = [[cx, cy, center_r]]
    remaining = n - 1
    # Ring around center
    ring_r = 0.08
    ring_dist = center_r + ring_r
    for i in range(remaining):
        angle = 2 * np.pi * i / remaining
        px = cx + ring_dist * np.cos(angle)
        py = cy + ring_dist * np.sin(angle)
        r = min(ring_r, px, 1 - px, py, 1 - py)
        if r > 0.01:
            cells.append([px, py, r])
    # Fill remaining with small circles
    while len(cells) < n:
        cells.append([0.1, 0.1, 0.05])
    return np.array(cells[:n])

def _make_random(n, seed):
    """Create a random configuration."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.1, 0.9, n)
    y = rng.uniform(0.1, 0.9, n)
    r = rng.uniform(0.05, 0.15, n)
    return np.column_stack([x, y, r])

def _objective(v, deadline, beta=50.0):
    """
    Smooth objective using log-sum-exp for penalties.
    v: flat array [x1..xN, y1..yN, r1..rN]
    """
    if time.time() > deadline:
        return 1e10
    
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:3*N]
    
    # Boundary violations (how much circle extends outside [0,1]^2)
    boundary_viol = np.maximum(0, r - x) + np.maximum(0, r - (1 - x)) + \
                    np.maximum(0, r - y) + np.maximum(0, r - (1 - y))
    
    # Negative radius violation
    neg_r = np.maximum(0, -r)
    
    # Overlap violations (how much circles overlap)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    overlap_viol = np.maximum(0, min_dist - dist)
    
    # Combine all violations
    all_viol = np.concatenate([boundary_viol, neg_r, overlap_viol.ravel()])
    
    # Log-sum-exp for smooth max (focuses on worst violations)
    if len(all_viol) > 0 and np.any(all_viol > 0):
        max_v = np.max(all_viol)
        if max_v > 1e-15:
            lse = max_v + np.log(np.sum(np.exp((all_viol - max_v) / beta))) * beta
        else:
            lse = 0.0
    else:
        lse = 0.0
    
    # Also add sum of squares for gradient signal
    pen = np.sum(all_viol**2)
    
    return -np.sum(r) + 10 * lse + 5 * pen

def _post_process(arr, deadline):
    """Ensure validity by clamping and shrinking."""
    if arr is None:
        return None
    
    x = arr[:, 0].copy()
    y = arr[:, 1].copy()
    r = arr[:, 2].copy()
    
    # Clamp to bounds
    for i in range(N):
        max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
        r[i] = max(1e-6, min(r[i], max_r))
        x[i] = np.clip(x[i], r[i], 1 - r[i])
        y[i] = np.clip(y[i], r[i], 1 - r[i])
    
    # Iterative shrink to resolve overlaps
    for _ in range(200):
        if time.time() > deadline:
            break
        invalid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                req = r[i] + r[j]
                if dist < req:
                    shrink = (req - dist) / 2 + 1e-8
                    r[i] -= shrink
                    r[j] -= shrink
                    invalid = True
        if not invalid:
            break
    
    # Final clamp
    for i in range(N):
        max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
        r[i] = max(1e-6, min(r[i], max_r))
    
    return np.column_stack([x, y, r])

def _slsqp_polish(arr, deadline, maxiter=500):
    """High-precision polish using SLSQP with explicit constraints."""
    x0 = arr.flatten()
    
    # Define constraints for SLSQP
    constraints = []
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, etc.
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[i] - v[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[N+i] - v[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*N+i]})
    
    # Pairwise distance constraints: dist - (r_i + r_j) >= 0
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: np.sqrt((v[i]-v[j])**2 + (v[N+i]-v[N+j])**2) - (v[2*N+i] + v[2*N+j])
            })
    
    bounds = [(0, 1)] * (2*N) + [(1e-6, 0.5)] * N
    
    try:
        result = minimize(lambda v: -np.sum(v[2*N:3*N]), x0, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': maxiter, 'ftol': 1e-12})
        
        if time.time() > deadline:
            return arr
        
        arr_out = np.column_stack([result.x[:N], result.x[N:2*N], result.x[2*N:3*N]])
        arr_out = _post_process(arr_out, deadline)
        return arr_out if arr_out is not None else arr
    except Exception:
        return arr

def _force_directed_anneal(arr, deadline, n_iter=2000, t_start=0.05, t_end=0.001):
    """Force-directed simulated annealing to escape local minima."""
    x = arr[:, 0].copy()
    y = arr[:, 1].copy()
    r = arr[:, 2].copy()
    
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    best_score = _score(np.column_stack([x, y, r]))
    
    t = t_start
    t_decay = (t_end / t_start) ** (1.0 / max(1, n_iter))
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Repulsion between overlapping circles
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy) + 1e-12
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    force = (min_dist - dist) / dist
                    fx[i] += force * dx
                    fy[i] += force * dy
                    fx[j] -= force * dx
                    fy[j] -= force * dy
        
        # Boundary forces (push inward if too close to edge)
        for i in range(N):
            margin = 0.02
            if x[i] - r[i] < margin:
                fx[i] += (margin - (x[i] - r[i]))
            if 1 - x[i] - r[i] < margin:
                fx[i] -= (margin - (1 - x[i] - r[i]))
            if y[i] - r[i] < margin:
                fy[i] += (margin - (y[i] - r[i]))
            if 1 - y[i] - r[i] < margin:
                fy[i] -= (margin - (1 - y[i] - r[i]))
        
        # Small random perturbation (annealing)
        noise_scale = t * 0.1
        fx += np.random.randn(N) * noise_scale
        fy += np.random.randn(N) * noise_scale
        
        # Update positions
        step = 0.01
        x += step * fx
        y += step * fy
        
        # Clamp to valid range
        for i in range(N):
            max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
            r[i] = max(1e-6, min(r[i], max_r))
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        
        # Try to grow radii slightly
        for i in range(N):
            max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
            r[i] = min(max_r, r[i] + 0.001)
        
        # Check score
        current_arr = np.column_stack([x, y, r])
        current_score = _score(current_arr)
        
        if current_score > best_score:
            best_score = current_score
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        # Cool down
        t *= t_decay
    
    return np.column_stack([best_x, best_y, best_r])

def _polish(arr, deadline):
    """Multi-stage polish: L-BFGS-B then SLSQP."""
    if arr is None:
        return None
    
    # Stage 1: L-BFGS-B with smooth objective
    x0 = arr.flatten()
    try:
        result = minimize(lambda v: _objective(v, deadline), x0, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-14})
        if time.time() > deadline:
            return arr
        arr1 = np.column_stack([result.x[:N], result.x[N:2*N], result.x[2*N:3*N]])
        arr1 = _post_process(arr1, deadline)
        if arr1 is not None:
            arr = arr1
    except Exception:
        pass
    
    # Stage 2: SLSQP with explicit constraints
    if time.time() < deadline - 10:
        arr = _slsqp_polish(arr, deadline)
    
    # Stage 3: Final L-BFGS-B from polished state
    if time.time() < deadline - 5 and arr is not None:
        x0 = arr.flatten()
        try:
            result = minimize(lambda v: _objective(v, deadline), x0, method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-14})
            if time.time() > deadline:
                return arr
            arr2 = np.column_stack([result.x[:N], result.x[N:2*N], result.x[2*N:3*N]])
            arr2 = _post_process(arr2, deadline)
            if arr2 is not None:
                arr = arr2
        except Exception:
            pass
    
    return arr

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit
    
    best_arr = None
    best_score = -np.inf
    
    def _update_best(arr):
        nonlocal best_arr, best_score
        if arr is None:
            return
        arr = _post_process(arr, deadline)
        if arr is None:
            return
        s = _score(arr)
        if s > best_score:
            best_score = s
            best_arr = arr.copy()
    
    # Initial configurations
    configs = []
    configs.append(_make_grid(N))
    configs.append(_make_hexagonal(N))
    configs.append(_make_center_ring(N))
    
    # Random configs with different seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        configs.append(_make_random(N, seed))
    
    # For each config: anneal then polish
    for cfg in configs:
        if time.time() > deadline:
            break
        
        # Force-directed annealing for global exploration
        annealed = _force_directed_anneal(cfg, deadline, n_iter=1000, t_start=0.03, t_end=0.0005)
        _update_best(annealed)
        
        # Polish the annealed result
        if time.time() < deadline - 15:
            polished = _polish(annealed, deadline)
            _update_best(polished)
        
        # Also try polishing the original config directly
        if time.time() < deadline - 15:
            polished_orig = _polish(cfg, deadline)
            _update_best(polished_orig)
    
    # Final high-precision polish on best result
    if time.time() < deadline - 10 and best_arr is not None:
        final = _slsqp_polish(best_arr, deadline, maxiter=1000)
        _update_best(final)
    
    # One more L-BFGS-B polish
    if time.time() < deadline - 5 and best_arr is not None:
        x0 = best_arr.flatten()
        try:
            result = minimize(lambda v: _objective(v, deadline), x0, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-15})
            if time.time() < deadline:
                arr_final = np.column_stack([result.x[:N], result.x[N:2*N], result.x[2*N:3*N]])
                _update_best(arr_final)
        except Exception:
            pass
    
    if best_arr is None:
        best_arr = _make_grid(N)
    
    return best_arr
