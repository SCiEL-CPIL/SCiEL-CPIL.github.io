"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _validate(arr):
    """Validate a packing solution. Returns True if valid."""
    if arr is None or arr.shape != (N, 3):
        return False
    if not np.all(np.isfinite(arr)):
        return False
    
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    
    # Check radii are non-negative
    if np.any(r < -1e-9):
        return False
    
    # Check all circles are within bounds
    if np.any(x < -1e-7) or np.any(x > 2.0 + 1e-7):
        return False
    if np.any(y < -1e-7) or np.any(y > 2.0 + 1e-7):
        return False
    
    # Check non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.hypot(x[i] - x[j], y[i] - y[j])
            if dist < r[i] + r[j] - 1e-6:
                return False
    
    return True

def _compute_score(arr):
    """Compute sum of radii."""
    return np.sum(arr[:, 2])

def _hex_grid_init(w, h, seed=42):
    """Generate hexagonal lattice initialization."""
    rng = np.random.RandomState(seed)
    
    # Try different hex spacings
    best_positions = None
    best_coverage = -1
    
    for attempt in range(3):
        # Estimate spacing from area
        area = w * h
        # For hex packing, each circle occupies ~ (2*r)^2 * sqrt(3)/2
        # N circles: N * 2 * sqrt(3) * r^2 ≈ area
        # r ≈ sqrt(area / (N * 2 * sqrt(3)))
        r_est = np.sqrt(area / (N * 2 * np.sqrt(3)))
        
        # Hex spacing
        dx = 2 * r_est
        dy = dx * np.sqrt(3) / 2
        
        # Add some randomness
        dx *= rng.uniform(0.9, 1.1)
        dy *= rng.uniform(0.9, 1.1)
        
        positions = []
        row = 0
        while len(positions) < N:
            offset = 0 if row % 2 == 0 else dx / 2
            for col in range(10):
                if len(positions) >= N:
                    break
                x = dx / 2 + col * dx + offset
                y = dy / 2 + row * dy
                # Add jitter
                x += rng.uniform(-0.1 * dx, 0.1 * dx)
                y += rng.uniform(-0.1 * dy, 0.1 * dy)
                x = np.clip(x, 0.01, w - 0.01)
                y = np.clip(y, 0.01, h - 0.01)
                positions.append([x, y])
            row += 1
        
        positions = np.array(positions[:N])
        
        # Compute coverage (sum of radii we could fit)
        radii = np.zeros(N)
        for i in range(N):
            max_r = min(positions[i, 0], w - positions[i, 0], 
                        positions[i, 1], h - positions[i, 1])
            for j in range(N):
                if i != j:
                    dist = np.hypot(positions[i, 0] - positions[j, 0],
                                   positions[i, 1] - positions[j, 1])
                    max_r = min(max_r, dist / 2)
            radii[i] = max_r
        
        coverage = np.sum(radii)
        if coverage > best_coverage:
            best_coverage = coverage
            best_positions = positions.copy()
    
    return best_positions

def _solve_aspect(w, h, n_restarts=5, time_limit=15.0):
    """Solve for given aspect ratio using hex init + SLSQP."""
    start_time = time.time()
    best_local = None
    best_local_score = -1.0
    
    for restart in range(n_restarts):
        if time.time() > start_time + time_limit:
            break
        
        # Hexagonal initialization
        positions = _hex_grid_init(w, h, seed=42 + restart)
        
        # Initial radii
        radii = np.zeros(N)
        for i in range(N):
            max_r = min(positions[i, 0], w - positions[i, 0], 
                        positions[i, 1], h - positions[i, 1])
            for j in range(N):
                if i != j:
                    dist = np.hypot(positions[i, 0] - positions[j, 0],
                                   positions[i, 1] - positions[j, 1])
                    max_r = min(max_r, dist / 2)
            radii[i] = max(0.01, max_r * 0.8)
        
        # SLSQP optimization
        def objective(vars):
            pos = vars[:2*N].reshape(N, 2)
            rad = vars[2*N:]
            
            # Wall violations
            wall_viol = 0.0
            for i in range(N):
                wall_viol += max(0, rad[i] - pos[i, 0])
                wall_viol += max(0, pos[i, 0] - (w - rad[i]))
                wall_viol += max(0, rad[i] - pos[i, 1])
                wall_viol += max(0, pos[i, 1] - (h - rad[i]))
            
            # Overlap violations
            overlap_viol = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                    overlap_viol += max(0, rad[i] + rad[j] - d)
            
            penalty = 100.0 * (wall_viol + overlap_viol)
            return -np.sum(rad) + penalty
        
        def jac(vars):
            pos = vars[:2*N].reshape(N, 2)
            rad = vars[2*N:]
            
            grad = np.zeros_like(vars)
            # Gradient of -sum(rad)
            grad[2*N:] = -1.0
            
            # Gradient of wall penalty
            for i in range(N):
                # x walls
                if pos[i, 0] < rad[i]:
                    grad[2*i] += 100.0  # d(pos)/d(x)
                    grad[2*N + i] += -100.0  # d(rad)/d(x)
                if pos[i, 0] > w - rad[i]:
                    grad[2*i] += -100.0
                    grad[2*N + i] += -100.0
                # y walls
                if pos[i, 1] < rad[i]:
                    grad[2*i + 1] += 100.0
                    grad[2*N + i] += -100.0
                if pos[i, 1] > h - rad[i]:
                    grad[2*i + 1] += -100.0
                    grad[2*N + i] += -100.0
            
            # Gradient of overlap penalty
            for i in range(N):
                for j in range(i + 1, N):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    d = np.hypot(dx, dy)
                    if d < 1e-10:
                        continue
                    viol = rad[i] + rad[j] - d
                    if viol > 0:
                        grad[2*i] += 100.0 * dx / d
                        grad[2*i + 1] += 100.0 * dy / d
                        grad[2*j] += -100.0 * dx / d
                        grad[2*j + 1] += -100.0 * dy / d
                        grad[2*N + i] += 100.0
                        grad[2*N + j] += 100.0
            
            return grad
        
        x0 = np.concatenate([positions.flatten(), radii])
        bounds = []
        for i in range(N):
            bounds.append((0.0, w))  # x
            bounds.append((0.0, h))  # y
            bounds.append((0.001, min(w, h) / 2))  # radius
        
        try:
            res = minimize(objective, x0, method='SLSQP', jac=jac,
                          bounds=bounds, options={'maxiter': 200, 'ftol': 1e-9, 'disp': False})
            
            if res.fun < 0:  # Negative means we got a valid solution
                pos_opt = res.x[:2*N].reshape(N, 2)
                rad_opt = res.x[2*N:]
                
                # Clip to bounds
                for i in range(N):
                    pos_opt[i, 0] = np.clip(pos_opt[i, 0], 0, w)
                    pos_opt[i, 1] = np.clip(pos_opt[i, 1], 0, h)
                    rad_opt[i] = max(0, rad_opt[i])
                
                result = np.column_stack([pos_opt, rad_opt])
                
                if _validate(result):
                    score = _compute_score(result)
                    if score > best_local_score:
                        best_local_score = score
                        best_local = result.copy()
        except Exception:
            pass
        
        # Also try the initial positions with computed radii
        result_init = np.column_stack([positions, radii])
        if _validate(result_init):
            score = _compute_score(result_init)
            if score > best_local_score:
                best_local_score = score
                best_local = result_init.copy()
    
    return best_local, best_local_score

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    best_result = None
    best_score = -1.0
    
    # Candidate aspect ratios (w/h)
    aspect_ratios = [
        0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.8, 2.0, 2.5, 3.0
    ]
    
    # Sort by expected quality (closer to 1 might be better for 21 circles)
    # But let's try a range
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        h = 2.0 / (ar + 1.0)
        w = ar * h
        
        # Adjust time limit based on remaining time
        remaining = deadline - time.time()
        time_limit = min(15.0, remaining * 0.6)
        
        if time_limit < 2.0:
            break
        
        local_result, local_score = _solve_aspect(w, h, n_restarts=4, time_limit=time_limit)
        
        if local_result is not None and local_score > best_score:
            best_score = local_score
            best_result = local_result.copy()
    
    # Fallback: simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side * 0.85
            cells.append([x, y, r])
        return np.array(cells)
    
    return best_result

# EVOLVE-BLOCK-END
