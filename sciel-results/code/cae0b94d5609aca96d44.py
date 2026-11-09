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
    if np.any(r < 0):
        return False
    
    # Check all circles are within bounds
    if np.any(x < -1e-9) or np.any(x > 2.0 + 1e-9):
        return False
    if np.any(y < -1e-9) or np.any(y > 2.0 + 1e-9):
        return False
    
    # Check non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.hypot(x[i] - x[j], y[i] - y[j])
            if dist < r[i] + r[j] - 1e-7:
                return False
    
    return True

def _rect_dims(arr):
    """Get bounding rectangle dimensions from positions."""
    x, y = arr[:, 0], arr[:, 1]
    w = np.max(x) - np.min(x)
    h = np.max(y) - np.min(y)
    return w, h

def _compute_score(arr):
    """Compute sum of radii."""
    return np.sum(arr[:, 2])

def _solve_aspect(w, h, n_restarts=3, time_limit=30.0):
    """Solve for given aspect ratio using force-directed + SLSQP polish."""
    start_time = time.time()
    best_local = None
    best_local_score = -1.0
    
    cols = max(1, int(np.round(np.sqrt(N * w / h))))
    rows = max(1, int(np.ceil(N / cols)))
    while cols * rows < N:
        cols += 1
    
    for restart in range(n_restarts):
        if time.time() > start_time + time_limit:
            break
        
        # Initial positions: Grid layout with jitter
        positions = np.zeros((N, 2))
        for i in range(N):
            row = i // cols
            col = i % cols
            
            cell_w = w / cols
            cell_h = h / rows
            
            x = (col + 0.5) * cell_w
            y = (row + 0.5) * cell_h
            
            if restart > 0:
                jitter_scale = min(cell_w, cell_h) * 0.2
                x += np.random.uniform(-jitter_scale, jitter_scale)
                y += np.random.uniform(-jitter_scale, jitter_scale)
            
            x = np.clip(x, 0, w)
            y = np.clip(y, 0, h)
            positions[i] = [x, y]
        
        min_cell_dim = min(w/cols, h/rows)
        radii = np.full(N, min_cell_dim * 0.3)
        
        # Force-directed relaxation
        num_iterations = 150
        for it in range(num_iterations):
            if time.time() > start_time + time_limit - 2.0:
                break
            
            # Maximize radii
            new_radii = np.zeros(N)
            for i in range(N):
                max_r = min(positions[i, 0], w - positions[i, 0], positions[i, 1], h - positions[i, 1])
                for j in range(N):
                    if i == j:
                        continue
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    avail = dist - radii[j]
                    if avail < max_r:
                        max_r = avail
                new_radii[i] = max(0.0, max_r)
            radii = new_radii
            
            # Forces
            forces = np.zeros_like(positions)
            for i in range(N):
                if positions[i, 0] - radii[i] < 0:
                    forces[i, 0] += (positions[i, 0] - radii[i])
                if positions[i, 0] + radii[i] > w:
                    forces[i, 0] -= (positions[i, 0] + radii[i] - w)
                if positions[i, 1] - radii[i] < 0:
                    forces[i, 1] += (positions[i, 1] - radii[i])
                if positions[i, 1] + radii[i] > h:
                    forces[i, 1] -= (positions[i, 1] + radii[i] - h)
            
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dist_sq = dx*dx + dy*dy
                    min_dist = radii[i] + radii[j]
                    
                    if dist_sq < min_dist * min_dist:
                        dist = np.sqrt(dist_sq)
                        if dist > 1e-10:
                            overlap = min_dist - dist
                            fx = overlap * dx / dist
                            fy = overlap * dy / dist
                            forces[i, 0] -= fx
                            forces[i, 1] -= fy
                            forces[j, 0] += fx
                            forces[j, 1] += fy
                        else:
                            forces[i, 0] += np.random.randn() * 0.01
                            forces[i, 1] += np.random.randn() * 0.01
                            forces[j, 0] -= np.random.randn() * 0.01
                            forces[j, 1] -= np.random.randn() * 0.01
            
            lr = 0.5 * (0.95 ** (it / 10.0))
            lr = max(0.01, lr)
            positions += forces * lr
            
            for i in range(N):
                positions[i, 0] = np.clip(positions[i, 0], radii[i], w - radii[i])
                positions[i, 1] = np.clip(positions[i, 1], radii[i], h - radii[i])
        
        # SLSQP polish
        if time.time() < start_time + time_limit - 1.0:
            def objective(vars):
                pos = vars[:2*N].reshape(N, 2)
                rad = vars[2*N:]
                if np.any(rad < 0):
                    return -1e10
                # Wall constraints
                if np.any(pos[:, 0] < rad) or np.any(pos[:, 0] > w - rad):
                    return -1e10
                if np.any(pos[:, 1] < rad) or np.any(pos[:, 1] > h - rad):
                    return -1e10
                # Overlap constraints
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                        if d < rad[i] + rad[j]:
                            return -1e10
                return -np.sum(rad)
            
            def jac(vars):
                pos = vars[:2*N].reshape(N, 2)
                rad = vars[2*N:]
                j = np.zeros_like(vars)
                # Only gradient for radii is -1
                j[2*N:] = -1.0
                return j
            
            x0 = np.concatenate([positions.flatten(), radii])
            bounds = []
            for i in range(N):
                bounds.append((radii[i]*0.5, min(w, h)/2))  # x
                bounds.append((radii[i]*0.5, min(w, h)/2))  # y
                bounds.append((0.01, min(w, h)/2))  # radius
            
            try:
                res = minimize(objective, x0, method='SLSQP', jac=jac, 
                              bounds=bounds, options={'maxiter': 50, 'ftol': 1e-8})
                if res.success and res.fun > -1e9:
                    pos_opt = res.x[:2*N].reshape(N, 2)
                    rad_opt = res.x[2*N:]
                    if _validate(np.column_stack([pos_opt, rad_opt])):
                        score = _compute_score(np.column_stack([pos_opt, rad_opt]))
                        if score > best_local_score:
                            best_local_score = score
                            best_local = np.column_stack([pos_opt, rad_opt]).copy()
            except:
                pass
        
        # Validate force-directed result too
        if _validate(np.column_stack([positions, radii])):
            score = _compute_score(np.column_stack([positions, radii]))
            if score > best_local_score:
                best_local_score = score
                best_local = np.column_stack([positions, radii]).copy()
    
    return best_local, best_local_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_result = None
    best_score = -1.0
    
    # Candidate aspect ratios
    aspect_ratios = [
        0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.8, 2.0, 2.5, 3.0
    ]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        h = 2.0 / (ar + 1.0)
        w = ar * h
        
        local_result, local_score = _solve_aspect(w, h, n_restarts=3, time_limit=6.0)
        
        if local_result is not None and local_score > best_score:
            best_score = local_score
            best_result = local_result.copy()
    
    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side * 0.9
            cells.append([x, y, r])
        return np.array(cells)
    
    return best_result

# EVOLVE-BLOCK-END
