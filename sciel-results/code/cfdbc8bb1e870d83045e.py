"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26

# EVOLVE-BLOCK-START

def _check_validity(pos, radii):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    # Check inside square
    for i in range(n):
        if pos[i, 0] < radii[i] or pos[i, 0] > 1.0 - radii[i]:
            return False
        if pos[i, 1] < radii[i] or pos[i, 1] > 1.0 - radii[i]:
            return False
    # Check disjoint interiors
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((pos[i, 0] - pos[j, 0])**2 + (pos[i, 1] - pos[j, 1])**2)
            if dist < radii[i] + radii[j] - 1e-9:
                return False
    return True


def _objective(x):
    """Objective: minimize negative sum of radii + penalty for violations."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    # Penalty for going outside square
    penalty = 0.0
    for i in range(n):
        if pos[i, 0] < radii[i]:
            penalty += (radii[i] - pos[i, 0])**2
        if pos[i, 0] > 1.0 - radii[i]:
            penalty += (pos[i, 0] - (1.0 - radii[i]))**2
        if pos[i, 1] < radii[i]:
            penalty += (radii[i] - pos[i, 1])**2
        if pos[i, 1] > 1.0 - radii[i]:
            penalty += (pos[i, 1] - (1.0 - radii[i]))**2
    
    # Penalty for overlapping
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((pos[i, 0] - pos[j, 0])**2 + (pos[i, 1] - pos[j, 1])**2)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                penalty += (min_dist - dist)**2
    
    return -np.sum(radii) + 100.0 * penalty


def _make_initial_config(strategy, seed=42):
    """Create an initial configuration based on strategy."""
    rng = np.random.RandomState(seed)
    
    if strategy == 'grid':
        # 5x5 grid, use 25 cells, add one more
        side = 5
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                if idx < N:
                    pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                    radii[idx] = 0.5 / side - 0.005
        # Add 26th circle in a gap
        pos[25] = [0.5, 0.05]
        radii[25] = 0.04
        return pos, radii
    
    elif strategy == 'hexagonal':
        # Hexagonal packing pattern
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        r = 0.1
        rows = 5
        for row in range(rows):
            y = 0.1 + row * r * np.sqrt(3)
            if row % 2 == 0:
                xs = np.arange(5) * 2 * r + r
            else:
                xs = np.arange(5) * 2 * r + 2 * r
            for col, x in enumerate(xs):
                idx = row * 5 + col
                if idx < N:
                    pos[idx] = [x, y]
                    radii[idx] = r
        return pos, radii
    
    elif strategy == 'center_ring':
        # One center, rings around
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        pos[0] = [0.5, 0.5]
        radii[0] = 0.15
        
        # Inner ring: 8 circles
        r_inner = 0.12
        dist_inner = 0.3
        for i in range(8):
            angle = 2 * np.pi * i / 8
            pos[i+1] = [0.5 + dist_inner * np.cos(angle), 0.5 + dist_inner * np.sin(angle)]
            radii[i+1] = r_inner
        
        # Outer ring: 17 circles
        r_outer = 0.08
        dist_outer = 0.42
        for i in range(17):
            angle = 2 * np.pi * i / 17
            pos[i+9] = [0.5 + dist_outer * np.cos(angle), 0.5 + dist_outer * np.sin(angle)]
            radii[i+9] = r_outer
        
        return pos, radii
    
    elif strategy == 'random':
        # Random initial positions
        pos = rng.uniform(0.1, 0.9, size=(N, 2))
        radii = rng.uniform(0.03, 0.1, size=N)
        return pos, radii
    
    else:
        # Default: slightly perturbed grid
        pos, radii = _make_initial_config('grid', seed)
        pos += rng.normal(0, 0.02, size=pos.shape)
        pos = np.clip(pos, 0.05, 0.95)
        return pos, radii


def _post_process(pos, radii):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(100):
        valid = True
        # Check boundaries
        for i in range(N):
            if pos[i, 0] < radii[i]:
                radii[i] = pos[i, 0] * 0.99
                valid = False
            if pos[i, 0] > 1.0 - radii[i]:
                radii[i] = (1.0 - pos[i, 0]) * 0.99
                valid = False
            if pos[i, 1] < radii[i]:
                radii[i] = min(radii[i], pos[i, 1] * 0.99)
                valid = False
            if pos[i, 1] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 1]) * 0.99)
                valid = False
        
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((pos[i, 0] - pos[j, 0])**2 + (pos[i, 1] - pos[j, 1])**2)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Shrink both proportionally
                    scale = dist / (min_dist + 1e-12) * 0.99
                    radii[i] *= scale
                    radii[j] *= scale
                    valid = False
        
        if valid:
            break
    
    return pos, radii


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    strategies = ['grid', 'hexagonal', 'center_ring', 'random']
    
    for strategy in strategies:
        if time.time() > deadline - 20:
            break
        
        for seed in range(3):
            if time.time() > deadline - 15:
                break
            
            pos, radii = _make_initial_config(strategy, seed)
            
            # Combine into single vector
            x0 = np.concatenate([pos.flatten(), radii])
            
            # Bounds
            bounds = []
            for i in range(N):
                bounds.append((0.01, 0.99))  # x
                bounds.append((0.01, 0.99))  # y
            for i in range(N):
                bounds.append((0.01, 0.3))  # radius
            
            # Try L-BFGS-B first
            try:
                result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                                  options={'maxiter': 200, 'ftol': 1e-8})
                if result.success or result.fun < _objective(x0):
                    x_opt = result.x
                    pos_opt = x_opt[:2*N].reshape(N, 2)
                    radii_opt = x_opt[2*N:]
                    
                    # Post-process to ensure validity
                    pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                    score = np.sum(radii_pp)
                    
                    if _check_validity(pos_pp, radii_pp) and score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
            except Exception:
                pass
            
            # Try Nelder-Mead on best result
            if time.time() > deadline - 10:
                break
            
            try:
                x_start = np.concatenate([pos.flatten(), radii])
                result_nm = minimize(_objective, x_start, method='Nelder-Mead',
                                     options={'maxiter': 500, 'fatol': 1e-8})
                if result_nm.fun < _objective(x_start):
                    x_opt = result_nm.x
                    pos_opt = x_opt[:2*N].reshape(N, 2)
                    radii_opt = x_opt[2*N:]
                    
                    pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                    score = np.sum(radii_pp)
                    
                    if _check_validity(pos_pp, radii_pp) and score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
            except Exception:
                pass
    
    # If we didn't find a valid solution, fall back to simple grid
    if best_pos is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.005) for i in range(side) for j in range(side)]
        result = np.array(cells[:N])
        return result
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
