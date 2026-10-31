"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _objective(x, penalty_weight=500.0):
    """Objective: minimize negative sum of radii + smooth penalty for violations."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    radii = np.maximum(radii, 1e-4)
    
    penalty = 0.0
    # Boundary penalties (smooth)
    for i in range(n):
        penalty += max(0.0, radii[i] - pos[i, 0])**2
        penalty += max(0.0, pos[i, 0] - (1.0 - radii[i]))**2
        penalty += max(0.0, radii[i] - pos[i, 1])**2
        penalty += max(0.0, pos[i, 1] - (1.0 - radii[i]))**2
    
    # Pairwise overlap penalties (smooth)
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            min_dist = radii[i] + radii[j]
            min_dist_sq = min_dist * min_dist
            if dist_sq < min_dist_sq:
                # Use squared distance difference for smoother gradient
                penalty += (min_dist_sq - dist_sq)**2 / (4.0 * min_dist_sq)
    
    return -np.sum(radii) + penalty_weight * penalty


def _initial_grid():
    """Create a 5x5 grid of circles with the 26th placed in a corner gap."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Use slightly larger radius for better packing
    r = 0.095
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            pos[idx, 0] = (i + 0.5) / 5.0
            pos[idx, 1] = (j + 0.5) / 5.0
            radii[idx] = r
    
    # Place 26th circle in top-right corner area
    pos[25, 0] = 0.92
    pos[25, 1] = 0.92
    radii[25] = 0.07
    
    return pos, radii


def _initial_hexagonal():
    """Create a hexagonal packing pattern with optimized spacing."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Hexagonal packing: rows alternate between 5 and 4 circles
    # Use slightly larger radius for better packing
    r = 0.095
    x_offset = 0.09
    y_spacing = r * np.sqrt(3) * 0.98  # Slightly compressed vertically
    
    # Row 0: 5 circles
    for i in range(5):
        idx = i
        pos[idx, 0] = x_offset + i * 2 * r
        pos[idx, 1] = x_offset
        radii[idx] = r
    
    # Row 1: 4 circles (offset by r)
    for i in range(4):
        idx = 5 + i
        pos[idx, 0] = x_offset + r + i * 2 * r
        pos[idx, 1] = x_offset + y_spacing
        radii[idx] = r
    
    # Row 2: 5 circles
    for i in range(5):
        idx = 9 + i
        pos[idx, 0] = x_offset + i * 2 * r
        pos[idx, 1] = x_offset + 2 * y_spacing
        radii[idx] = r
    
    # Row 3: 4 circles
    for i in range(4):
        idx = 14 + i
        pos[idx, 0] = x_offset + r + i * 2 * r
        pos[idx, 1] = x_offset + 3 * y_spacing
        radii[idx] = r
    
    # Row 4: 5 circles
    for i in range(5):
        idx = 18 + i
        pos[idx, 0] = x_offset + i * 2 * r
        pos[idx, 1] = x_offset + 4 * y_spacing
        radii[idx] = r
    
    # 26th circle in top-right corner gap
    pos[25, 0] = 0.93
    pos[25, 1] = 0.93
    radii[25] = 0.065
    
    return pos, radii


def _initial_center_ring():
    """Center circle surrounded by a ring of circles."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Center circle
    pos[0, 0] = 0.5
    pos[0, 1] = 0.5
    radii[0] = 0.12
    
    # Ring of 12 circles around center
    ring_r = 0.08
    center_r = 0.12
    ring_dist = center_r + ring_r  # distance from center to ring centers
    
    for i in range(12):
        idx = 1 + i
        angle = 2 * np.pi * i / 12
        pos[idx, 0] = 0.5 + ring_dist * np.cos(angle)
        pos[idx, 1] = 0.5 + ring_dist * np.sin(angle)
        radii[idx] = ring_r
    
    # 13 more circles in corners and edges
    corner_r = 0.06
    corners = [
        (0.08, 0.08), (0.92, 0.08), (0.08, 0.92), (0.92, 0.92),
        (0.5, 0.05), (0.5, 0.95), (0.05, 0.5), (0.95, 0.5),
        (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8),
        (0.5, 0.5)  # This will overlap with center, will be fixed
    ]
    for i in range(13):
        idx = 13 + i
        pos[idx, 0] = corners[i][0]
        pos[idx, 1] = corners[i][1]
        radii[idx] = corner_r
    
    return pos, radii


def _initial_random(seed=42):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.15, 0.85, size=(N, 2))
    radii = rng.uniform(0.05, 0.12, size=N)
    return pos, radii


def _post_process(pos, radii):
    """Iteratively fix validity by shrinking the smaller circle when overlapping."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(500):
        valid = True
        
        # Fix boundary violations
        for i in range(N):
            max_r_x = min(pos[i, 0], 1.0 - pos[i, 0])
            max_r_y = min(pos[i, 1], 1.0 - pos[i, 1])
            max_r = max(min(max_r_x, max_r_y) * 0.999, 1e-6)
            if radii[i] > max_r:
                radii[i] = max_r
                valid = False
        
        # Fix pairwise overlaps: shrink the smaller circle
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Shrink the smaller circle to just fit
                    if radii[i] <= radii[j]:
                        # Shrink i so that ri + rj = dist * 0.999
                        target_sum = dist * 0.999
                        if target_sum > radii[j]:
                            radii[i] = max(target_sum - radii[j], 1e-6)
                        else:
                            # Both need to shrink
                            scale = target_sum / min_dist * 0.999
                            radii[i] = max(radii[i] * scale, 1e-6)
                            radii[j] = max(radii[j] * scale, 1e-6)
                    else:
                        # Shrink j so that ri + rj = dist * 0.999
                        target_sum = dist * 0.999
                        if target_sum > radii[i]:
                            radii[j] = max(target_sum - radii[i], 1e-6)
                        else:
                            # Both need to shrink
                            scale = target_sum / min_dist * 0.999
                            radii[i] = max(radii[i] * scale, 1e-6)
                            radii[j] = max(radii[j] * scale, 1e-6)
                    valid = False
        
        if valid:
            break
    
    return pos, radii


def _polish(pos, radii, deadline):
    """High-precision local search with multiple methods to squeeze out more sum of radii."""
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.5))
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Multiple L-BFGS-B passes with decreasing tolerances and increasing penalty
    configs = [
        (1e-12, 1e-8, 500.0),
        (1e-14, 1e-10, 1000.0),
        (1e-16, 1e-12, 2000.0),
    ]
    for ftol, gtol, pw in configs:
        if time.time() > deadline - 4:
            break
        try:
            result = minimize(lambda x: _objective(x, penalty_weight=pw), x0,
                             method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 2000, 'ftol': ftol, 'gtol': gtol})
            
            if result.fun < _objective(x0, penalty_weight=pw) - 1e-15:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
    
    # SLSQP with tight bounds for high-precision refinement
    if time.time() < deadline - 4:
        try:
            tight_bounds = []
            for i in range(N):
                x_lo = max(0.0, best_pos[i, 0] - 0.08)
                x_hi = min(1.0, best_pos[i, 0] + 0.08)
                y_lo = max(0.0, best_pos[i, 1] - 0.08)
                y_hi = min(1.0, best_pos[i, 1] + 0.08)
                tight_bounds.append((x_lo, x_hi))
                tight_bounds.append((y_lo, y_hi))
            for i in range(N):
                r_lo = max(1e-5, best_radii[i] * 0.7)
                r_hi = min(0.5, best_radii[i] * 1.3)
                tight_bounds.append((r_lo, r_hi))
            
            result = minimize(lambda x: _objective(x, penalty_weight=2000.0),
                             x0, method='SLSQP', bounds=tight_bounds,
                             options={'maxiter': 1500, 'ftol': 1e-14})
            
            if result.fun < _objective(x0, penalty_weight=2000.0) - 1e-15:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Final Powell pass for fine-tuning
    if time.time() < deadline - 2:
        try:
            result = minimize(lambda x: _objective(x, penalty_weight=2000.0), x0,
                             method='Powell', bounds=bounds,
                             options={'maxiter': 800, 'ftol': 1e-14})
            
            if result.fun < _objective(x0, penalty_weight=2000.0) - 1e-15:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    return best_pos, best_radii, best_score


def _optimize_from(pos, radii, deadline):
    """Run optimization from given initial configuration with time-aware allocation."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.25))
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # L-BFGS-B first (fast gradient-based) with moderate penalty
    if time.time() < deadline - 10:
        try:
            result = minimize(lambda x: _objective(x, penalty_weight=500.0), x0,
                             method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 1200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.fun < _objective(x0, penalty_weight=500.0) - 1e-15:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
    
    # Second L-BFGS-B with higher penalty for tighter constraints
    if time.time() < deadline - 8:
        try:
            result = minimize(lambda x: _objective(x, penalty_weight=1000.0), x0,
                             method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 1000, 'ftol': 1e-13, 'gtol': 1e-9})
            
            if result.fun < _objective(x0, penalty_weight=1000.0) - 1e-15:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
    
    # Powell (derivative-free, good for fine-tuning)
    if time.time() < deadline - 6:
        try:
            result = minimize(lambda x: _objective(x, penalty_weight=1000.0), x0,
                             method='Powell', bounds=bounds,
                             options={'maxiter': 600, 'ftol': 1e-13})
            
            if result.fun < _objective(x0, penalty_weight=1000.0) - 1e-15:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        best_pos, best_radii, best_score = _polish(best_pos, best_radii, deadline)
        
    return best_pos, best_radii, best_score


def _differential_evolution_run(deadline):
    """Run differential evolution for global search with time monitoring."""
    # Define bounds for DE
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))  # x
        bounds.append((0.01, 0.99))  # y
    for i in range(N):
        bounds.append((0.01, 0.25))  # radius
    
    # Use a subset of the time for DE
    de_deadline = deadline - 25
    
    try:
        result = differential_evolution(
            _objective, bounds,
            maxiter=80, popsize=12,
            tol=1e-7, mutation=(0.5, 1.5), recombination=0.7,
            seed=42, polish=False,
            updating='deferred',
            callback=lambda xk, convergence: (time.time() > de_deadline)
        )
        
        if result.fun < 0:  # Valid improvement
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-4)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp, score
    except Exception:
        pass
    
    return None, None, -np.inf


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # Build initial configurations
    init_configs = []
    
    pos, radii = _initial_grid()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_hexagonal()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_center_ring()
    init_configs.append((pos, radii))
    
    for seed in [42, 123, 456]:
        pos, radii = _initial_random(seed)
        init_configs.append((pos, radii))
    
    # Perturbed grid
    rng = np.random.RandomState(99)
    pos, radii = _initial_grid()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # Perturbed hexagonal
    pos, radii = _initial_hexagonal()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # Run optimization from each initial config with time-aware allocation
    # Prioritize better initial configs
    config_order = [0, 1, 5, 6, 2, 3, 4]  # grid, hex, perturbed_grid, perturbed_hex, ring, randoms
    for idx in config_order:
        if time.time() > deadline - 25:
            break
        if idx >= len(init_configs):
            continue
        
        pos, radii = init_configs[idx]
        pos_opt, radii_opt, score = _optimize_from(pos, radii, deadline)
        
        if score > best_score + 1e-12:
            best_score = score
            best_pos = pos_opt.copy()
            best_radii = radii_opt.copy()
    
    # Try differential evolution if time permits
    if time.time() < deadline - 35:
        de_pos, de_radii, de_score = _differential_evolution_run(deadline)
        if de_pos is not None and de_score > best_score + 1e-12:
            best_score = de_score
            best_pos = de_pos.copy()
            best_radii = de_radii.copy()
    
    # Fallback if nothing valid found
    if best_pos is None or best_score < 0:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(5):
            for j in range(5):
                idx2 = i * 5 + j
                if idx2 < N:
                    pos[idx2, 0] = (i + 0.5) / 5.0
                    pos[idx2, 1] = (j + 0.5) / 5.0
                    radii[idx2] = 0.09
        pos[25, 0] = 0.1
        pos[25, 1] = 0.9
        radii[25] = 0.07
        best_pos = pos
        best_radii = radii
    
    # Final high-precision polish
    if time.time() < deadline - 3:
        best_pos, best_radii, best_score = _polish(best_pos, best_radii, deadline)
    
    # Final post-processing to ensure validity
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
