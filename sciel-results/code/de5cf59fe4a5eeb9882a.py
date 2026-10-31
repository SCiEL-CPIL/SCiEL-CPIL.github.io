"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import Delaunay

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


def _objective(x, penalty_weight=5000.0):
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
                penalty += (min_dist_sq - dist_sq)**2 / (4.0 * min_dist_sq)
    
    return -np.sum(radii) + penalty_weight * penalty


def _initial_hexagonal():
    """Create a hexagonal packing pattern."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.095
    y_spacing = r * np.sqrt(3)
    
    total_height = 4 * y_spacing + 2 * r
    y_start = (1.0 - total_height) / 2.0
    
    x_start_5 = (1.0 - 4 * 2 * r) / 2.0
    for i in range(5):
        idx = i
        pos[idx, 0] = x_start_5 + i * 2 * r
        pos[idx, 1] = y_start + r
        radii[idx] = r
    
    x_start_4 = (1.0 - 3 * 2 * r) / 2.0
    for i in range(4):
        idx = 5 + i
        pos[idx, 0] = x_start_4 + i * 2 * r
        pos[idx, 1] = y_start + r + y_spacing
        radii[idx] = r
    
    for i in range(5):
        idx = 9 + i
        pos[idx, 0] = x_start_5 + i * 2 * r
        pos[idx, 1] = y_start + r + 2 * y_spacing
        radii[idx] = r
    
    for i in range(4):
        idx = 14 + i
        pos[idx, 0] = x_start_4 + i * 2 * r
        pos[idx, 1] = y_start + r + 3 * y_spacing
        radii[idx] = r
    
    for i in range(5):
        idx = 18 + i
        pos[idx, 0] = x_start_5 + i * 2 * r
        pos[idx, 1] = y_start + r + 4 * y_spacing
        radii[idx] = r
    
    # 26th circle: find best position in gaps
    best_score = -1
    best_pos = (0.5, 0.5)
    for cx in np.linspace(0.05, 0.95, 20):
        for cy in np.linspace(0.05, 0.95, 20):
            min_clearance = 1.0
            for i in range(25):
                dist = np.sqrt((cx - pos[i, 0])**2 + (cy - pos[i, 1])**2)
                clearance = dist - radii[i]
                min_clearance = min(min_clearance, clearance)
            min_clearance = min(min_clearance, cx, 1-cx, cy, 1-cy)
            if min_clearance > best_score:
                best_score = min_clearance
                best_pos = (cx, cy)
    
    pos[25, 0] = best_pos[0]
    pos[25, 1] = best_pos[1]
    radii[25] = max(best_score * 0.9, 0.02)
    
    return pos, radii


def _initial_grid():
    """Create a 5x5 grid of circles with the 26th placed in a corner gap."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.095
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            pos[idx, 0] = (i + 0.5) / 5.0
            pos[idx, 1] = (j + 0.5) / 5.0
            radii[idx] = r
    
    pos[25, 0] = 0.92
    pos[25, 1] = 0.92
    radii[25] = 0.07
    
    return pos, radii


def _initial_corner_heavy():
    """Corner-heavy packing: larger circles in corners, smaller in center."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 4 corners with larger circles
    corner_r = 0.11
    corners = [(0.12, 0.12), (0.88, 0.12), (0.12, 0.88), (0.88, 0.88)]
    for i, (cx, cy) in enumerate(corners):
        pos[i, 0] = cx
        pos[i, 1] = cy
        radii[i] = corner_r
    
    # 4 edge midpoints
    edge_r = 0.095
    edges = [(0.5, 0.08), (0.5, 0.92), (0.08, 0.5), (0.92, 0.5)]
    for i, (cx, cy) in enumerate(edges):
        idx = 4 + i
        pos[idx, 0] = cx
        pos[idx, 1] = cy
        radii[idx] = edge_r
    
    # Center and inner ring
    center_r = 0.085
    pos[8, 0] = 0.5
    pos[8, 1] = 0.5
    radii[8] = center_r
    
    # 8 inner ring circles
    ring_r = 0.075
    ring_dist = 0.18
    for i in range(8):
        idx = 9 + i
        angle = 2 * np.pi * i / 8
        pos[idx, 0] = 0.5 + ring_dist * np.cos(angle)
        pos[idx, 1] = 0.5 + ring_dist * np.sin(angle)
        radii[idx] = ring_r
    
    # Remaining 9 circles scattered
    remaining = [
        (0.25, 0.25, 0.065), (0.75, 0.25, 0.065), (0.25, 0.75, 0.065), (0.75, 0.75, 0.065),
        (0.35, 0.15, 0.055), (0.65, 0.15, 0.055), (0.35, 0.85, 0.055), (0.65, 0.85, 0.055),
        (0.15, 0.35, 0.055)
    ]
    for i, (cx, cy, r) in enumerate(remaining):
        idx = 17 + i
        pos[idx, 0] = cx
        pos[idx, 1] = cy
        radii[idx] = r
    
    return pos, radii


def _initial_random(seed=42):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.15, 0.85, size=(N, 2))
    radii = rng.uniform(0.05, 0.12, size=N)
    return pos, radii


def _post_process(pos, radii):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(500):
        valid = True
        
        for i in range(N):
            if pos[i, 0] < radii[i]:
                radii[i] = max(pos[i, 0] * 0.99, 1e-4)
                valid = False
            if pos[i, 0] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 0]) * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
            if pos[i, 1] < radii[i]:
                radii[i] = min(radii[i], pos[i, 1] * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
            if pos[i, 1] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 1]) * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    scale = dist / (min_dist + 1e-12) * 0.99
                    radii[i] = max(radii[i] * scale, 1e-4)
                    radii[j] = max(radii[j] * scale, 1e-4)
                    valid = False
        
        if valid:
            break
    
    return pos, radii


def _grow_and_push(pos, radii, deadline):
    """Iteratively try to grow circles and push others away."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for i in range(N):
            # Try to grow circle i
            old_r = radii[i]
            new_r = old_r * 1.01
            
            # Check if it fits
            fits = True
            if pos[i, 0] - new_r < 0 or pos[i, 0] + new_r > 1:
                fits = False
            if pos[i, 1] - new_r < 0 or pos[i, 1] + new_r > 1:
                fits = False
            
            if fits:
                for j in range(N):
                    if i == j:
                        continue
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < new_r + radii[j]:
                        fits = False
                        break
            
            if fits:
                radii[i] = new_r
                improved = True
                # Push other circles away
                for j in range(N):
                    if i == j:
                        continue
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < 1e-10:
                        continue
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist:
                        push = (min_dist - dist) / dist
                        pos[j, 0] += dx * push * 0.5
                        pos[j, 1] += dy * push * 0.5
                        # Clamp to square
                        pos[j, 0] = np.clip(pos[j, 0], radii[j], 1 - radii[j])
                        pos[j, 1] = np.clip(pos[j, 1], radii[j], 1 - radii[j])
        
        if not improved:
            break
    
    pos_pp, radii_pp = _post_process(pos, radii)
    return pos_pp, radii_pp


def _polish(pos, radii, deadline):
    """High-precision local search with multiple restarts to squeeze out more sum of radii."""
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
    
    # Multiple L-BFGS-B passes with decreasing tolerances
    tolerances = [(1e-12, 1e-8), (1e-14, 1e-10), (1e-16, 1e-12)]
    for ftol, gtol in tolerances:
        if time.time() > deadline - 3:
            break
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 3000, 'ftol': ftol, 'gtol': gtol})
            
            if result.fun < _objective(x0) - 1e-15:
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
    
    # SLSQP polish with tighter bounds for high-precision refinement
    if time.time() < deadline - 3:
        try:
            tight_bounds = []
            for i in range(N):
                x_lo = max(0.0, best_pos[i, 0] - 0.05)
                x_hi = min(1.0, best_pos[i, 0] + 0.05)
                y_lo = max(0.0, best_pos[i, 1] - 0.05)
                y_hi = min(1.0, best_pos[i, 1] + 0.05)
                r_lo = max(1e-5, best_radii[i] * 0.8)
                r_hi = min(0.5, best_radii[i] * 1.2)
                tight_bounds.append((x_lo, x_hi))
                tight_bounds.append((y_lo, y_hi))
            for i in range(N):
                tight_bounds.append((r_lo, r_hi))
            
            result = minimize(_objective, x0, method='SLSQP', bounds=tight_bounds,
                              options={'maxiter': 2000, 'ftol': 1e-16, 'eps': 1e-12})
            
            if result.fun < _objective(x0) - 1e-15:
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
    
    # Final Powell pass for fine-tuning
    if time.time() < deadline - 2:
        try:
            result = minimize(_objective, x0, method='Powell', bounds=bounds,
                              options={'maxiter': 1000, 'ftol': 1e-14})
            
            if result.fun < _objective(x0) - 1e-15:
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
    
    # L-BFGS-B first (fast gradient-based)
    if time.time() < deadline - 8:
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.fun < _objective(x0) - 1e-15:
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
    
    # SLSQP intermediate pass for constraint-aware refinement
    if time.time() < deadline - 7:
        try:
            result = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                              options={'maxiter': 1000, 'ftol': 1e-14, 'eps': 1e-10})
            
            if result.fun < _objective(x0) - 1e-15:
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
            result = minimize(_objective, x0, method='Powell', bounds=bounds,
                              options={'maxiter': 800, 'ftol': 1e-13})
            
            if result.fun < _objective(x0) - 1e-15:
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
    
    # Grow and push to expand circles
    if time.time() < deadline - 5:
        try:
            pos_gp, radii_gp = _grow_and_push(best_pos, best_radii, deadline)
            score_gp = np.sum(radii_gp)
            if _is_valid(pos_gp, radii_gp) and score_gp > best_score + 1e-12:
                best_score = score_gp
                best_pos = pos_gp.copy()
                best_radii = radii_gp.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 4:
        best_pos, best_radii, best_score = _polish(best_pos, best_radii, deadline)
        
    return best_pos, best_radii, best_score


def _differential_evolution_run(deadline):
    """Run differential evolution for global search with time monitoring."""
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.25))
    
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
        
        if result.fun < 0:
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
    deadline = start_time + 140.0
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    init_configs = []
    
    pos, radii = _initial_hexagonal()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_grid()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_corner_heavy()
    init_configs.append((pos, radii))
    
    for seed in [42, 123, 456]:
        pos, radii = _initial_random(seed)
        init_configs.append((pos, radii))
    
    rng = np.random.RandomState(99)
    pos, radii = _initial_hexagonal()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    pos, radii = _initial_corner_heavy()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    config_order = [0, 2, 5, 6, 1, 3, 4]
    for idx in config_order:
        if time.time() > deadline - 20:
            break
        if idx >= len(init_configs):
            continue
        
        pos, radii = init_configs[idx]
        pos_opt, radii_opt, score = _optimize_from(pos, radii, deadline)
        
        if score > best_score + 1e-12:
            best_score = score
            best_pos = pos_opt.copy()
            best_radii = radii_opt.copy()
    
    if time.time() < deadline - 30:
        de_pos, de_radii, de_score = _differential_evolution_run(deadline)
        if de_pos is not None and de_score > best_score + 1e-12:
            best_score = de_score
            best_pos = de_pos.copy()
            best_radii = de_radii.copy()
    
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
    
    if time.time() < deadline - 2:
        best_pos, best_radii, best_score = _polish(best_pos, best_radii, deadline)
    
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
