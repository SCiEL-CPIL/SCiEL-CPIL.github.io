"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

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
    """
    Objective: minimize negative sum of radii + smooth penalty for violations.
    Uses a high penalty weight to strictly enforce constraints.
    """
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    radii = np.maximum(radii, 1e-4)
    
    penalty = 0.0
    # Boundary penalties (smooth)
    left_viol = np.maximum(0.0, radii - pos[:, 0])
    right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    bottom_viol = np.maximum(0.0, radii - pos[:, 1])
    top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    penalty += np.sum(left_viol**2) + np.sum(right_viol**2) + np.sum(bottom_viol**2) + np.sum(top_viol**2)
    
    # Pairwise overlap penalties (vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    # Only consider upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    overlap = np.maximum(0.0, min_dist_sq - dist_sq)
    # Use a smoother penalty: scale by min_dist to normalize
    penalty += np.sum(overlap[mask] ** 2 / (4.0 * np.maximum(min_dist_sq[mask], 1e-8)))
    
    return -np.sum(radii) + penalty_weight * penalty


def _initial_grid():
    """Create a 5x5 grid of circles with the 26th placed in a corner gap."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.098
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            pos[idx, 0] = (i + 0.5) / 5.0
            pos[idx, 1] = (j + 0.5) / 5.0
            radii[idx] = r
    
    pos[25, 0] = 0.90
    pos[25, 1] = 0.90
    radii[25] = 0.08
    
    return pos, radii


def _initial_hexagonal():
    """Create a hexagonal packing pattern with optimized spacing."""
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
    for cx in [0.05, 0.15, 0.35, 0.5, 0.65, 0.85, 0.95]:
        for cy in [0.05, 0.15, 0.35, 0.5, 0.65, 0.85, 0.95]:
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


def _initial_corner_edge():
    """Circles in corners and edges with interior grid - inspired by trust-constr approach."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    idx = 0
    # 4 corner circles
    corner_r = 0.18
    corners = [(corner_r, corner_r), (1-corner_r, corner_r), 
               (corner_r, 1-corner_r), (1-corner_r, 1-corner_r)]
    for x, y in corners:
        pos[idx, 0] = x
        pos[idx, 1] = y
        radii[idx] = corner_r
        idx += 1
    
    # 8 edge circles (2 per edge, avoiding corners)
    edge_r = 0.12
    # Top edge
    for x in [0.3, 0.7]:
        pos[idx, 0] = x
        pos[idx, 1] = 1 - edge_r
        radii[idx] = edge_r
        idx += 1
    # Bottom edge
    for x in [0.3, 0.7]:
        pos[idx, 0] = x
        pos[idx, 1] = edge_r
        radii[idx] = edge_r
        idx += 1
    # Left edge
    for y in [0.3, 0.7]:
        pos[idx, 0] = edge_r
        pos[idx, 1] = y
        radii[idx] = edge_r
        idx += 1
    # Right edge
    for y in [0.3, 0.7]:
        pos[idx, 0] = 1 - edge_r
        pos[idx, 1] = y
        radii[idx] = edge_r
        idx += 1
    
    # 14 interior circles in a grid-like pattern
    interior_r = 0.08
    cx, cy = 0.5, 0.5
    spacing = 0.22
    for i in range(-1, 2):
        for j in range(-1, 2):
            if idx < N:
                pos[idx, 0] = cx + i*spacing
                pos[idx, 1] = cy + j*spacing
                radii[idx] = interior_r
                idx += 1
    
    # Add remaining circles if needed
    while idx < N:
        pos[idx, 0] = 0.5
        pos[idx, 1] = 0.5
        radii[idx] = 0.05
        idx += 1
    
    return pos, radii


def _initial_center_ring():
    """Center circle surrounded by a ring of circles."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    pos[0, 0] = 0.5
    pos[0, 1] = 0.5
    radii[0] = 0.12
    
    ring_r = 0.08
    center_r = 0.12
    ring_dist = center_r + ring_r
    
    for i in range(12):
        idx = 1 + i
        angle = 2 * np.pi * i / 12
        pos[idx, 0] = 0.5 + ring_dist * np.cos(angle)
        pos[idx, 1] = 0.5 + ring_dist * np.sin(angle)
        radii[idx] = ring_r
    
    corner_r = 0.06
    corners = [
        (0.08, 0.08), (0.92, 0.08), (0.08, 0.92), (0.92, 0.92),
        (0.5, 0.05), (0.5, 0.95), (0.05, 0.5), (0.95, 0.5),
        (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8),
        (0.5, 0.5)
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
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(300):
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


def _grow_and_push(pos, radii, deadline, n_iter=20000):
    """
    Grow-and-Push local search.
    Iteratively tries to grow each circle to its maximum possible radius.
    If blocked, it pushes the blocking objects to create space.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    rng = np.random.RandomState(12345)
    
    for iteration in range(n_iter):
        if time.time() > deadline - 5:
            break
        
        # Pick a random circle to grow
        i = rng.randint(N)
        
        # Calculate max radius for i given current positions of others
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        
        dx = pos[:, 0] - pos[i, 0]
        dy = pos[:, 1] - pos[i, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        limits = dist - radii
        limits[i] = np.inf
        
        min_pair_limit = np.min(limits)
        max_r = min(max_r, min_pair_limit)
        max_r = max(max_r, 1e-4)
        
        if max_r > radii[i] + 1e-7:
            radii[i] = max_r
        else:
            blocking_indices = np.where(limits <= radii[i] + 1e-5)[0]
            blocking_indices = blocking_indices[blocking_indices != i]
            
            if len(blocking_indices) > 0:
                for j in blocking_indices[:3]:
                    dx_ij = pos[j, 0] - pos[i, 0]
                    dy_ij = pos[j, 1] - pos[i, 1]
                    dist_ij = np.sqrt(dx_ij*dx_ij + dy_ij*dy_ij)
                    
                    if dist_ij < 1e-9:
                        dx_ij = rng.normal(0, 0.01)
                        dy_ij = rng.normal(0, 0.01)
                        dist_ij = np.sqrt(dx_ij*dx_ij + dy_ij*dy_ij)
                    
                    tightness = max(0, (radii[i] + radii[j]) - dist_ij)
                    push_mag = 0.003 + 0.005 * min(tightness / 0.01, 1.0) + rng.uniform(0, 0.005)
                    
                    nx = dx_ij / dist_ij
                    ny = dy_ij / dist_ij
                    
                    pos[j, 0] += nx * push_mag
                    pos[j, 1] += ny * push_mag
                    
                    pos[i, 0] -= nx * push_mag * 0.3
                    pos[i, 1] -= ny * push_mag * 0.3
                    
                    pos[j] = np.clip(pos[j], 0, 1)
                    pos[i] = np.clip(pos[i], 0, 1)
            else:
                wall_pushed = False
                if pos[i, 0] - radii[i] < 1e-6:
                    pos[i, 0] += 0.005
                    wall_pushed = True
                elif 1.0 - pos[i, 0] - radii[i] < 1e-6:
                    pos[i, 0] -= 0.005
                    wall_pushed = True
                elif pos[i, 1] - radii[i] < 1e-6:
                    pos[i, 1] += 0.005
                    wall_pushed = True
                elif 1.0 - pos[i, 1] - radii[i] < 1e-6:
                    pos[i, 1] -= 0.005
                    wall_pushed = True
                
                if not wall_pushed:
                    pos[i, 0] += rng.uniform(-0.005, 0.005)
                    pos[i, 1] += rng.uniform(-0.005, 0.005)
            
            pos = np.clip(pos, 0, 1)
        
        if _is_valid(pos, radii, tol=1e-8):
            current_score = np.sum(radii)
            if current_score > best_score:
                best_score = current_score
                best_pos = pos.copy()
                best_radii = radii.copy()
        else:
            pos_pp, radii_pp = _post_process(pos, radii)
            pos = pos_pp
            radii = radii_pp
            current_score = np.sum(radii)
            if current_score > best_score:
                best_score = current_score
                best_pos = pos.copy()
                best_radii = radii.copy()
                
    return best_pos, best_radii, best_score


def _trust_constr_solve(pos, radii, deadline):
    """
    Trust-constr NLP with explicit constraints - inspired by the higher-scoring solution.
    Uses NonlinearConstraint with analytic Jacobian for precise constraint handling.
    """
    x0 = np.concatenate([pos.flatten(), radii])
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.5))
    
    # Constraint function: g(x) >= 0
    def _constraint_func(x):
        x_c = x[:N]
        y_c = x[N:2*N]
        r = x[2*N:]
        
        constraints = []
        
        # Boundary constraints
        for i in range(N):
            constraints.append(x_c[i] - r[i])
            constraints.append(1 - x_c[i] - r[i])
            constraints.append(y_c[i] - r[i])
            constraints.append(1 - y_c[i] - r[i])
            constraints.append(r[i])
        
        # Pairwise distance constraints
        for i in range(N):
            for j in range(i+1, N):
                dx = x_c[i] - x_c[j]
                dy = y_c[i] - y_c[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                constraints.append(dist_sq - min_dist*min_dist)
        
        return np.array(constraints)
    
    # Jacobian
    def _jacobian_func(x):
        x_c = x[:N]
        y_c = x[N:2*N]
        r = x[2*N:]
        
        n_constraints = 5*N + N*(N-1)//2
        jac = np.zeros((n_constraints, 3*N))
        
        idx = 0
        for i in range(N):
            jac[idx, i] = 1.0
            jac[idx, 2*N+i] = -1.0
            idx += 1
            jac[idx, i] = -1.0
            jac[idx, 2*N+i] = -1.0
            idx += 1
            jac[idx, N+i] = 1.0
            jac[idx, 2*N+i] = -1.0
            idx += 1
            jac[idx, N+i] = -1.0
            jac[idx, 2*N+i] = -1.0
            idx += 1
            jac[idx, 2*N+i] = 1.0
            idx += 1
        
        for i in range(N):
            for j in range(i+1, N):
                dx = x_c[i] - x_c[j]
                dy = y_c[i] - y_c[j]
                min_dist = r[i] + r[j]
                
                jac[idx, i] = 2*dx
                jac[idx, j] = -2*dx
                jac[idx, N+i] = 2*dy
                jac[idx, N+j] = -2*dy
                jac[idx, 2*N+i] = -2*min_dist
                jac[idx, 2*N+j] = -2*min_dist
                idx += 1
        
        return jac
    
    # Objective and gradient
    def _obj(x):
        return -np.sum(x[2*N:])
    
    def _obj_grad(x):
        grad = np.zeros(3*N)
        grad[2*N:] = -1.0
        return grad
    
    constraint = NonlinearConstraint(
        _constraint_func,
        0, np.inf,
        jac=_jacobian_func
    )
    
    try:
        res = minimize(
            _obj,
            x0,
            method='trust-constr',
            jac=_obj_grad,
            bounds=bounds,
            constraints=[constraint],
            options={'maxiter': 800, 'gtol': 1e-8, 'xtol': 1e-10}
        )
        
        if res.fun < -np.sum(radii) - 1e-10:
            x_opt = res.x
            pos_opt = x_opt[:N]
            pos_opt_2d = np.column_stack([pos_opt, x_opt[N:2*N]])
            radii_opt = np.maximum(x_opt[2*N:], 1e-5)
            
            pos_pp, radii_pp = _post_process(pos_opt_2d, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    except Exception:
        pass
    
    return best_pos, best_radii, best_score


def _polish(pos, radii, deadline):
    """High-precision local search with multiple methods."""
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
    
    # Trust-constr with explicit constraints
    if time.time() < deadline - 10:
        tc_pos, tc_radii, tc_score = _trust_constr_solve(best_pos, best_radii, deadline)
        if tc_score > best_score + 1e-12:
            best_score = tc_score
            best_pos = tc_pos.copy()
            best_radii = tc_radii.copy()
            x0 = np.concatenate([best_pos.flatten(), best_radii])
    
    # L-BFGS-B passes with penalty
    tolerances = [(1e-12, 1e-8), (1e-14, 1e-10)]
    for ftol, gtol in tolerances:
        if time.time() > deadline - 5:
            break
        try:
            result = minimize(lambda x: _objective(x, 5000.0), x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 5000, 'ftol': ftol, 'gtol': gtol})
            
            if result.fun < _objective(x0, 5000.0) - 1e-15:
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
    
    # Final Powell pass
    if time.time() < deadline - 3:
        try:
            result = minimize(lambda x: _objective(x, 5000.0), x0, method='Powell', bounds=bounds,
                              options={'maxiter': 3000, 'ftol': 1e-15})
            
            if result.fun < _objective(x0, 5000.0) - 1e-15:
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
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # 1. Grow and Push
    if time.time() < deadline - 25:
        gp_pos, gp_radii, gp_score = _grow_and_push(best_pos, best_radii, deadline, n_iter=15000)
        if gp_score > best_score + 1e-12:
            best_score = gp_score
            best_pos = gp_pos.copy()
            best_radii = gp_radii.copy()
    
    # 2. Polish (includes trust-constr)
    if time.time() < deadline - 8:
        pol_pos, pol_radii, pol_score = _polish(best_pos, best_radii, deadline)
        if pol_score > best_score + 1e-12:
            best_score = pol_score
            best_pos = pol_pos.copy()
            best_radii = pol_radii.copy()
    
    # 3. Second Grow and Push pass
    if time.time() < deadline - 8:
        gp_pos, gp_radii, gp_score = _grow_and_push(best_pos, best_radii, deadline, n_iter=8000)
        if gp_score > best_score + 1e-12:
            best_score = gp_score
            best_pos = gp_pos.copy()
            best_radii = gp_radii.copy()
    
    # 4. Final Polish
    if time.time() < deadline - 3:
        pol_pos, pol_radii, pol_score = _polish(best_pos, best_radii, deadline)
        if pol_score > best_score + 1e-12:
            best_score = pol_score
            best_pos = pol_pos.copy()
            best_radii = pol_radii.copy()
            
    return best_pos, best_radii, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # Build initial configurations
    init_configs = []
    
    pos, radii = _initial_grid()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_hexagonal()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_corner_edge()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_center_ring()
    init_configs.append((pos, radii))
    
    for seed in [42, 123]:
        pos, radii = _initial_random(seed)
        init_configs.append((pos, radii))
    
    # Perturbed hexagonal
    rng = np.random.RandomState(99)
    pos, radii = _initial_hexagonal()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # Perturbed corner-edge
    pos, radii = _initial_corner_edge()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # Run optimization from each initial config with time-aware allocation
    # Prioritize hexagonal, corner-edge, and grid
    config_order = [1, 2, 0, 6, 5, 3, 4]
    
    for idx in config_order:
        if time.time() > deadline - 18:
            break
        if idx >= len(init_configs):
            continue
        
        pos, radii = init_configs[idx]
        pos_opt, radii_opt, score = _optimize_from(pos, radii, deadline)
        
        if score > best_score + 1e-12:
            best_score = score
            best_pos = pos_opt.copy()
            best_radii = radii_opt.copy()
    
    # Fallback if nothing valid found
    if best_pos is None or best_score < 0:
        pos, radii = _initial_grid()
        best_pos = pos
        best_radii = radii
        best_score = np.sum(radii)
    
    # Final high-precision polish on the best found
    if time.time() < deadline - 3:
        best_pos, best_radii, best_score = _polish(best_pos, best_radii, deadline)
    
    # Final post-processing to ensure validity
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
