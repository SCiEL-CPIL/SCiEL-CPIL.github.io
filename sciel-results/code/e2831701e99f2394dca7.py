"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

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


def _post_process(pos, radii):
    """Iteratively fix validity by repositioning first, then shrinking."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(300):
        valid = True
        
        # First try to reposition circles to fix boundary violations
        for i in range(N):
            min_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > min_r:
                # Try moving toward center
                target_x = 0.5
                target_y = 0.5
                # Move proportionally toward center
                pos[i, 0] = pos[i, 0] + (target_x - pos[i, 0]) * 0.3
                pos[i, 1] = pos[i, 1] + (target_y - pos[i, 1]) * 0.3
                valid = False
        
        # Check boundaries and shrink if needed
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r:
                radii[i] = max(0.001, max_r * 0.99)
                valid = False
        
        # Check overlaps: try to separate first, then shrink
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-14 else 0.0
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    # Try to push apart
                    if dist > 1e-6:
                        push = (min_dist - dist) * 0.5
                        nx = dx / dist
                        ny = dy / dist
                        pos[i, 0] += nx * push
                        pos[i, 1] += ny * push
                        pos[j, 0] -= nx * push
                        pos[j, 1] -= ny * push
                    else:
                        # Overlapping at same point, move randomly
                        angle = np.random.uniform(0, 2 * np.pi)
                        pos[i, 0] += 0.01 * np.cos(angle)
                        pos[i, 1] += 0.01 * np.sin(angle)
                        pos[j, 0] -= 0.01 * np.cos(angle)
                        pos[j, 1] -= 0.01 * np.sin(angle)
                    valid = False
        
        # Clamp positions to valid range
        for i in range(N):
            pos[i, 0] = np.clip(pos[i, 0], 0.001, 0.999)
            pos[i, 1] = np.clip(pos[i, 1], 0.001, 0.999)
        
        if valid:
            break
    
    # Final pass: ensure strict validity with small safety margin
    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = min(radii[i], max(0.001, max_r * 0.999))
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                # Shrink the smaller circle more
                scale = (dist / (min_dist + 1e-12)) * 0.995
                if radii[i] >= radii[j]:
                    radii[j] = max(0.001, radii[j] * scale)
                else:
                    radii[i] = max(0.001, radii[i] * scale)
    
    return pos, radii


def _make_grid(seed=42):
    """5x5 grid layout."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    side = 5
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if idx < N:
                pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                radii[idx] = 0.5 / side - 0.005
    if N > 25:
        pos[25] = [0.5, 0.05]
        radii[25] = 0.04
    return pos, radii


def _make_hexagonal(seed=42):
    """Hexagonal packing pattern."""
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


def _make_center_ring(seed=42):
    """One center circle with rings around it."""
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


def _make_random(seed=42):
    """Random initial positions."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(N, 2))
    radii = rng.uniform(0.03, 0.1, size=N)
    return pos, radii


def _slsqp_polish(pos, radii, deadline=None):
    """SLSQP polish with explicit constraints."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.35))
    
    # Build constraint list
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i],
                           'jac': lambda x, i=i: _slsqp_bjac(x, i, 0)})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N+i],
                           'jac': lambda x, i=i: _slsqp_bjac(x, i, 1)})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i],
                           'jac': lambda x, i=i: _slsqp_bjac(x, i, 2)})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N+i],
                           'jac': lambda x, i=i: _slsqp_bjac(x, i, 3)})
    
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: _slsqp_overlap_val(x, i, j),
                               'jac': lambda x, i=i, j=j: _slsqp_overlap_jac(x, i, j)})
    
    try:
        result = minimize(lambda x: -np.sum(x[2*N:]), x0, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 200, 'ftol': 1e-12, 'disp': False})
        if result.success or result.fun < -np.sum(radii):
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = x_opt[2*N:]
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass
    
    return pos, radii


def _slsqp_bjac(x, i, which):
    """Jacobian for boundary constraint."""
    n = N
    jac = np.zeros(3*n)
    if which == 0:
        jac[2*i] = 1.0
        jac[2*n+i] = -1.0
    elif which == 1:
        jac[2*i] = -1.0
        jac[2*n+i] = -1.0
    elif which == 2:
        jac[2*i+1] = 1.0
        jac[2*n+i] = -1.0
    elif which == 3:
        jac[2*i+1] = -1.0
        jac[2*n+i] = -1.0
    return jac


def _slsqp_overlap_val(x, i, j):
    """Non-overlap constraint value."""
    n = N
    dx = x[2*i] - x[2*j]
    dy = x[2*i+1] - x[2*j+1]
    dist = np.sqrt(dx*dx + dy*dy)
    return dist - x[2*n+i] - x[2*n+j]


def _slsqp_overlap_jac(x, i, j):
    """Jacobian for non-overlap constraint."""
    n = N
    jac = np.zeros(3*n)
    dx = x[2*i] - x[2*j]
    dy = x[2*i+1] - x[2*j+1]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(dist_sq) if dist_sq > 1e-14 else 1e-7
    inv_dist = 1.0 / dist
    jac[2*i] = dx * inv_dist
    jac[2*i+1] = dy * inv_dist
    jac[2*j] = -dx * inv_dist
    jac[2*j+1] = -dy * inv_dist
    jac[2*n+i] = -1.0
    jac[2*n+j] = -1.0
    return jac


def _force_directed_anneal(pos, radii, n_iter=500, deadline=None):
    """Force-directed annealing to explore topology and improve packing."""
    pos = pos.copy()
    radii = radii.copy()
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    vel = np.zeros_like(pos)
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        temp = 0.02 * np.exp(-0.002 * it)
        
        forces = np.zeros_like(pos)
        
        for i in range(N):
            if pos[i, 0] < radii[i] + 0.01:
                forces[i, 0] += 5.0 * (radii[i] + 0.01 - pos[i, 0])
            if pos[i, 0] > 1.0 - radii[i] - 0.01:
                forces[i, 0] -= 5.0 * (pos[i, 0] - (1.0 - radii[i] - 0.01))
            if pos[i, 1] < radii[i] + 0.01:
                forces[i, 1] += 5.0 * (radii[i] + 0.01 - pos[i, 1])
            if pos[i, 1] > 1.0 - radii[i] - 0.01:
                forces[i, 1] -= 5.0 * (pos[i, 1] - (1.0 - radii[i] - 0.01))
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-6:
                    overlap = min_dist - dist
                    force_mag = 10.0 * overlap
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                elif dist < min_dist * 1.5 and dist > 1e-6:
                    force_mag = 0.5 * (min_dist * 1.5 - dist)
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        vel = 0.9 * vel + 0.1 * forces
        pos += vel * 0.01 + np.random.normal(0, temp, size=pos.shape)
        
        for i in range(N):
            pos[i, 0] = np.clip(pos[i, 0], 0.01, 0.99)
            pos[i, 1] = np.clip(pos[i, 1], 0.01, 0.99)
        
        for i in range(N):
            can_grow = True
            max_grow = 0.001
            max_grow = min(max_grow, pos[i, 0] - radii[i], 1.0 - pos[i, 0] - radii[i],
                          pos[i, 1] - radii[i], 1.0 - pos[i, 1] - radii[i])
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                available = dist - radii[j]
                if available < max_grow:
                    max_grow = available
            
            if max_grow > 0.0001:
                radii[i] += max_grow * 0.5
        
        pos_pp, radii_pp = _post_process(pos, radii)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    return best_pos, best_radii


def _objective_smooth(x, beta=50.0, smooth_eps=0.005):
    """Smooth objective for L-BFGS-B."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    penalty = 0.0
    for i in range(n):
        for coord in range(2):
            d = pos[i, coord] - radii[i]
            if d < 0:
                penalty += d * d
            d = (1.0 - pos[i, coord]) - radii[i]
            if d < 0:
                penalty += d * d
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            min_dist = radii[i] + radii[j]
            if dist_sq < min_dist * min_dist:
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 0.0
                violation = min_dist - dist
                if violation > 0:
                    penalty += violation * violation
    
    return -np.sum(radii) + beta * penalty


def _polish(pos, radii, deadline=None):
    """High-precision polish using multiple methods."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.35))
    
    best_pos, best_radii = pos.copy(), radii.copy()
    best_score = np.sum(radii)
    
    # L-BFGS-B with smooth objective
    try:
        result = minimize(_objective_smooth, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
        if result.fun < _objective_smooth(x0):
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = x_opt[2*N:]
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp) and np.sum(radii_pp) > best_score:
                best_score = np.sum(radii_pp)
                best_pos, best_radii = pos_pp, radii_pp
    except Exception:
        pass
    
    # SLSQP with explicit constraints
    if deadline is None or time.time() < deadline - 5:
        try:
            pos_s, rad_s = _slsqp_polish(best_pos, best_radii, deadline=deadline)
            if _is_valid(pos_s, rad_s) and np.sum(rad_s) > best_score:
                best_score = np.sum(rad_s)
                best_pos, best_radii = pos_s, rad_s
        except Exception:
            pass
    
    # Growth pass
    if deadline is None or time.time() < deadline - 2:
        try:
            pos_g, rad_g = _growth_pass(best_pos, best_radii, deadline=deadline)
            if _is_valid(pos_g, rad_g) and np.sum(rad_g) > best_score:
                best_score = np.sum(rad_g)
                best_pos, best_radii = pos_g, rad_g
        except Exception:
            pass
    
    return best_pos, best_radii


def _growth_pass(pos, radii, deadline=None):
    """Try to grow each circle individually to fill available space."""
    pos = pos.copy()
    radii = radii.copy()
    improved = True
    n_iter = 0
    
    while improved and n_iter < 50:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        n_iter += 1
        for i in range(N):
            max_r = min(pos[i,0], 1.0-pos[i,0], pos[i,1], 1.0-pos[i,1])
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                max_r = min(max_r, dist - radii[j])
            max_r = max(0.001, max_r * 0.999)
            if max_r > radii[i] + 1e-6:
                radii[i] = max_r
                improved = True
    
    pos_pp, radii_pp = _post_process(pos, radii)
    return pos_pp, radii_pp


def _gradient_polish(pos, radii, deadline=None):
    """High-precision gradient-based polish using L-BFGS-B with tight tolerances."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.35))
    
    best_pos, best_radii = pos.copy(), radii.copy()
    best_score = np.sum(radii)
    
    for attempt in range(3):
        if deadline is not None and time.time() > deadline:
            break
        
        try:
            result = minimize(_objective_smooth, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100})
            if result.fun < _objective_smooth(x0) - 1e-12:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = x_opt[2*N:]
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score + 1e-10:
                        best_score = score
                        best_pos, best_radii = pos_pp, radii_pp
                        x0 = np.concatenate([pos_pp.flatten(), radii_pp])
                    else:
                        break
        except Exception:
            break
    
    return best_pos, best_radii


def _contact_graph_slsqp(pos, radii, deadline=None):
    """SLSQP optimization using only contact graph constraints for efficiency."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    # Find contact pairs (distances close to sum of radii)
    contact_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = radii[i] + radii[j]
            # Include pairs that are in contact or very close
            if dist < min_dist * 1.1:
                contact_pairs.append((i, j))
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.35))
    
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i],
                           'jac': lambda x, i=i: _slsqp_bjac(x, i, 0)})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N+i],
                           'jac': lambda x, i=i: _slsqp_bjac(x, i, 1)})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i],
                           'jac': lambda x, i=i: _slsqp_bjac(x, i, 2)})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N+i],
                           'jac': lambda x, i=i: _slsqp_bjac(x, i, 3)})
    
    # Only add constraints for contact pairs and nearby pairs
    for (i, j) in contact_pairs:
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: _slsqp_overlap_val(x, i, j),
                           'jac': lambda x, i=i, j=j: _slsqp_overlap_jac(x, i, j)})
    
    # Also add constraints for all pairs to ensure global validity
    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) not in contact_pairs:
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist * 1.5:
                    constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: _slsqp_overlap_val(x, i, j),
                                       'jac': lambda x, i=i, j=j: _slsqp_overlap_jac(x, i, j)})
    
    try:
        result = minimize(lambda x: -np.sum(x[2*N:]), x0, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 300, 'ftol': 1e-12, 'disp': False})
        if result.success or result.fun < -np.sum(radii):
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = x_opt[2*N:]
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass
    
    return pos, radii


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    strategies = [
        ('grid', _make_grid),
        ('hexagonal', _make_hexagonal),
        ('center_ring', _make_center_ring),
        ('random', _make_random)
    ]
    
    # Phase 1: Multi-start local optimization
    for strategy_name, strategy_func in strategies:
        if time.time() > deadline - 80:
            break
        
        for seed in range(5):
            if time.time() > deadline - 75:
                break
            
            pos, radii = strategy_func(seed)
            
            x0 = np.concatenate([pos.flatten(), radii])
            
            bounds = []
            for i in range(N):
                bounds.append((0.005, 0.995))
                bounds.append((0.005, 0.995))
            for i in range(N):
                bounds.append((0.005, 0.35))
            
            # Try L-BFGS-B with smooth objective
            try:
                result = minimize(_objective_smooth, x0, method='L-BFGS-B', bounds=bounds,
                                  options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10})
                if result.fun < _objective_smooth(x0):
                    x_opt = result.x
                    pos_opt = x_opt[:2*N].reshape(N, 2)
                    radii_opt = x_opt[2*N:]
                    
                    pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                    score = np.sum(radii_pp)
                    
                    if _is_valid(pos_pp, radii_pp) and score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
            except Exception:
                pass
            
            # Try Nelder-Mead
            if time.time() > deadline - 65:
                continue
            
            try:
                x_start = np.concatenate([pos.flatten(), radii])
                result_nm = minimize(_objective_smooth, x_start, method='Nelder-Mead',
                                     options={'maxiter': 1500, 'fatol': 1e-12, 'xatol': 1e-12})
                if result_nm.fun < _objective_smooth(x_start):
                    x_opt = result_nm.x
                    pos_opt = x_opt[:2*N].reshape(N, 2)
                    radii_opt = x_opt[2*N:]
                    
                    pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                    score = np.sum(radii_pp)
                    
                    if _is_valid(pos_pp, radii_pp) and score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
            except Exception:
                pass
    
    # Phase 2: Force-directed annealing from best result
    if best_pos is not None and time.time() < deadline - 35:
        try:
            pos_fd, radii_fd = _force_directed_anneal(best_pos, best_radii, n_iter=1000, deadline=deadline - 30)
            score_fd = np.sum(radii_fd)
            if _is_valid(pos_fd, radii_fd) and score_fd > best_score:
                best_score = score_fd
                best_pos = pos_fd.copy()
                best_radii = radii_fd.copy()
        except Exception:
            pass
    
    # Phase 3: Contact graph SLSQP polish
    if best_pos is not None and time.time() < deadline - 25:
        try:
            pos_cg, rad_cg = _contact_graph_slsqp(best_pos, best_radii, deadline=deadline - 20)
            score_cg = np.sum(rad_cg)
            if _is_valid(pos_cg, rad_cg) and score_cg > best_score:
                best_score = score_cg
                best_pos = pos_cg.copy()
                best_radii = rad_cg.copy()
        except Exception:
            pass
    
    # Phase 4: High-precision polish with constrained methods
    if best_pos is not None and time.time() < deadline - 15:
        try:
            pos_pol, radii_pol = _polish(best_pos, best_radii, deadline=deadline - 8)
            score_pol = np.sum(radii_pol)
            if _is_valid(pos_pol, radii_pol) and score_pol > best_score:
                best_score = score_pol
                best_pos = pos_pol.copy()
                best_radii = radii_pol.copy()
        except Exception:
            pass
    
    # Phase 5: Gradient polish for final fine-tuning
    if best_pos is not None and time.time() < deadline - 8:
        try:
            pos_gp, rad_gp = _gradient_polish(best_pos, best_radii, deadline=deadline - 5)
            score_gp = np.sum(rad_gp)
            if _is_valid(pos_gp, rad_gp) and score_gp > best_score:
                best_score = score_gp
                best_pos = pos_gp.copy()
                best_radii = rad_gp.copy()
        except Exception:
            pass
    
    # Phase 6: Additional SLSQP + growth from best
    if best_pos is not None and time.time() < deadline - 5:
        try:
            pos_s2, rad_s2 = _slsqp_polish(best_pos, best_radii, deadline=deadline - 3)
            if _is_valid(pos_s2, rad_s2):
                pos_g2, rad_g2 = _growth_pass(pos_s2, rad_s2, deadline=deadline - 2)
                score_g2 = np.sum(rad_g2)
                if _is_valid(pos_g2, rad_g2) and score_g2 > best_score:
                    best_score = score_g2
                    best_pos = pos_g2.copy()
                    best_radii = rad_g2.copy()
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
