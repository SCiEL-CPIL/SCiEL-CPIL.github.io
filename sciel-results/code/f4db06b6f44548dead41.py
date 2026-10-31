"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import cKDTree

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Strict validity check."""
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    if np.any(dist[mask] < min_dist[mask] - tol):
        return False
    return True

def _post_process(pos, radii, max_iter=1000):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(max_iter):
        valid = True
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.99999, 1e-6)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    if dist > 1e-10:
                        k = (dist / sum_r) * 0.99999
                    else:
                        k = 0.5
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-6)
                        radii[j] = max(radii[j] * k, 1e-6)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _max_radius_at(x, y, pos, radii):
    """Maximum radius for a circle at (x, y) given existing circles."""
    max_r = min(x, 1.0 - x, y, 1.0 - y)
    for i in range(len(radii)):
        dx = x - pos[i, 0]
        dy = y - pos[i, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        max_r = min(max_r, dist - radii[i])
    return max(0.0, max_r)

def _greedy_init(seed=42):
    """Greedy best-fit initialization with multiple attempts."""
    rng = np.random.RandomState(seed)
    best_pos = None
    best_radii = None
    best_score = 0
    
    for attempt in range(5):
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        
        # Place first circle
        x0 = rng.uniform(0.3, 0.7)
        y0 = rng.uniform(0.3, 0.7)
        pos[0] = [x0, y0]
        radii[0] = min(x0, 1-x0, y0, 1-y0) * 0.99
        
        # Place remaining circles greedily
        for i in range(1, N):
            best_r = 0
            best_x, best_y = 0.5, 0.5
            
            # Try many random positions
            for _ in range(3000):
                x = rng.uniform(0.01, 0.99)
                y = rng.uniform(0.01, 0.99)
                r = _max_radius_at(x, y, pos[:i], radii[:i])
                if r > best_r:
                    best_r = r
                    best_x, best_y = x, y
            
            pos[i] = [best_x, best_y]
            radii[i] = best_r * 0.999
        
        pos_pp, radii_pp = _post_process(pos, radii)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def _hex_init():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Try different hexagonal configurations
    best_score = 0
    best_pos = None
    best_radii = None
    
    for r in np.linspace(0.08, 0.15, 20):
        dx = 2 * r * np.sqrt(3) / 2
        dy = 3 * r
        
        k = 0
        p = np.zeros((N, 2))
        rad = np.zeros(N)
        
        for row in range(8):
            offset = dx / 2 if row % 2 == 1 else 0
            y = row * dy + r
            for col in range(6):
                if k >= N:
                    break
                x = col * dx + offset + r
                if 0 <= x - r and x + r <= 1.0 and 0 <= y - r and y + r <= 1.0:
                    p[k] = [x, y]
                    rad[k] = r
                    k += 1
            if k >= N:
                break
        
        if k < N:
            # Fill remaining with small circles
            rng = np.random.RandomState(123)
            for _ in range(1000):
                if k >= N:
                    break
                x = rng.uniform(0.05, 0.95)
                y = rng.uniform(0.05, 0.95)
                r_max = _max_radius_at(x, y, p[:k], rad[:k])
                if r_max > 0.01:
                    p[k] = [x, y]
                    rad[k] = r_max * 0.99
                    k += 1
        
        if k == N:
            pos_pp, radii_pp = _post_process(p, rad)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def _corner_init(seed=7):
    """Corner-biased initialization."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    corners = [(0.12, 0.12), (0.88, 0.12), (0.12, 0.88), (0.88, 0.88)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx, cy]
        radii[i] = 0.19
    
    edges = [
        (0.5, 0.06), (0.5, 0.94), (0.06, 0.5), (0.94, 0.5),
        (0.25, 0.06), (0.75, 0.06), (0.25, 0.94), (0.75, 0.94),
        (0.06, 0.25), (0.06, 0.75), (0.94, 0.25), (0.94, 0.75)
    ]
    
    for i in range(4, N):
        if i - 4 < len(edges):
            ex, ey = edges[i - 4]
        else:
            ex, ey = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        
        pos[i] = [ex, ey]
        radii[i] = _max_radius_at(ex, ey, pos[:i], radii[:i]) * 0.999
    
    pos_pp, radii_pp = _post_process(pos, radii)
    return pos_pp, radii_pp

def _objective(x, penalty_weight=1000.0):
    """
    Objective: minimize -sum(radii) + penalty * violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    # Boundary violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    viol = np.maximum(0.0, min_dist - dist)
    overlap_pen = np.sum(viol[mask] * viol[mask])
    
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _slsqp_optimize(x0, deadline):
    """SLSQP optimization with hard constraints."""
    pos0 = x0[:2*N].reshape(N, 2)
    radii0 = x0[2*N:]
    
    # Define constraints
    def boundary_constraints(x):
        pos = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        # x >= r, x <= 1-r, y >= r, y <= 1-r
        c1 = pos[:, 0] - radii
        c2 = (1.0 - pos[:, 0]) - radii
        c3 = pos[:, 1] - radii
        c4 = (1.0 - pos[:, 1]) - radii
        return np.concatenate([c1, c2, c3, c4])
    
    def overlap_constraints(x):
        pos = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        constraints = []
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                constraints.append(dist - (radii[i] + radii[j]))
        return np.array(constraints)
    
    constraints = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': overlap_constraints}
    ]
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.5))
    
    try:
        result = minimize(
            lambda x: -np.sum(x[2*N:]),  # Maximize sum of radii
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-12}
        )
        if result.success and result.fun < -np.sum(radii0) + 1e-6:
            return result.x
    except Exception:
        pass
    
    return x0

def _lbfgs_optimize(x0, deadline, penalty_weights=None):
    """L-BFGS-B optimization with penalty method."""
    if penalty_weights is None:
        penalty_weights = [100.0, 500.0, 2000.0, 10000.0]
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.5))
    
    current_x = x0.copy()
    
    for p_w in penalty_weights:
        if time.time() > deadline - 5:
            break
        try:
            result = minimize(
                _objective,
                current_x,
                method='L-BFGS-B',
                bounds=bounds,
                args=(p_w,),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < _objective(current_x, p_w) - 1e-10:
                current_x = result.x
        except Exception:
            pass
    
    return current_x

def _simulated_annealing(pos, radii, n_steps=5000, t0=0.03, t_end=0.0005, seed=0, deadline=None):
    """Simulated annealing with adaptive step size."""
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    current_energy = _objective(np.concatenate([pos.flatten(), radii]), 500.0)
    
    for step in range(n_steps):
        if deadline and time.time() > deadline - 5:
            break
        
        frac = step / max(n_steps - 1, 1)
        temp = t0 * (t_end / t0) ** frac
        
        # Adaptive step size
        step_size = 0.01 * (1.0 - 0.5 * frac)
        
        # Perturb: move one circle
        i = rng.randint(n)
        dx = rng.normal(0, step_size)
        dy = rng.normal(0, step_size)
        dr = rng.normal(0, step_size * 0.3)
        
        new_x = np.clip(pos[i, 0] + dx, 0.001, 0.999)
        new_y = np.clip(pos[i, 1] + dy, 0.001, 0.999)
        new_r = np.clip(radii[i] + dr, 0.001, 0.5)
        
        old_x, old_y, old_r = pos[i, 0], pos[i, 1], radii[i]
        pos[i, 0] = new_x
        pos[i, 1] = new_y
        radii[i] = new_r
        
        x_new = np.concatenate([pos.flatten(), radii])
        new_energy = _objective(x_new, 500.0)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-8)):
            current_energy = new_energy
        else:
            pos[i, 0] = old_x
            pos[i, 1] = old_y
            radii[i] = old_r
        
        # Periodically check and update best
        if step % 200 == 0:
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def _local_search(pos, radii, deadline, max_iter=2000):
    """Local search: try to grow each circle and move to fill gaps."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(max_iter):
        if time.time() > deadline - 5:
            break
        improved = False
        
        # Try to grow each circle
        for i in range(N):
            # Check if we can grow circle i
            can_grow = True
            new_r = radii[i]
            
            # Check boundaries
            max_r_boundary = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            new_r = min(new_r * 1.001, max_r_boundary)
            
            # Check overlaps
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_allowed = dist - radii[j]
                new_r = min(new_r, max_allowed)
            
            if new_r > radii[i] * 1.0001:
                radii[i] = new_r
                improved = True
        
        # Try to move circles to better positions
        if not improved:
            i = np.random.randint(N)
            old_x, old_y = pos[i]
            # Try small random moves
            for _ in range(10):
                dx = np.random.normal(0, 0.005)
                dy = np.random.normal(0, 0.005)
                new_x = np.clip(old_x + dx, 0.001, 0.999)
                new_y = np.clip(old_y + dy, 0.001, 0.999)
                
                # Check if this position allows larger radius
                max_r = min(new_x, 1.0 - new_x, new_y, 1.0 - new_y)
                for j in range(N):
                    if i == j:
                        continue
                    ddx = new_x - pos[j, 0]
                    ddy = new_y - pos[j, 1]
                    dist = np.sqrt(ddx * ddx + ddy * ddy)
                    max_r = min(max_r, dist - radii[j])
                
                if max_r > radii[i] * 1.001:
                    pos[i] = [new_x, new_y]
                    radii[i] = max_r * 0.999
                    improved = True
                    break
        
        if not improved:
            break
    
    pos_pp, radii_pp = _post_process(pos, radii)
    return pos_pp, radii_pp

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate multiple initializations
    inits = []
    
    pos_g, rad_g = _greedy_init(seed=42)
    if pos_g is not None:
        inits.append((pos_g, rad_g))
    
    pos_h, rad_h = _hex_init()
    if pos_h is not None:
        inits.append((pos_h, rad_h))
    
    pos_c, rad_c = _corner_init(seed=7)
    inits.append((pos_c, rad_c))
    
    # Add random inits
    for seed in [100, 200]:
        rng = np.random.RandomState(seed)
        pos_r = rng.uniform(0.1, 0.9, (N, 2))
        rad_r = rng.uniform(0.05, 0.12, N)
        pos_r, rad_r = _post_process(pos_r, rad_r)
        inits.append((pos_r, rad_r))
    
    best_pos = inits[0][0].copy()
    best_radii = inits[0][1].copy()
    best_score = np.sum(best_radii)
    
    # Optimize each initialization
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()
        
        # Stage 1: L-BFGS-B with penalty
        if time.time() < deadline - 20:
            current_x = _lbfgs_optimize(current_x, deadline)
        
        # Stage 2: SLSQP with hard constraints
        if time.time() < deadline - 15:
            current_x = _slsqp_optimize(current_x, deadline)
        
        # Stage 3: Simulated annealing
        if time.time() < deadline - 10:
            pos_sa, rad_sa = _simulated_annealing(
                current_x[:2*N].reshape(N, 2),
                np.maximum(current_x[2*N:], 1e-4),
                n_steps=3000,
                seed=idx * 100,
                deadline=deadline
            )
            current_x = np.concatenate([pos_sa.flatten(), rad_sa])
        
        # Stage 4: Local search
        if time.time() < deadline - 5:
            pos_ls, rad_ls = _local_search(
                current_x[:2*N].reshape(N, 2),
                np.maximum(current_x[2*N:], 1e-4),
                deadline
            )
            current_x = np.concatenate([pos_ls.flatten(), rad_ls])
        
        # Check result
        pos_opt = current_x[:2*N].reshape(N, 2)
        radii_opt = np.maximum(current_x[2*N:], 1e-4)
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
        
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    # Final polish: try to grow all circles
    if time.time() < deadline - 3:
        pos_final, radii_final = _local_search(best_pos, best_radii, deadline, max_iter=1000)
        if _is_valid(pos_final, radii_final):
            score = np.sum(radii_final)
            if score > best_score:
                best_score = score
                best_pos = pos_final.copy()
                best_radii = radii_final.copy()
    
    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        # Fallback: grid of small circles
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N - 1:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = _post_process(pos, radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
