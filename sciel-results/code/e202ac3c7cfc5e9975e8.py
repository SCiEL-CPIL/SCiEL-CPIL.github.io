"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time

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


def _penalty(pos, radii):
    """Total constraint violation (sum of squared violations)."""
    n = len(radii)
    pen = 0.0
    for i in range(n):
        if pos[i, 0] < radii[i]:
            pen += (radii[i] - pos[i, 0])**2
        if pos[i, 0] > 1.0 - radii[i]:
            pen += (pos[i, 0] - (1.0 - radii[i]))**2
        if pos[i, 1] < radii[i]:
            pen += (radii[i] - pos[i, 1])**2
        if pos[i, 1] > 1.0 - radii[i]:
            pen += (pos[i, 1] - (1.0 - radii[i]))**2
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_d = radii[i] + radii[j]
            if dist < min_d:
                pen += (min_d - dist)**2
    return pen


def _smooth_penalty(pos, radii, beta=80.0):
    """Smooth log-sum-exp penalty for better gradient flow in optimization."""
    n = len(radii)
    violations = np.zeros(4 * n + n * (n - 1) // 2)
    idx = 0
    for i in range(n):
        v = radii[i] - pos[i, 0]
        violations[idx] = max(0.0, v)
        idx += 1
        v = pos[i, 0] - (1.0 - radii[i])
        violations[idx] = max(0.0, v)
        idx += 1
        v = radii[i] - pos[i, 1]
        violations[idx] = max(0.0, v)
        idx += 1
        v = pos[i, 1] - (1.0 - radii[i])
        violations[idx] = max(0.0, v)
        idx += 1
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_d = radii[i] + radii[j]
            violations[idx] = max(0.0, min_d - dist)
            idx += 1
    
    # Log-sum-exp for smooth max-like penalty
    max_v = np.max(violations)
    if max_v < 1e-15:
        return 0.0
    log_sum = max_v + np.log(np.sum(np.exp((violations - max_v) * beta)) / len(violations))
    return log_sum / beta


def _score(pos, radii):
    """Sum of radii (the objective to maximize)."""
    return float(np.sum(radii))


def _fix_boundaries(pos, radii):
    """Shrink radii to fit within boundaries."""
    radii = radii.copy()
    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        if radii[i] > max_r:
            radii[i] = max(0.001, max_r * 0.999)
    return radii


def _resolve_overlaps(pos, radii, max_iter=200):
    """Iteratively shrink radii to resolve overlaps."""
    radii = radii.copy()
    for _ in range(max_iter):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_d = radii[i] + radii[j]
                if dist < min_d:
                    scale = (dist / min_d) * 0.999 if min_d > 0 else 0.999
                    radii[i] *= scale
                    radii[j] *= scale
                    changed = True
        radii = _fix_boundaries(pos, radii)
        if not changed:
            break
    return radii


def _growth_pass(pos, radii, max_iter=500, step=0.002):
    """Try to grow each radius greedily while maintaining validity."""
    radii = radii.copy()
    for it in range(max_iter):
        improved = False
        for i in range(N):
            # Try growing circle i
            new_r = radii[i] + step
            # Check boundary
            if new_r > min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1]):
                continue
            # Check overlap with all others
            valid = True
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < new_r + radii[j] - 1e-10:
                    valid = False
                    break
            if valid:
                radii[i] = new_r
                improved = True
        if not improved:
            # Try smaller steps
            step *= 0.5
            if step < 1e-6:
                break
    return radii


def _make_grid(seed=42):
    """5x5 grid with one extra circle."""
    rng = np.random.RandomState(seed)
    side = 5
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if idx < N:
                pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                radii[idx] = 0.5 / side - 0.003
    # 26th circle in a gap
    pos[25] = [0.5, 0.02]
    radii[25] = 0.015
    return pos, radii


def _make_hexagonal(seed=42):
    """Hexagonal packing pattern."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r = 0.095
    idx = 0
    for row in range(6):
        y = 0.1 + row * r * np.sqrt(3)
        if y > 0.9:
            break
        offset = r if row % 2 == 0 else 2 * r
        for col in range(5):
            x = offset + col * 2 * r
            if x > 0.9:
                break
            if idx < N:
                pos[idx] = [x, y]
                radii[idx] = r
                idx += 1
    # Fill remaining
    while idx < N:
        pos[idx] = [0.5, 0.5]
        radii[idx] = 0.05
        idx += 1
    return pos, radii


def _make_random(seed=42):
    """Random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(N, 2))
    radii = rng.uniform(0.03, 0.08, size=N)
    return pos, radii


def _make_mixed(seed=42):
    """Mixed: some grid, some random."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    # 16 grid circles
    side = 4
    idx = 0
    for i in range(4):
        for j in range(4):
            pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
            radii[idx] = 0.5 / side - 0.005
            idx += 1
    # 10 random circles
    pos[idx:idx+10] = rng.uniform(0.05, 0.95, size=(10, 2))
    radii[idx:idx+10] = rng.uniform(0.03, 0.07, size=10)
    return pos, radii


def _make_center_ring(seed=42):
    """Center circle with ring of circles around it."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Center circle
    pos[0] = [0.5, 0.5]
    radii[0] = 0.15
    
    # Ring of 8 circles around center
    ring_r = 0.09
    ring_dist = radii[0] + ring_r + 0.005
    for k in range(8):
        angle = 2 * np.pi * k / 8
        pos[k+1] = [0.5 + ring_dist * np.cos(angle), 0.5 + ring_dist * np.sin(angle)]
        radii[k+1] = ring_r
    
    # Outer ring of 9 circles
    outer_r = 0.07
    outer_dist = 0.5
    for k in range(9):
        angle = 2 * np.pi * k / 9 + np.pi / 9
        pos[k+9] = [0.5 + outer_dist * np.cos(angle), 0.5 + outer_dist * np.sin(angle)]
        radii[k+9] = outer_r
    
    # Fill remaining 7 circles in gaps
    gap_positions = []
    for k in range(7):
        angle = 2 * np.pi * k / 7 + np.pi / 7
        r = 0.38
        x = 0.5 + r * np.cos(angle)
        y = 0.5 + r * np.sin(angle)
        if 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95:
            gap_positions.append((x, y, 0.055))
    
    for k in range(min(7, len(gap_positions))):
        pos[k+18] = [gap_positions[k][0], gap_positions[k][1]]
        radii[k+18] = gap_positions[k][2]
    
    # Any remaining
    for k in range(18 + min(7, len(gap_positions)), N):
        pos[k] = rng.uniform(0.1, 0.9)
        radii[k] = 0.04
    
    return pos, radii


def _force_directed_anneal(pos, radii, time_budget=30.0, seed=42):
    """Simulated annealing with force-directed moves."""
    rng = np.random.RandomState(seed)
    pos = pos.copy()
    radii = radii.copy()
    radii = _fix_boundaries(pos, radii)
    radii = _resolve_overlaps(pos, radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = _score(pos, radii)
    
    T = 0.01
    T_min = 1e-6
    cooling = 0.995
    max_iter = 50000
    
    start = time.time()
    
    for it in range(max_iter):
        if time.time() - start > time_budget:
            break
        
        # Random move: perturb positions and/or radii
        move_type = rng.random()
        if move_type < 0.5:
            # Move one circle
            i = rng.randint(N)
            dx = rng.normal(0, 0.02)
            dy = rng.normal(0, 0.02)
            pos[i, 0] = np.clip(pos[i, 0] + dx, 0.01, 0.99)
            pos[i, 1] = np.clip(pos[i, 1] + dy, 0.01, 0.99)
        elif move_type < 0.8:
            # Change radius of one circle
            i = rng.randint(N)
            dr = rng.normal(0, 0.005)
            radii[i] = np.clip(radii[i] + dr, 0.005, 0.3)
        else:
            # Move two circles
            i, j = rng.choice(N, size=2, replace=False)
            pos[i, 0] = np.clip(pos[i, 0] + rng.normal(0, 0.02), 0.01, 0.99)
            pos[i, 1] = np.clip(pos[i, 1] + rng.normal(0, 0.02), 0.01, 0.99)
            pos[j, 0] = np.clip(pos[j, 0] + rng.normal(0, 0.02), 0.01, 0.99)
            pos[j, 1] = np.clip(pos[j, 1] + rng.normal(0, 0.02), 0.01, 0.99)
        
        # Fix boundaries
        radii = _fix_boundaries(pos, radii)
        
        # Compute new score with penalty
        pen = _penalty(pos, radii)
        new_score = _score(pos, radii) - 100.0 * pen
        old_score = _score(best_pos, best_radii) - 100.0 * _penalty(best_pos, best_radii)
        
        delta = new_score - old_score
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            if new_score > best_score - 100.0 * _penalty(pos, radii):
                # Accept
                pos = pos.copy()
                radii = radii.copy()
                if pen < 1e-12:
                    s = _score(pos, radii)
                    if s > best_score:
                        best_score = s
                        best_pos = pos.copy()
                        best_radii = radii.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_pos, best_radii, best_score


def _gradient_polish(pos, radii, time_budget=15.0):
    """Gradient-based polishing with smooth penalty to fine-tune positions and radii."""
    from scipy.optimize import minimize
    
    pos = pos.copy()
    radii = radii.copy()
    radii = _fix_boundaries(pos, radii)
    radii = _resolve_overlaps(pos, radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = _score(pos, radii)
    
    def objective(x):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        # Use smooth penalty for better gradients
        pen = _smooth_penalty(p, r, beta=150.0)
        return -np.sum(r) + 30.0 * pen
    
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.3))
    
    start = time.time()
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
        if time.time() - start < time_budget:
            x_opt = result.x
            p_opt = x_opt[:2*N].reshape(N, 2)
            r_opt = x_opt[2*N:]
            r_opt = _fix_boundaries(p_opt, r_opt)
            r_opt = _resolve_overlaps(p_opt, r_opt)
            if _is_valid(p_opt, r_opt):
                s = _score(p_opt, r_opt)
                if s > best_score:
                    best_score = s
                    best_pos = p_opt.copy()
                    best_radii = r_opt.copy()
    except Exception:
        pass
    
    return best_pos, best_radii, best_score


def _slsqp_polish(pos, radii, time_budget=10.0):
    """SLSQP polishing with hard constraints."""
    from scipy.optimize import minimize
    
    pos = pos.copy()
    radii = radii.copy()
    radii = _fix_boundaries(pos, radii)
    radii = _resolve_overlaps(pos, radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = _score(pos, radii)
    
    def objective(x):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        return -np.sum(r)
    
    def objective_grad(x):
        grad = np.zeros_like(x)
        # Gradient of -sum(r) w.r.t. r is -1
        grad[2*N:] = -1.0
        return grad
    
    # Hard constraints
    constraints = []
    # Boundary constraints: r_i <= p_i, r_i <= 1-p_i, etc.
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})  # x >= r
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2*i]) - x[2*N+i]})  # 1-x >= r
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})  # y >= r
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2*i+1]) - x[2*N+i]})  # 1-y >= r
    
    # Overlap constraints: dist(i,j) >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            def overlap_con(x, i=i, j=j):
                dx = x[2*i] - x[2*j]
                dy = x[2*i+1] - x[2*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                return dist - x[2*N+i] - x[2*N+j]
            constraints.append({'type': 'ineq', 'fun': overlap_con})
    
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.3))
    
    start = time.time()
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          jac=objective_grad,
                          options={'maxiter': 600, 'ftol': 1e-14, 'maxls': 50})
        if time.time() - start < time_budget:
            x_opt = result.x
            p_opt = x_opt[:2*N].reshape(N, 2)
            r_opt = x_opt[2*N:]
            r_opt = _fix_boundaries(p_opt, r_opt)
            r_opt = _resolve_overlaps(p_opt, r_opt)
            if _is_valid(p_opt, r_opt):
                s = _score(p_opt, r_opt)
                if s > best_score:
                    best_score = s
                    best_pos = p_opt.copy()
                    best_radii = r_opt.copy()
    except Exception:
        pass
    
    return best_pos, best_radii, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    strategies = [
        ('grid', 42), ('grid', 123), ('grid', 7),
        ('hexagonal', 42), ('hexagonal', 99),
        ('random', 42), ('random', 123), ('random', 7), ('random', 55),
        ('mixed', 42), ('mixed', 123), ('mixed', 7),
        ('center_ring', 42), ('center_ring', 123),
    ]
    
    for strat_name, seed in strategies:
        if time.time() > deadline - 30:
            break
        
        if strat_name == 'grid':
            pos, radii = _make_grid(seed)
        elif strat_name == 'hexagonal':
            pos, radii = _make_hexagonal(seed)
        elif strat_name == 'random':
            pos, radii = _make_random(seed)
        elif strat_name == 'mixed':
            pos, radii = _make_mixed(seed)
        elif strat_name == 'center_ring':
            pos, radii = _make_center_ring(seed)
        else:
            continue
        
        # Force-directed annealing
        remaining = deadline - time.time()
        if remaining < 12:
            break
        anneal_budget = min(18.0, remaining * 0.45)
        pos_a, radii_a, score_a = _force_directed_anneal(pos, radii, time_budget=anneal_budget, seed=seed)
        
        if score_a > best_score:
            best_score = score_a
            best_pos = pos_a.copy()
            best_radii = radii_a.copy()
        
        # Gradient polish on annealed result
        remaining = deadline - time.time()
        if remaining < 18:
            continue
        polish_budget = min(12.0, remaining * 0.35)
        pos_p, radii_p, score_p = _gradient_polish(pos_a, radii_a, time_budget=polish_budget)
        
        if score_p > best_score:
            best_score = score_p
            best_pos = pos_p.copy()
            best_radii = radii_p.copy()
        
        # Growth pass to expand radii
        remaining = deadline - time.time()
        if remaining < 15:
            continue
        radii_grown = _growth_pass(best_pos, best_radii, max_iter=300, step=0.002)
        if _is_valid(best_pos, radii_grown):
            score_grown = _score(best_pos, radii_grown)
            if score_grown > best_score:
                best_score = score_grown
                best_radii = radii_grown.copy()
        
        # SLSQP polish
        remaining = deadline - time.time()
        if remaining < 12:
            continue
        slsqp_budget = min(10.0, remaining * 0.3)
        pos_s, radii_s, score_s = _slsqp_polish(best_pos, best_radii, time_budget=slsqp_budget)
        
        if score_s > best_score:
            best_score = score_s
            best_pos = pos_s.copy()
            best_radii = radii_s.copy()
        
        # Second gradient polish after SLSQP
        remaining = deadline - time.time()
        if remaining < 10:
            continue
        gp_budget = min(8.0, remaining * 0.3)
        pos_gp, radii_gp, score_gp = _gradient_polish(best_pos, best_radii, time_budget=gp_budget)
        if score_gp > best_score:
            best_score = score_gp
            best_pos = pos_gp.copy()
            best_radii = radii_gp.copy()
        
        # Final growth pass after second polish
        remaining = deadline - time.time()
        if remaining < 5:
            continue
        radii_grown2 = _growth_pass(best_pos, best_radii, max_iter=200, step=0.001)
        if _is_valid(best_pos, radii_grown2):
            score_grown2 = _score(best_pos, radii_grown2)
            if score_grown2 > best_score:
                best_score = score_grown2
                best_radii = radii_grown2.copy()
    
    # Final polish on best result if time allows
    if best_pos is not None:
        remaining = deadline - time.time()
        if remaining > 18:
            pos_f, radii_f, score_f = _gradient_polish(best_pos, best_radii, time_budget=min(12.0, remaining - 8))
            if score_f > best_score:
                best_score = score_f
                best_pos = pos_f.copy()
                best_radii = radii_f.copy()
        
        # Final SLSQP polish
        remaining = deadline - time.time()
        if remaining > 12:
            pos_fs, radii_fs, score_fs = _slsqp_polish(best_pos, best_radii, time_budget=min(8.0, remaining - 5))
            if score_fs > best_score:
                best_score = score_fs
                best_pos = pos_fs.copy()
                best_radii = radii_fs.copy()
        
        # Second final gradient polish
        remaining = deadline - time.time()
        if remaining > 8:
            pos_ff, radii_ff, score_ff = _gradient_polish(best_pos, best_radii, time_budget=min(6.0, remaining - 4))
            if score_ff > best_score:
                best_score = score_ff
                best_pos = pos_ff.copy()
                best_radii = radii_ff.copy()
        
        # Final growth pass
        remaining = deadline - time.time()
        if remaining > 3:
            radii_final = _growth_pass(best_pos, best_radii, max_iter=300, step=0.001)
            if _is_valid(best_pos, radii_final):
                score_final = _score(best_pos, radii_final)
                if score_final > best_score:
                    best_score = score_final
                    best_radii = radii_final.copy()
    
    # Fallback: simple grid
    if best_pos is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.005) for i in range(side) for j in range(side)]
        result = np.array(cells[:N])
        return result
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
