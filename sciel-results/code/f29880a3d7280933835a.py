"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, linprog

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
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(500):
        valid = True
        # Check boundaries
        for i in range(N):
            if pos[i, 0] < radii[i]:
                radii[i] = max(0.001, pos[i, 0] * 0.999)
                valid = False
            if pos[i, 0] > 1.0 - radii[i]:
                radii[i] = min(radii[i], max(0.001, (1.0 - pos[i, 0]) * 0.999))
                valid = False
            if pos[i, 1] < radii[i]:
                radii[i] = min(radii[i], max(0.001, pos[i, 1] * 0.999))
                valid = False
            if pos[i, 1] > 1.0 - radii[i]:
                radii[i] = min(radii[i], max(0.001, (1.0 - pos[i, 1]) * 0.999))
                valid = False
        
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Shrink the larger circle more
                    if radii[i] >= radii[j]:
                        scale = (dist / (min_dist + 1e-12)) * 0.999
                        radii[i] = max(0.001, radii[i] * scale)
                    else:
                        scale = (dist / (min_dist + 1e-12)) * 0.999
                        radii[j] = max(0.001, radii[j] * scale)
                    valid = False
        
        if valid:
            break
    
    return pos, radii


def _growth_pass(pos, radii, deadline=None):
    """Try to grow each circle individually to fill available space."""
    pos = pos.copy()
    radii = radii.copy()
    improved = True
    n_iter = 0
    
    while improved and n_iter < 100:
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
            max_r = max(0.001, max_r * 0.9995)
            if max_r > radii[i] + 1e-8:
                radii[i] = max_r
                improved = True
    
    pos_pp, radii_pp = _post_process(pos, radii)
    return pos_pp, radii_pp


def _make_initial_config(strategy, seed=42):
    """Create an initial configuration based on strategy."""
    rng = np.random.RandomState(seed)
    
    if strategy == 'grid':
        side = 5
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                if idx < N:
                    pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                    radii[idx] = 0.5 / side - 0.005
        pos[25] = [0.5, 0.05]
        radii[25] = 0.04
        return pos, radii
    
    elif strategy == 'hexagonal':
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
    
    elif strategy == 'corner_heavy':
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        corner_r = 0.15
        pos[0] = [corner_r, corner_r]
        pos[1] = [1-corner_r, corner_r]
        pos[2] = [corner_r, 1-corner_r]
        pos[3] = [1-corner_r, 1-corner_r]
        radii[:4] = corner_r
        pos[4] = [0.5, 0.5]
        radii[4] = 0.12
        edge_r = 0.08
        pos[5] = [0.5, edge_r]
        pos[6] = [0.5, 1-edge_r]
        pos[7] = [edge_r, 0.5]
        pos[8] = [1-edge_r, 0.5]
        radii[5:9] = edge_r
        remaining = N - 9
        for i in range(remaining):
            angle = 2 * np.pi * i / remaining
            r = 0.06
            pos[9+i] = [0.5 + 0.25*np.cos(angle), 0.5 + 0.25*np.sin(angle)]
            radii[9+i] = r
        return pos, radii
    
    elif strategy == 'random':
        pos = rng.uniform(0.1, 0.9, size=(N, 2))
        radii = rng.uniform(0.03, 0.1, size=N)
        return pos, radii
    
    else:
        pos, radii = _make_initial_config('grid', seed)
        pos += rng.normal(0, 0.02, size=pos.shape)
        pos = np.clip(pos, 0.05, 0.95)
        return pos, radii


def _qp_polish(pos, radii, deadline=None, max_iter=50):
    """
    Convex QP subproblem approach: linearize non-overlap constraints 
    around current positions, solve for optimal radii and position adjustments.
    """
    pos = pos.copy()
    radii = radii.copy()
    n = N
    
    # Precompute pairs
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    n_pairs = len(pairs)
    
    for outer_iter in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Build linearized constraints
        # Variables: delta_pos (2n), delta_r (n)
        # Minimize: -sum(delta_r)
        # Constraints:
        #   1. Boundary: pos_i + delta_pos_i - (r_i + delta_r_i) >= 0
        #   2. Boundary: 1 - pos_i - delta_pos_i - (r_i + delta_r_i) >= 0
        #   3. Boundary: pos_i_y + delta_pos_iy - (r_i + delta_r_i) >= 0
        #   4. Boundary: 1 - pos_i_y - delta_pos_iy - (r_i + delta_r_i) >= 0
        #   5. Non-overlap (linearized): dist_ij + (delta_pos_i - delta_pos_j)·u_ij - delta_r_i - delta_r_j >= 0
        #      where u_ij = (pos_i - pos_j)/dist_ij
        #   6. r_i + delta_r_i >= 0.001
        #   7. r_i + delta_r_i <= 0.4
        #   8. |delta_pos| <= 0.05 (step size limit)
        #   9. |delta_r| <= 0.02 (step size limit)
        
        # Decision variables: x = [dx_0, dy_0, dx_1, dy_1, ..., dr_0, dr_1, ...]
        # Size: 3n
        
        # Objective: minimize -sum(dr)
        c = np.zeros(3*n)
        for i in range(n):
            c[2*n + i] = -1.0
        
        # Inequality constraints: A_ub @ x <= b_ub
        A_ub_rows = []
        b_ub_rows = []
        
        # Boundary constraints (4 per circle)
        for i in range(n):
            # x_i - r_i >= 0  =>  -(x_i - r_i) <= 0  =>  -dx_i - dr_i <= -(x_i - r_i)
            row = np.zeros(3*n)
            row[2*i] = -1.0
            row[2*n+i] = -1.0
            A_ub_rows.append(row)
            b_ub_rows.append(-(pos[i,0] - radii[i]))
            
            # 1 - x_i - r_i >= 0  =>  dx_i + dr_i <= 1 - x_i - r_i
            row = np.zeros(3*n)
            row[2*i] = 1.0
            row[2*n+i] = 1.0
            A_ub_rows.append(row)
            b_ub_rows.append(1.0 - pos[i,0] - radii[i])
            
            # y_i - r_i >= 0
            row = np.zeros(3*n)
            row[2*i+1] = -1.0
            row[2*n+i] = -1.0
            A_ub_rows.append(row)
            b_ub_rows.append(-(pos[i,1] - radii[i]))
            
            # 1 - y_i - r_i >= 0
            row = np.zeros(3*n)
            row[2*i+1] = 1.0
            row[2*n+i] = 1.0
            A_ub_rows.append(row)
            b_ub_rows.append(1.0 - pos[i,1] - radii[i])
        
        # Linearized non-overlap constraints
        for (i, j) in pairs:
            dx = pos[i,0] - pos[j,0]
            dy = pos[i,1] - pos[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1e-10:
                dist = 1e-10
                ux, uy = 1.0, 0.0
            else:
                ux = dx / dist
                uy = dy / dist
            
            # dist + (dx_i - dx_j)*ux + (dy_i - dy_j)*uy - dr_i - dr_j >= 0
            # => -(dist) - (dx_i - dx_j)*ux - (dy_i - dy_j)*uy + dr_i + dr_j <= 0
            # => dx_i*(-ux) + dx_j*(ux) + dy_i*(-uy) + dy_j*(uy) + dr_i + dr_j <= -dist
            row = np.zeros(3*n)
            row[2*i] = -ux
            row[2*i+1] = -uy
            row[2*j] = ux
            row[2*j+1] = uy
            row[2*n+i] = 1.0
            row[2*n+j] = 1.0
            A_ub_rows.append(row)
            b_ub_rows.append(-dist)
        
        # Radius bounds: r_i + dr_i >= 0.001  =>  -dr_i <= radii[i] - 0.001
        for i in range(n):
            row = np.zeros(3*n)
            row[2*n+i] = -1.0
            A_ub_rows.append(row)
            b_ub_rows.append(radii[i] - 0.001)
            
            # r_i + dr_i <= 0.4  =>  dr_i <= 0.4 - radii[i]
            row = np.zeros(3*n)
            row[2*n+i] = 1.0
            A_ub_rows.append(row)
            b_ub_rows.append(0.4 - radii[i])
        
        # Step size limits
        step_limit = 0.03
        for i in range(n):
            for coord in range(2):
                # dx_i <= step_limit
                row = np.zeros(3*n)
                row[2*i+coord] = 1.0
                A_ub_rows.append(row)
                b_ub_rows.append(step_limit)
                
                # -dx_i <= step_limit
                row = np.zeros(3*n)
                row[2*i+coord] = -1.0
                A_ub_rows.append(row)
                b_ub_rows.append(step_limit)
            
            # dr_i limits
            dr_limit = 0.015
            row = np.zeros(3*n)
            row[2*n+i] = 1.0
            A_ub_rows.append(row)
            b_ub_rows.append(dr_limit)
            
            row = np.zeros(3*n)
            row[2*n+i] = -1.0
            A_ub_rows.append(row)
            b_ub_rows.append(dr_limit)
        
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_rows)
        
        # Solve LP
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
            if res.success:
                dx = res.x[:2*n].reshape(n, 2)
                dr = res.x[2*n:]
                
                new_pos = pos + dx
                new_radii = radii + dr
                
                # Ensure validity
                new_pos, new_radii = _post_process(new_pos, new_radii)
                if _is_valid(new_pos, new_radii):
                    if np.sum(new_radii) > np.sum(radii) - 1e-10:
                        pos = new_pos
                        radii = new_radii
                    else:
                        break
                else:
                    break
            else:
                break
        except Exception:
            break
    
    return pos, radii


def _slsqp_polish(pos, radii, deadline=None, maxiter=200):
    """SLSQP polish with explicit constraints for tighter packing."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.35))
    
    # Build constraints
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N+i]})
    
    for i in range(N):
        for j in range(i+1, N):
            def overlap_fun(x, i=i, j=j):
                dx = x[2*i] - x[2*j]
                dy = x[2*i+1] - x[2*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                return dist - x[2*N+i] - x[2*N+j]
            constraints.append({'type': 'ineq', 'fun': overlap_fun})
    
    try:
        result = minimize(lambda x: -np.sum(x[2*N:]), x0, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': maxiter, 'ftol': 1e-12, 'disp': False})
        if result.fun < -np.sum(radii):
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = x_opt[2*N:]
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass
    
    return pos, radii


def _force_directed_anneal(pos, radii, n_iter=600, deadline=None):
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
        
        temp = 0.015 * np.exp(-0.003 * it)
        
        forces = np.zeros_like(pos)
        
        for i in range(N):
            if pos[i, 0] < radii[i] + 0.005:
                forces[i, 0] += 8.0 * (radii[i] + 0.005 - pos[i, 0])
            if pos[i, 0] > 1.0 - radii[i] - 0.005:
                forces[i, 0] -= 8.0 * (pos[i, 0] - (1.0 - radii[i] - 0.005))
            if pos[i, 1] < radii[i] + 0.005:
                forces[i, 1] += 8.0 * (radii[i] + 0.005 - pos[i, 1])
            if pos[i, 1] > 1.0 - radii[i] - 0.005:
                forces[i, 1] -= 8.0 * (pos[i, 1] - (1.0 - radii[i] - 0.005))
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-6:
                    overlap = min_dist - dist
                    force_mag = 15.0 * overlap
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                elif dist < min_dist * 1.3 and dist > 1e-6:
                    force_mag = 0.3 * (min_dist * 1.3 - dist)
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        vel = 0.85 * vel + 0.15 * forces
        pos += vel * 0.008 + np.random.normal(0, temp, size=pos.shape)
        
        for i in range(N):
            pos[i, 0] = np.clip(pos[i, 0], 0.01, 0.99)
            pos[i, 1] = np.clip(pos[i, 1], 0.01, 0.99)
        
        # Try to grow radii
        for i in range(N):
            max_grow = min(pos[i, 0] - radii[i], 1.0 - pos[i, 0] - radii[i],
                          pos[i, 1] - radii[i], 1.0 - pos[i, 1] - radii[i])
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                available = dist - radii[j]
                if available < max_grow:
                    max_grow = available
            
            if max_grow > 0.0005:
                radii[i] = min(radii[i] + max_grow * 0.3, 0.35)
        
        pos_pp, radii_pp = _post_process(pos, radii)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    return best_pos, best_radii


def _smooth_objective(x, beta=100.0):
    """Smooth objective with soft penalties for L-BFGS-B."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    penalty = 0.0
    for i in range(n):
        for coord in range(2):
            d1 = pos[i, coord] - radii[i]
            if d1 < 0:
                penalty += d1 * d1
            d2 = (1.0 - pos[i, coord]) - radii[i]
            if d2 < 0:
                penalty += d2 * d2
    
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


def _lbfgs_pass(pos, radii, deadline=None, maxiter=400):
    """L-BFGS-B with smooth objective for a robust first pass."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.35))
    
    try:
        result = minimize(_smooth_objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
        x_opt = result.x
        pos_opt = x_opt[:2*N].reshape(N, 2)
        radii_opt = x_opt[2*N:]
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
        if _is_valid(pos_pp, radii_pp):
            return pos_pp, radii_pp
    except Exception:
        pass
    
    return pos, radii


def _nelder_mead_pass(pos, radii, deadline=None, maxiter=800):
    """Nelder-Mead for exploration."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    try:
        result = minimize(_smooth_objective, x0, method='Nelder-Mead',
                          options={'maxiter': maxiter, 'fatol': 1e-10, 'xatol': 1e-8})
        x_opt = result.x
        pos_opt = x_opt[:2*N].reshape(N, 2)
        radii_opt = x_opt[2*N:]
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
        if _is_valid(pos_pp, radii_pp):
            return pos_pp, radii_pp
    except Exception:
        pass
    
    return pos, radii


def _polish(pos, radii, deadline=None):
    """Multi-method high-precision polish."""
    best_pos, best_radii = pos.copy(), radii.copy()
    best_score = np.sum(radii)
    
    # L-BFGS-B
    if deadline is None or time.time() < deadline - 10:
        try:
            pos_l, rad_l = _lbfgs_pass(best_pos, best_radii, deadline=deadline - 5)
            if _is_valid(pos_l, rad_l) and np.sum(rad_l) > best_score:
                best_score = np.sum(rad_l)
                best_pos, best_radii = pos_l, rad_l
        except Exception:
            pass
    
    # QP polish
    if deadline is None or time.time() < deadline - 10:
        try:
            pos_q, rad_q = _qp_polish(best_pos, best_radii, deadline=deadline - 5, max_iter=30)
            if _is_valid(pos_q, rad_q) and np.sum(rad_q) > best_score:
                best_score = np.sum(rad_q)
                best_pos, best_radii = pos_q, rad_q
        except Exception:
            pass
    
    # SLSQP
    if deadline is None or time.time() < deadline - 10:
        try:
            pos_s, rad_s = _slsqp_polish(best_pos, best_radii, deadline=deadline - 5, maxiter=150)
            if _is_valid(pos_s, rad_s) and np.sum(rad_s) > best_score:
                best_score = np.sum(rad_s)
                best_pos, best_radii = pos_s, rad_s
        except Exception:
            pass
    
    # Growth pass
    if deadline is None or time.time() < deadline - 5:
        try:
            pos_g, rad_g = _growth_pass(best_pos, best_radii, deadline=deadline - 2)
            if _is_valid(pos_g, rad_g) and np.sum(rad_g) > best_score:
                best_score = np.sum(rad_g)
                best_pos, best_radii = pos_g, rad_g
        except Exception:
            pass
    
    # Another QP pass after growth
    if deadline is None or time.time() < deadline - 5:
        try:
            pos_q2, rad_q2 = _qp_polish(best_pos, best_radii, deadline=deadline - 3, max_iter=20)
            if _is_valid(pos_q2, rad_q2) and np.sum(rad_q2) > best_score:
                best_score = np.sum(rad_q2)
                best_pos, best_radii = pos_q2, rad_q2
        except Exception:
            pass
    
    return best_pos, best_radii


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    strategies = ['grid', 'hexagonal', 'corner_heavy', 'random']
    
    # Phase 1: Multi-start with L-BFGS-B
    for strategy in strategies:
        if time.time() > deadline - 60:
            break
        
        for seed in range(3):
            if time.time() > deadline - 55:
                break
            
            pos, radii = _make_initial_config(strategy, seed)
            
            # L-BFGS-B pass
            if time.time() > deadline - 50:
                break
            try:
                pos_l, rad_l = _lbfgs_pass(pos, radii, deadline=deadline - 45)
                if _is_valid(pos_l, rad_l):
                    score = np.sum(rad_l)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_l.copy()
                        best_radii = rad_l.copy()
            except Exception:
                pass
            
            # Nelder-Mead from original
            if time.time() > deadline - 45:
                break
            try:
                pos_nm, rad_nm = _nelder_mead_pass(pos, radii, deadline=deadline - 40)
                if _is_valid(pos_nm, rad_nm):
                    score = np.sum(rad_nm)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_nm.copy()
                        best_radii = rad_nm.copy()
            except Exception:
                pass
    
    # Phase 2: Force-directed annealing from best
    if best_pos is not None and time.time() < deadline - 30:
        try:
            pos_fd, radii_fd = _force_directed_anneal(best_pos, best_radii, n_iter=500, deadline=deadline - 25)
            score_fd = np.sum(radii_fd)
            if _is_valid(pos_fd, radii_fd) and score_fd > best_score:
                best_score = score_fd
                best_pos = pos_fd.copy()
                best_radii = radii_fd.copy()
        except Exception:
            pass
    
    # Phase 3: High-precision polish
    if best_pos is not None and time.time() < deadline - 15:
        try:
            pos_pol, radii_pol = _polish(best_pos, best_radii, deadline=deadline - 5)
            score_pol = np.sum(radii_pol)
            if _is_valid(pos_pol, radii_pol) and score_pol > best_score:
                best_score = score_pol
                best_pos = pos_pol.copy()
                best_radii = radii_pol.copy()
        except Exception:
            pass
    
    # Phase 4: Final QP + growth
    if best_pos is not None and time.time() < deadline - 8:
        try:
            pos_q, rad_q = _qp_polish(best_pos, best_radii, deadline=deadline - 5, max_iter=25)
            if _is_valid(pos_q, rad_q):
                pos_g, rad_g = _growth_pass(pos_q, rad_q, deadline=deadline - 3)
                score_g = np.sum(rad_g)
                if _is_valid(pos_g, rad_g) and score_g > best_score:
                    best_score = score_g
                    best_pos = pos_g.copy()
                    best_radii = rad_g.copy()
        except Exception:
            pass
    
    # Fallback
    if best_pos is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.005) for i in range(side) for j in range(side)]
        result = np.array(cells[:N])
        return result
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
