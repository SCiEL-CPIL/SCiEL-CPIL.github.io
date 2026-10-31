"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _softplus(x, beta=50.0):
    """Numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 20.0/beta, x, np.log1p(np.exp(beta * x)) / beta)

def _soft_max(v, beta=50.0):
    """Smooth approximation of max(0, v) using softplus."""
    return _softplus(v, beta)

def _objective(x, penalty_weight=100.0, beta=50.0):
    """
    Objective: minimize -sum(radii) + penalty * smooth_violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    # Boundary violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap violations (vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    viol = np.maximum(0.0, min_dist_sq - dist_sq)
    overlap_pen = np.sum(viol[mask] * viol[mask])
    
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _objective_smooth(x, penalty_weight=100.0, beta=30.0):
    """
    Objective with softplus-based smooth penalties.
    Uses a soft-min (log-sum-exp) surrogate for the distance to the boundary
    and a smooth max(0, required - dist) for overlaps, which is more
    differentiable than the squared-penalty objective.
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)

    # Smooth boundary: soft-min of the four wall distances
    d = np.stack([pos[:, 0], 1.0 - pos[:, 0], pos[:, 1], 1.0 - pos[:, 1]], axis=1)
    neg_d = -d
    m = np.max(neg_d, axis=1, keepdims=True)
    neg_d_clipped = np.clip(neg_d, -50.0, 50.0)
    soft_max_neg = m + np.log(np.sum(np.exp(neg_d_clipped - m), axis=1, keepdims=True))
    soft_min_d = -soft_max_neg[:, 0]
    boundary_pen = np.sum(_soft_max(radii - soft_min_d, beta))

    # Smooth overlap: for each pair, required = r_i + r_j, dist = ||p_i - p_j||
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    viol_raw = min_dist - dist
    overlap_pen = np.sum(_soft_max(viol_raw[mask], beta))

    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _is_valid(pos, radii, tol=1e-9):
    """Strict validity check with tight tolerance."""
    n = len(radii)
    # Boundary check
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    # Overlap check
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # dist >= min_dist + tol  =>  dist_sq >= (min_dist + tol)^2
    # To avoid sqrt, check dist_sq < (min_dist - tol)^2 is the failure condition for "dist < min_dist"
    # Actually, we want dist >= min_dist. Failure if dist < min_dist.
    # dist_sq < min_dist^2 is the failure condition.
    # With tolerance: we want dist >= min_dist + tol. Failure if dist < min_dist + tol.
    # dist_sq < (min_dist + tol)^2.
    
    # Let's stick to the simpler and safer: dist_sq < (min_dist - tol)^2 is definitely invalid?
    # No, if dist_sq < min_dist^2, they overlap.
    # We want to ensure no overlap.
    # Check: dist_sq + tol^2 < min_dist^2 ?
    
    # Simplest robust check:
    # dist = sqrt(dist_sq)
    # if dist < min_dist - tol: return False
    
    # Vectorized without sqrt for speed, but careful with tol.
    # (min_dist - tol)^2 < dist_sq
    # If dist_sq < (min_dist - tol)^2, it's invalid.
    
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity - robust version."""
    pos = pos.copy()
    radii = radii.copy()
    
    # Ensure positions are within bounds
    pos = np.clip(pos, 0.0, 1.0)
    
    for _ in range(1000):
        valid = True
        
        # 1. Boundary check: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
        max_r_wall = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                                np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
        # Apply a small safety margin for floating point errors
        max_r_wall_safe = max_r_wall * 0.99999
        over_boundary = radii > max_r_wall_safe
        if np.any(over_boundary):
            radii[over_boundary] = max_r_wall_safe[over_boundary]
            valid = False
        
        # 2. Overlap check: dist(i,j) >= r_i + r_j
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(dist_sq)
        min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        # Check violation: min_dist - dist > tol
        viol = min_dist - dist
        # We need min_dist <= dist - tol  =>  viol <= -tol
        # If viol > 0, we have overlap.
        # To be safe, we want viol <= -1e-9
        over = viol[mask] > -1e-9
        
        if np.any(over):
            # Find the worst violating pair
            # We want to shrink radii to satisfy: r_i + r_j <= dist - tol
            # Simple strategy: shrink both radii by half of the violation amount (distributed)
            # Or just shrink the larger one? Let's shrink both proportionally or just reduce the sum.
            # A robust way: for each violating pair, reduce r_i and r_j such that (r_i + r_j) = dist - tol.
            # Since we do this iteratively, we can just shrink the radii involved in the most severe violation.
            
            # Get indices of violating pairs
            viol_matrix = np.where(mask, viol, -np.inf)
            idx = np.unravel_index(np.argmax(viol_matrix), (N, N))
            i, j = idx
            
            d_ij = dist[i, j]
            target_sum = d_ij - 1e-9
            current_sum = radii[i] + radii[j]
            
            if current_sum > target_sum:
                # Scale down both radii to fit
                scale = target_sum / current_sum
                radii[i] *= scale
                radii[j] *= scale
                valid = False
        
        if valid:
            break
            
    return pos, radii

def _initial_best_fit():
    """Greedy best-fit initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        rng = np.random.RandomState(i * 1000 + 42)
        
        for _ in range(5000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            if max_r_circ > best_r and max_r_circ > 0.003:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        
        if not found or best_r < 0.003:
            best_x, best_y = 0.5, 0.5
            best_r = 0.003
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.97
        
    return pos, radii

def _initial_hexagonal():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r
        
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        
        if k >= N:
            break
            
    while k < N:
        rng = np.random.RandomState(k * 7 + 13)
        for _ in range(100):
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0])**2 + (y - pos[j, 1])**2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.95
                k += 1
                break
        if k < N:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
            
    return pos, radii

def _initial_corner_biased(seed=7):
    """Corner-biased initialization."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx + rng.uniform(-0.03, 0.03), cy + rng.uniform(-0.03, 0.03)]
        radii[i] = 0.18
    
    for i in range(4, N):
        if rng.random() < 0.4:
            edge = rng.randint(4)
            t = rng.uniform(0.1, 0.9)
            if edge == 0:
                x, y = t, 0.08
            elif edge == 1:
                x, y = t, 0.92
            elif edge == 2:
                x, y = 0.08, t
            else:
                x, y = 0.92, t
        else:
            x, y = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        for j in range(i):
            dx = x - pos[j, 0]
            dy = y - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, dist - radii[j])
        
        if max_r < 0.01:
            for attempt in range(200):
                x2 = rng.uniform(0.05, 0.95)
                y2 = rng.uniform(0.05, 0.95)
                max_r2 = min(x2, 1.0 - x2, y2, 1.0 - y2)
                for j in range(i):
                    dx = x2 - pos[j, 0]
                    dy = y2 - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    max_r2 = min(max_r2, dist - radii[j])
                if max_r2 > max_r:
                    max_r = max_r2
                    x, y = x2, y2
        
        pos[i] = [x, y]
        radii[i] = max(max_r, 0.01) * 0.95
        
    return pos, radii

def _initial_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _incremental_energy(pos, radii, changed_idx, penalty_weight=50.0):
    """
    Fast incremental energy computation for a single changed circle.
    Only recomputes terms involving the changed circle.
    """
    n = len(radii)
    x, y, r = pos[changed_idx, 0], pos[changed_idx, 1], radii[changed_idx]
    
    # Boundary penalty for changed circle
    bv = np.maximum(0.0, r - x) + np.maximum(0.0, x - (1.0 - r)) + \
         np.maximum(0.0, r - y) + np.maximum(0.0, y - (1.0 - r))
    boundary_pen = bv * bv
    
    # Overlap penalty with all other circles
    overlap_pen = 0.0
    for j in range(n):
        if j == changed_idx:
            continue
        dx = x - pos[j, 0]
        dy = y - pos[j, 1]
        dist_sq = dx * dx + dy * dy
        min_dist = r + radii[j]
        viol = max(0.0, min_dist * min_dist - dist_sq)
        overlap_pen += viol * viol
    
    return penalty_weight * (boundary_pen + overlap_pen)

def _full_energy(pos, radii, penalty_weight=50.0):
    """Full energy computation (same as _objective but returns penalty only)."""
    n = len(radii)
    
    # Boundary violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    viol = np.maximum(0.0, min_dist_sq - dist_sq)
    overlap_pen = np.sum(viol[mask] * viol[mask])
    
    return penalty_weight * (boundary_pen + overlap_pen)

def _gradient_descent(pos, radii, n_steps=5000, lr_init=0.005, lr_decay=0.999, seed=0, deadline=None):
    """
    Gradient descent with correct pairwise force computation.
    Uses repulsive forces for overlaps and attractive forces for boundary violations.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    lr = lr_init
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros((n, 2))
        r_forces = np.zeros(n)
        
        # Boundary forces: push circles away from walls
        for i in range(n):
            x, y, r = pos[i, 0], pos[i, 1], radii[i]
            # Left wall
            if x < r:
                forces[i, 0] += (r - x) * 10.0
                r_forces[i] += (r - x) * 5.0
            # Right wall
            if x > 1.0 - r:
                forces[i, 0] -= (x - (1.0 - r)) * 10.0
                r_forces[i] += (x - (1.0 - r)) * 5.0
            # Bottom wall
            if y < r:
                forces[i, 1] += (r - y) * 10.0
                r_forces[i] += (r - y) * 5.0
            # Top wall
            if y > 1.0 - r:
                forces[i, 1] -= (y - (1.0 - r)) * 10.0
                r_forces[i] += (y - (1.0 - r)) * 5.0
        
        # Overlap forces: repel overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx * dx + dy * dy
                sum_r = radii[i] + radii[j]
                
                if dist_sq < sum_r * sum_r:
                    dist = np.sqrt(dist_sq + 1e-12)
                    overlap = sum_r - dist
                    if dist > 1e-10:
                        fx = (dx / dist) * overlap * 5.0
                        fy = (dy / dist) * overlap * 5.0
                    else:
                        # Random direction if coincident
                        angle = rng.uniform(0, 2 * np.pi)
                        fx = np.cos(angle) * overlap * 5.0
                        fy = np.sin(angle) * overlap * 5.0
                    
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                    r_forces[i] += overlap * 2.0
                    r_forces[j] += overlap * 2.0
        
        # Update positions and radii
        for i in range(n):
            pos[i, 0] += lr * forces[i, 0]
            pos[i, 1] += lr * forces[i, 1]
            radii[i] -= lr * r_forces[i]
            
            # Clamp
            pos[i, 0] = np.clip(pos[i, 0], 0.001, 0.999)
            pos[i, 1] = np.clip(pos[i, 1], 0.001, 0.999)
            radii[i] = np.clip(radii[i], 0.001, 0.4)
        
        # Adaptive learning rate
        total_force = np.sum(forces ** 2) + np.sum(r_forces ** 2)
        if total_force < 1e-6:
            lr *= 0.9
        else:
            lr *= lr_decay
        
        # Periodically check validity and update best
        if step % 100 == 0:
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    
    # Final check
    pos_pp, radii_pp = _post_process(pos, radii)
    if _is_valid(pos_pp, radii_pp):
        score = np.sum(radii_pp)
        if score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def _slsqp_constraints(pos, radii):
    """Generate constraint functions for SLSQP with hard non-overlap constraints."""
    n = len(radii)
    constraints = []
    
    # Boundary constraints: x_i >= r_i, x_i <= 1-r_i, y_i >= r_i, y_i <= 1-r_i
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*n+i]})
    
    # Non-overlap constraints: dist(i,j) >= r_i + r_j
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[2*i] - x[2*j])**2 + (x[2*i+1] - x[2*j+1])**2 - (x[2*n+i] + x[2*n+j])**2
            })
    
    return constraints

def _solve_radii_qp(pos, deadline=None):
    """
    Solve a QP to find maximum radii given fixed positions.
    Maximize sum(r_i) subject to:
      r_i <= min(x_i, 1-x_i, y_i, 1-y_i)  for all i
      r_i + r_j <= ||p_i - p_j||           for all pairs i<j
      r_i >= 0
    This is a linear program actually.
    """
    from scipy.optimize import linprog
    
    n = N
    # Variables: r_0, ..., r_{n-1}
    c = -np.ones(n)  # maximize sum(r)
    
    # Inequality constraints: A_ub @ r <= b_ub
    # Boundary: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    max_r_wall = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                            np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    
    # Pair constraints: r_i + r_j <= dist(i,j)
    A_rows = []
    b_rows = []
    
    # Boundary constraints: r_i <= max_r_wall[i]
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        A_rows.append(row)
        b_rows.append(max_r_wall[i])
    
    # Pair constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_rows.append(row)
            b_rows.append(d)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    bounds = [(0.0, 0.5)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            radii = np.maximum(res.x, 1e-6)
            # Verify validity
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass
    
    return pos, radii

def _contact_graph_search(pos, radii, deadline, n_iterations=50):
    """
    Local search: identify contact pairs (tight constraints) and try small
    position perturbations that maintain contacts while allowing radius growth.
    """
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for it in range(n_iterations):
        if time.time() > deadline - 3:
            break
        
        # Identify contact pairs (distance close to sum of radii)
        dx = best_pos[:, 0, np.newaxis] - best_pos[np.newaxis, :, 0]
        dy = best_pos[:, 1, np.newaxis] - best_pos[np.newaxis, :, 1]
        dist = np.sqrt(dx * dx + dy * dy + 1e-15)
        min_dist = best_radii[:, np.newaxis] + best_radii[np.newaxis, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        # Contact: dist < min_dist + epsilon
        contacts = mask & (dist < min_dist + 0.005)
        
        # Try random small perturbations
        rng = np.random.RandomState(it * 7919 + 12345)
        for attempt in range(20):
            # Random small perturbation
            delta = rng.normal(0, 0.003, (N, 2))
            new_pos = best_pos + delta
            new_pos = np.clip(new_pos, 0.01, 0.99)
            
            # Try to grow radii using LP
            new_pos_lp, new_radii_lp = _solve_radii_qp(new_pos)
            score = np.sum(new_radii_lp)
            
            if score > best_score + 1e-8:
                best_pos = new_pos_lp.copy()
                best_radii = new_radii_lp.copy()
                best_score = score
    
    return best_pos, best_radii

def _polish(pos, radii, deadline):
    """High-precision polishing with multiple strategies."""
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.5))
        
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Strategy 1: SLSQP with hard constraints
    try:
        cons = _slsqp_constraints(pos, radii)
        result = minimize(lambda x: -np.sum(np.maximum(x[2*N:], 1e-6)), x, method='SLSQP',
                          bounds=bounds, constraints=cons,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.success or result.fun < -np.sum(np.maximum(x[2*N:], 1e-6)) + 1e-10:
            x = result.x
            pos_slq = x[:2*N].reshape(N, 2)
            rad_slq = np.maximum(x[2*N:], 1e-4)
            pos_pp, radii_pp = _post_process(pos_slq, rad_slq)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    except Exception:
        pass
    
    # Strategy 2: QP radius solve at current positions
    if time.time() < deadline - 10:
        try:
            pos_qp, radii_qp = _solve_radii_qp(best_pos)
            if np.sum(radii_qp) > best_score:
                best_pos = pos_qp.copy()
                best_radii = radii_qp.copy()
                best_score = np.sum(radii_qp)
        except Exception:
            pass
    
    # Strategy 3: Gradient descent for fine-tuning
    if time.time() < deadline - 8:
        try:
            pos_gd, rad_gd = _gradient_descent(best_pos, best_radii, n_steps=3000, lr_init=0.003, seed=42, deadline=deadline - 5)
            pos_pp, radii_pp = _post_process(pos_gd, rad_gd)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Strategy 4: Contact graph local search
    if time.time() < deadline - 5:
        try:
            pos_cg, radii_cg = _contact_graph_search(best_pos, best_radii, deadline - 3, n_iterations=30)
            if np.sum(radii_cg) > best_score:
                best_pos = pos_cg.copy()
                best_radii = radii_cg.copy()
                best_score = np.sum(radii_cg)
        except Exception:
            pass
    
    # Strategy 5: L-BFGS-B with high penalty for final fine-tuning
    if time.time() < deadline - 3:
        try:
            x_final = np.concatenate([best_pos.flatten(), best_radii])
            result = minimize(_objective, x_final, method='L-BFGS-B', bounds=bounds,
                              args=(10000.0,),
                              options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            if result.fun < _objective(x_final, 10000.0) - 1e-12:
                x = result.x
                pos_lbfgs = x[:2*N].reshape(N, 2)
                rad_lbfgs = np.maximum(x[2*N:], 1e-4)
                pos_pp, radii_pp = _post_process(pos_lbfgs, rad_lbfgs)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
        except Exception:
            pass
    
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate multiple initializations
    inits = []
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    pos_hex, rad_hex = _initial_hexagonal()
    inits.append((pos_hex, rad_hex))
    
    pos_cb, rad_cb = _initial_corner_biased(seed=7)
    inits.append((pos_cb, rad_cb))
    
    for i in range(2):
        pos_r, rad_r = _initial_random(100 + i)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
        
    penalty_stages = [50.0, 500.0, 2000.0]
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()
        
        # Stage 1: Standard objective
        for p_w in penalty_stages:
            if time.time() > deadline - 10:
                break
            try:
                result = minimize(_objective, current_x, method='L-BFGS-B', bounds=bounds,
                                  args=(p_w,),
                                  options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-8})
                if result.fun < _objective(current_x, p_w) - 1e-10:
                    current_x = result.x
            except Exception:
                pass
            
            pos_opt = current_x[:2*N].reshape(N, 2)
            radii_opt = np.maximum(current_x[2*N:], 1e-4)
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        
        # Stage 2: Smooth objective
        if time.time() < deadline - 10:
            for p_w in [500.0, 2000.0]:
                if time.time() > deadline - 5:
                    break
                try:
                    result = minimize(_objective_smooth, current_x, method='L-BFGS-B', bounds=bounds,
                                      args=(p_w, 30.0),
                                      options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-8})
                    if result.fun < _objective_smooth(current_x, p_w, 30.0) - 1e-10:
                        current_x = result.x
                except Exception:
                    pass
                
                pos_opt = current_x[:2*N].reshape(N, 2)
                radii_opt = np.maximum(current_x[2*N:], 1e-4)
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        
        # Stage 3: Gradient descent with more steps
        if time.time() < deadline - 8:
            try:
                pos_gd, rad_gd = _gradient_descent(
                    current_x[:2*N].reshape(N, 2),
                    np.maximum(current_x[2*N:], 1e-4),
                    n_steps=5000, lr_init=0.005, lr_decay=0.9995,
                    seed=idx * 100, deadline=deadline - 5
                )
                pos_pp, radii_pp = _post_process(pos_gd, rad_gd)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass
    
    # Final polish - multiple attempts from different starting points
    if time.time() < deadline - 8:
        try:
            pos_pol, rad_pol = _polish(best_pos, best_radii, deadline)
            pos_pp, radii_pp = _post_process(pos_pol, rad_pol)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Second polish attempt with slightly perturbed positions
    if time.time() < deadline - 5:
        try:
            rng = np.random.RandomState(12345)
            perturbed_pos = best_pos + rng.normal(0, 0.002, (N, 2))
            perturbed_pos = np.clip(perturbed_pos, 0.01, 0.99)
            pos_pol2, rad_pol2 = _polish(perturbed_pos, best_radii.copy(), deadline)
            pos_pp2, radii_pp2 = _post_process(pos_pol2, rad_pol2)
            score2 = np.sum(radii_pp2)
            if _is_valid(pos_pp2, radii_pp2) and score2 > best_score:
                best_score = score2
                best_pos = pos_pp2.copy()
                best_radii = radii_pp2.copy()
        except Exception:
            pass
    
    # Third: QP radius solve at final positions
    if time.time() < deadline - 2:
        try:
            pos_final, radii_final = _solve_radii_qp(best_pos)
            score_final = np.sum(radii_final)
            if _is_valid(pos_final, radii_final) and score_final > best_score:
                best_score = score_final
                best_pos = pos_final.copy()
                best_radii = radii_final.copy()
        except Exception:
            pass
    
    # Final validation and enforcement
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        # Fallback to a simple valid grid if post-processing failed
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        # 5x5 grid = 25 circles, plus 1 in center?
        # 5x5 grid of circles with r=0.095 fits?
        # Spacing 0.2. Diameter 0.19. 0.2 > 0.19. OK.
        for i in range(5):
            for j in range(5):
                if k < N:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.09
                    k += 1
        if k < N:
             # Add remaining circles with very small radius if needed
             for j in range(k, N):
                 pos[j] = [0.5, 0.5] # Overlap will be handled by post_process shrinking
                 radii[j] = 0.01
                 
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
        
        # If still invalid (should not happen with grid), force tiny radii
        if not _is_valid(best_pos, best_radii):
            best_radii[:] = 0.001
            best_pos, best_radii = _post_process(best_pos, best_radii)

    # One final QP attempt to maximize radii on the now-valid configuration
    try:
        pos_q, rad_q = _solve_radii_qp(best_pos)
        pos_q, rad_q = _post_process(pos_q, rad_q)
        if _is_valid(pos_q, rad_q) and np.sum(rad_q) > np.sum(best_radii):
            best_pos = pos_q
            best_radii = rad_q
    except Exception:
        pass
        
    # Final safety check
    if not _is_valid(best_pos, best_radii):
        # This should never happen, but if it does, return a trivially valid set
        best_pos = np.tile([0.5, 0.5], (N, 1))
        best_radii = np.full(N, 0.001)
        best_pos, best_radii = _post_process(best_pos, best_radii)

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
