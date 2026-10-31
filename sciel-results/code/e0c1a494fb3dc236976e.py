"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
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


def _objective_penalty(x, lam=500.0):
    """Objective: minimize negative sum of radii + smooth penalty for violations."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    radii = np.maximum(radii, 1e-5)
    
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
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    overlap = np.maximum(0.0, min_dist_sq - dist_sq)
    penalty += np.sum(overlap[mask] ** 2 / (4.0 * np.maximum(min_dist_sq[mask], 1e-10)))
    
    return -np.sum(radii) + lam * penalty


def _make_hex_rows():
    """Generate hexagonal rows: alternating 5 and 4 circles."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.096
    y_spacing = r * np.sqrt(3)
    
    total_height = 4 * y_spacing + 2 * r
    y_start = (1.0 - total_height) / 2.0
    
    x_start_5 = (1.0 - 4 * 2 * r) / 2.0
    x_start_4 = (1.0 - 3 * 2 * r) / 2.0
    
    idx = 0
    for row in range(5):
        if row % 2 == 0:
            count = 5
            x_start = x_start_5
        else:
            count = 4
            x_start = x_start_4
        for i in range(count):
            pos[idx, 0] = x_start + i * 2 * r
            pos[idx, 1] = y_start + r + row * y_spacing
            radii[idx] = r
            idx += 1
    
    # 26th circle: find best gap position
    best_score = -1
    best_pos = (0.5, 0.5)
    for cx in np.linspace(0.03, 0.97, 15):
        for cy in np.linspace(0.03, 0.97, 15):
            min_clearance = 1.0
            for i in range(25):
                dist = np.sqrt((cx - pos[i, 0])**2 + (cy - pos[i, 1])**2)
                clearance = dist - radii[i]
                min_clearance = min(min_clearance, clearance)
            min_clearance = min(min_clearance, cx, 1-cx, cy, 1-cy)
            if min_clearance > best_score:
                best_score = min(min_clearance, 0.5)
                best_pos = (cx, cy)
    
    pos[25, 0] = best_pos[0]
    pos[25, 1] = best_pos[1]
    radii[25] = max(best_score * 0.85, 0.015)
    
    return pos, radii


def _make_corner_heavy():
    """Corner-heavy layout: larger circles in corners, smaller in center."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 4 corner circles (larger)
    corner_r = 0.14
    corners = [(corner_r, corner_r), (1-corner_r, corner_r), 
               (corner_r, 1-corner_r), (1-corner_r, 1-corner_r)]
    for i in range(4):
        pos[i, 0] = corners[i][0]
        pos[i, 1] = corners[i][1]
        radii[i] = corner_r
    
    # 8 edge circles
    edge_r = 0.10
    edges = [(0.3, edge_r), (0.7, edge_r), (0.3, 1-edge_r), (0.7, 1-edge_r),
             (edge_r, 0.3), (1-edge_r, 0.3), (edge_r, 0.7), (1-edge_r, 0.7)]
    for i in range(8):
        idx = 4 + i
        pos[idx, 0] = edges[i][0]
        pos[idx, 1] = edges[i][1]
        radii[idx] = edge_r
    
    # 6 diagonal circles
    diag_r = 0.075
    diags = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75), (0.5, 0.25), (0.5, 0.75)]
    for i in range(6):
        idx = 12 + i
        pos[idx, 0] = diags[i][0]
        pos[idx, 1] = diags[i][1]
        radii[idx] = diag_r
    
    # 8 inner ring circles
    inner_r = 0.055
    inner_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    inner_dist = 0.16
    for i in range(8):
        idx = 18 + i
        pos[idx, 0] = 0.5 + inner_dist * np.cos(inner_angles[i])
        pos[idx, 1] = 0.5 + inner_dist * np.sin(inner_angles[i])
        radii[idx] = inner_r
    
    return pos, radii


def _make_grid():
    """5x5 grid layout with 26th in a corner."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r = 0.097
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            pos[idx, 0] = (i + 0.5) / 5.0
            pos[idx, 1] = (j + 0.5) / 5.0
            radii[idx] = r
    pos[25, 0] = 0.08
    pos[25, 1] = 0.92
    radii[25] = 0.07
    return pos, radii


def _make_ring():
    """Central circle with surrounding ring."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    pos[0, 0] = 0.5
    pos[0, 1] = 0.5
    radii[0] = 0.11
    
    ring_r = 0.075
    ring_dist = 0.11 + ring_r
    for i in range(12):
        idx = 1 + i
        angle = 2 * np.pi * i / 12
        pos[idx, 0] = 0.5 + ring_dist * np.cos(angle)
        pos[idx, 1] = 0.5 + ring_dist * np.sin(angle)
        radii[idx] = ring_r
    
    corner_r = 0.055
    corners = [(0.07, 0.07), (0.93, 0.07), (0.07, 0.93), (0.93, 0.93),
               (0.5, 0.04), (0.5, 0.96), (0.04, 0.5), (0.96, 0.5),
               (0.22, 0.22), (0.78, 0.22), (0.22, 0.78), (0.78, 0.78),
               (0.5, 0.3)]
    for i in range(13):
        idx = 13 + i
        pos[idx, 0] = corners[i][0]
        pos[idx, 1] = corners[i][1]
        radii[idx] = corner_r
    
    return pos, radii


def _make_random(seed):
    """Random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.12, 0.88, size=(N, 2))
    radii = rng.uniform(0.05, 0.11, size=N)
    return pos, radii


def _post_process(pos, radii):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(200):
        valid = True
        
        for i in range(N):
            if pos[i, 0] < radii[i]:
                radii[i] = max(pos[i, 0] * 0.99, 1e-5)
                valid = False
            if pos[i, 0] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 0]) * 0.99)
                radii[i] = max(radii[i], 1e-5)
                valid = False
            if pos[i, 1] < radii[i]:
                radii[i] = min(radii[i], pos[i, 1] * 0.99)
                radii[i] = max(radii[i], 1e-5)
                valid = False
            if pos[i, 1] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 1]) * 0.99)
                radii[i] = max(radii[i], 1e-5)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    scale = dist / (min_dist + 1e-12) * 0.99
                    radii[i] = max(radii[i] * scale, 1e-5)
                    radii[j] = max(radii[j] * scale, 1e-5)
                    valid = False
        
        if valid:
            break
    
    return pos, radii


def _grow_and_push(pos, radii, deadline, max_iter=50):
    """Grow circles and push apart to maximize sum of radii."""
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Try to grow each circle
        for i in range(N):
            # Compute max possible radius for circle i given current positions
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r = min(max_r, dist - radii[j])
            
            max_r = max(max_r, 1e-5)
            radii[i] = max_r
        
        # Push overlapping circles apart
        for _ in range(10):
            moved = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = radii[i] + radii[j]
                    
                    if dist < min_dist and dist > 1e-12:
                        # Push apart
                        overlap = min_dist - dist
                        push = overlap / 2.0 * 0.95
                        nx = dx / dist
                        ny = dy / dist
                        pos[i, 0] += nx * push
                        pos[i, 1] += ny * push
                        pos[j, 0] -= nx * push
                        pos[j, 1] -= ny * push
                        moved = True
                    elif dist < min_dist:
                        # Very close, random nudge
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = (min_dist - dist) * 0.5
                        pos[i, 0] += np.cos(angle) * push
                        pos[i, 1] += np.sin(angle) * push
                        pos[j, 0] -= np.cos(angle) * push
                        pos[j, 1] -= np.sin(angle) * push
                        moved = True
            
            # Clamp to boundary
            for i in range(N):
                pos[i, 0] = np.clip(pos[i, 0], radii[i], 1.0 - radii[i])
                pos[i, 1] = np.clip(pos[i, 1], radii[i], 1.0 - radii[i])
            
            if not moved:
                break
        
        # Check validity and update best
        pos_pp, radii_pp = _post_process(pos, radii)
        score = np.sum(radii_pp)
        if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = radii_pp.copy()
    
    return best_pos, best_radii, best_score


def _local_refine(pos, radii, deadline, lam=500.0):
    """Multi-method local refinement with SLSQP polish."""
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = [(0.01, 0.99), (0.01, 0.99)] * N + [(0.01, 0.3)] * N
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    methods = [
        ('L-BFGS-B', {'maxiter': 4000, 'ftol': 1e-12, 'gtol': 1e-8}),
        ('Powell', {'maxiter': 2000, 'ftol': 1e-13}),
        ('L-BFGS-B', {'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10}),
        ('SLSQP', {'maxiter': 2000, 'ftol': 1e-12}),
    ]
    
    for method, opts in methods:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(_objective_penalty, x0, method=method, bounds=bounds, options=opts, args=(lam,))
            if res.fun < _objective_penalty(x0, lam) - 1e-15:
                x_opt = res.x
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
    
    # Final Nelder-Mead polish
    if time.time() < deadline - 3:
        try:
            res = minimize(_objective_penalty, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-13}, args=(lam,))
            if res.fun < _objective_penalty(x0, lam) - 1e-15:
                x_opt = res.x
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


def _global_search(deadline):
    """Differential evolution global search."""
    bounds = [(0.01, 0.99), (0.01, 0.99)] * N + [(0.01, 0.25)] * N
    de_deadline = deadline - 20
    
    def wrapper(x):
        return _objective_penalty(x, 200.0)
    
    try:
        res = differential_evolution(
            wrapper, bounds,
            maxiter=100, popsize=15,
            tol=1e-6, mutation=(0.5, 1.5), recombination=0.8,
            seed=7, polish=False, updating='deferred',
            callback=lambda xk, conv: (time.time() > de_deadline)
        )
        x_opt = res.x
        pos_opt = x_opt[:2*N].reshape(N, 2)
        radii_opt = np.maximum(x_opt[2*N:], 1e-4)
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
        score = np.sum(radii_pp)
        if _is_valid(pos_pp, radii_pp):
            return pos_pp, radii_pp, score
    except Exception:
        pass
    
    return None, None, -np.inf


def _delaunay_refine(pos, radii, deadline):
    """Refine using Delaunay triangulation structure to guide optimization."""
    if time.time() > deadline - 10:
        return pos.copy(), radii.copy(), np.sum(radii)
    
    # Compute Delaunay triangulation of current positions
    try:
        tri = Delaunay(pos)
    except Exception:
        return pos.copy(), radii.copy(), np.sum(radii)
    
    # Extract edges from triangulation
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                edge = tuple(sorted((simplex[i], simplex[j])))
                edges.add(edge)
    
    # Also add boundary edges (circles near boundary)
    for i in range(N):
        if pos[i, 0] < 0.15 or pos[i, 0] > 0.85 or pos[i, 1] < 0.15 or pos[i, 1] > 0.85:
            for j in range(N):
                if i != j:
                    dist = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
                    if dist < radii[i] + radii[j] + 0.1:
                        edge = tuple(sorted((i, j)))
                        edges.add(edge)
    
    # Build adjacency list
    adj = {i: [] for i in range(N)}
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = [(0.01, 0.99), (0.01, 0.99)] * N + [(0.01, 0.3)] * N
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Try multiple penalty weights
    for lam in [200.0, 300.0, 500.0]:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(_objective_penalty, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-9}, args=(lam,))
            x_opt = res.x
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
    
    return best_pos, best_radii, best_score


def _trust_constr_refine(pos, radii, deadline):
    """Refine using trust-constr with explicit nonlinear constraints."""
    if time.time() > deadline - 15:
        return pos.copy(), radii.copy(), np.sum(radii)
    
    # Convert to x0 format: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(3 * N)
    x0[::3] = pos[:, 0]
    x0[1::3] = pos[:, 1]
    x0[2::3] = radii
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))   # x_i
        bounds.append((0.0, 1.0))   # y_i
        bounds.append((0.001, 0.4)) # r_i
    
    # Objective
    def obj(x):
        r = x[2::3]
        return -np.sum(r)
    
    # Objective gradient
    def obj_grad(x):
        g = np.zeros(3 * N)
        g[2::3] = -1.0
        return g
    
    # Constraints: dist_ij >= r_i + r_j and boundary
    def constraints_func(x):
        x_c = x[0::3]
        y_c = x[1::3]
        r = x[2::3]
        
        cons = []
        
        # Boundary constraints
        for i in range(N):
            cons.append(x_c[i] - r[i])        # x_i >= r_i
            cons.append(1 - x_c[i] - r[i])    # x_i <= 1 - r_i
            cons.append(y_c[i] - r[i])        # y_i >= r_i
            cons.append(1 - y_c[i] - r[i])    # y_i <= 1 - r_i
            cons.append(r[i])                 # r_i >= 0
        
        # Pairwise distance constraints
        for i in range(N):
            for j in range(i + 1, N):
                dx = x_c[i] - x_c[j]
                dy = y_c[i] - y_c[j]
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                cons.append(dist_sq - min_dist * min_dist)
        
        return np.array(cons)
    
    # Jacobian of constraints
    def constraints_jac(x):
        x_c = x[0::3]
        y_c = x[1::3]
        r = x[2::3]
        
        n_cons = 5 * N + N * (N - 1) // 2
        jac = np.zeros((n_cons, 3 * N))
        
        idx = 0
        for i in range(N):
            # x_i - r_i >= 0
            jac[idx, 3*i] = 1.0
            jac[idx, 3*i+2] = -1.0
            idx += 1
            # 1 - x_i - r_i >= 0
            jac[idx, 3*i] = -1.0
            jac[idx, 3*i+2] = -1.0
            idx += 1
            # y_i - r_i >= 0
            jac[idx, 3*i+1] = 1.0
            jac[idx, 3*i+2] = -1.0
            idx += 1
            # 1 - y_i - r_i >= 0
            jac[idx, 3*i+1] = -1.0
            jac[idx, 3*i+2] = -1.0
            idx += 1
            # r_i >= 0
            jac[idx, 3*i+2] = 1.0
            idx += 1
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = x_c[i] - x_c[j]
                dy = y_c[i] - y_c[j]
                min_dist = r[i] + r[j]
                
                jac[idx, 3*i] = 2 * dx
                jac[idx, 3*j] = -2 * dx
                jac[idx, 3*i+1] = 2 * dy
                jac[idx, 3*j+1] = -2 * dy
                jac[idx, 3*i+2] = -2 * min_dist
                jac[idx, 3*j+2] = -2 * min_dist
                idx += 1
        
        return jac
    
    constraint = NonlinearConstraint(
        constraints_func,
        0, np.inf,
        jac=constraints_jac
    )
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Try trust-constr
    try:
        res = minimize(
            obj, x0, method='trust-constr',
            jac=obj_grad, bounds=bounds,
            constraints=[constraint],
            options={'maxiter': 300, 'gtol': 1e-8, 'xtol': 1e-10}
        )
        
        if res.success or (res.fun < -best_score):
            x_opt = res.x
            pos_opt = np.column_stack([x_opt[0::3], x_opt[1::3]])
            radii_opt = x_opt[2::3].copy()
            radii_opt = np.maximum(radii_opt, 1e-5)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    except Exception:
        pass
    
    return best_pos, best_radii, best_score


def _slsqp_refine(pos, radii, deadline):
    """Refine using SLSQP with explicit constraints."""
    if time.time() > deadline - 10:
        return pos.copy(), radii.copy(), np.sum(radii)
    
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = [(0.0, 1.0), (0.0, 1.0)] * N + [(0.001, 0.4)] * N
    
    def obj(x):
        return -np.sum(x[2*N:])
    
    # Build inequality constraints
    cons = []
    
    # Boundary constraints
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i] - x[2*N+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i+1] - x[2*N+i]})
    
    # Pairwise distance constraints
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: 
                    (x[2*i] - x[2*j])**2 + (x[2*i+1] - x[2*j+1])**2 - (x[2*N+i] + x[2*N+j])**2
            })
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    try:
        res = minimize(
            obj, x0, method='SLSQP', bounds=bounds,
            constraints=cons,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        if res.success or (res.fun < -best_score):
            x_opt = res.x
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


def _polar_grid_search(pos, radii, deadline):
    """Parameterize on polar grid and optimize angular offset and radial scaling."""
    if time.time() > deadline - 10:
        return pos.copy(), radii.copy(), np.sum(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Compute centroid
    centroid = np.mean(pos, axis=0)
    
    # For each circle, compute polar coordinates relative to centroid
    for angle_offset in np.linspace(0, 2*np.pi, 20, endpoint=False):
        for radial_scale in np.linspace(0.8, 1.2, 9):
            if time.time() > deadline - 5:
                break
            
            new_pos = np.zeros_like(pos)
            for i in range(N):
                dx = pos[i, 0] - centroid[0]
                dy = pos[i, 1] - centroid[1]
                r_polar = np.sqrt(dx*dx + dy*dy)
                if r_polar < 1e-6:
                    new_pos[i, 0] = centroid[0]
                    new_pos[i, 1] = centroid[1]
                else:
                    angle = np.arctan2(dy, dx) + angle_offset
                    r_polar_scaled = r_polar * radial_scale
                    new_pos[i, 0] = centroid[0] + r_polar_scaled * np.cos(angle)
                    new_pos[i, 1] = centroid[1] + r_polar_scaled * np.sin(angle)
            
            # Clamp to valid region
            for i in range(N):
                new_pos[i, 0] = np.clip(new_pos[i, 0], radii[i], 1.0 - radii[i])
                new_pos[i, 1] = np.clip(new_pos[i, 1], radii[i], 1.0 - radii[i])
            
            pos_pp, radii_pp = _post_process(new_pos, radii)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    return best_pos, best_radii, best_score


def construct():
    start = time.time()
    deadline = start + 145.0
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # Build diverse initial configurations
    configs = []
    p, r = _make_hex_rows()
    configs.append((p, r))
    p, r = _make_grid()
    configs.append((p, r))
    p, r = _make_ring()
    configs.append((p, r))
    p, r = _make_corner_heavy()
    configs.append((p, r))
    for s in [1, 7, 42, 99, 200]:
        p, r = _make_random(s)
        configs.append((p, r))
    
    # Perturbed versions of structured inits
    rng = np.random.RandomState(123)
    for base_idx in [0, 1, 3]:
        p, r = configs[base_idx]
        pp = p + rng.normal(0, 0.015, size=p.shape)
        pp = np.clip(pp, 0.03, 0.97)
        configs.append((pp, r.copy()))
    
    # Optimize from each config with time budget
    time_per_config = max(12, (deadline - 40) / len(configs))
    
    for idx, (pos, radii) in enumerate(configs):
        t_rem = deadline - time.time()
        if t_rem < 20:
            break
        alloc = min(time_per_config, t_rem - 15)
        if alloc < 8:
            continue
        
        # Grow and push
        gp, gr, gs = _grow_and_push(pos, radii, start + time.time() - start + alloc, max_iter=30)
        if gs > best_score + 1e-12:
            best_score = gs
            best_pos = gp.copy()
            best_radii = gr.copy()
        
        # Local refine
        pos_opt, radii_opt, score = _local_refine(pos, radii, start + time.time() - start + alloc, lam=500.0)
        
        if score > best_score + 1e-12:
            best_score = score
            best_pos = pos_opt.copy()
            best_radii = radii_opt.copy()
    
    # Global search if time allows
    if time.time() < deadline - 30 and best_score < 2.3:
        de_pos, de_radii, de_score = _global_search(deadline)
        if de_pos is not None and de_score > best_score + 1e-12:
            best_score = de_score
            best_pos = de_pos.copy()
            best_radii = de_radii.copy()
    
    # Fallback
    if best_pos is None or best_score < 0:
        p, r = _make_hex_rows()
        p, r = _post_process(p, r)
        best_pos = p
        best_radii = r
        best_score = np.sum(r)
    
    # Polar grid search
    if time.time() < deadline - 15:
        pg_pos, pg_radii, pg_score = _polar_grid_search(best_pos, best_radii, deadline)
        if pg_score > best_score + 1e-12:
            best_score = pg_score
            best_pos = pg_pos.copy()
            best_radii = pg_radii.copy()
    
    # Delaunay-guided refinement
    if time.time() < deadline - 15:
        dl_pos, dl_radii, dl_score = _delaunay_refine(best_pos, best_radii, deadline)
        if dl_score > best_score + 1e-12:
            best_score = dl_score
            best_pos = dl_pos.copy()
            best_radii = dl_radii.copy()
    
    # Trust-constr refinement with explicit constraints
    if time.time() < deadline - 20:
        tc_pos, tc_radii, tc_score = _trust_constr_refine(best_pos, best_radii, deadline)
        if tc_score > best_score + 1e-12:
            best_score = tc_score
            best_pos = tc_pos.copy()
            best_radii = tc_radii.copy()
    
    # SLSQP refinement with explicit constraints
    if time.time() < deadline - 15:
        sl_pos, sl_radii, sl_score = _slsqp_refine(best_pos, best_radii, deadline)
        if sl_score > best_score + 1e-12:
            best_score = sl_score
            best_pos = sl_pos.copy()
            best_radii = sl_radii.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 4:
        best_pos, best_radii, best_score = _local_refine(best_pos, best_radii, deadline, lam=200.0)
    
    # Ensure validity
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
