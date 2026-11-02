<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import sys

N = 26

def _is_valid(x, y, r, tol=1e-9):
    """Check if configuration is valid (no overlaps, all inside unit square)."""
    for i in range(N):
        if r[i] < 0:
            return False
        if x[i] < r[i] - tol or x[i] > 1.0 - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > 1.0 - r[i] + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < r[i] + r[j] - tol:
                return False
    return True

def _score(x, y, r):
    """Return sum of radii if valid, else -inf."""
    if not _is_valid(x, y, r):
        return -np.inf
    return np.sum(r)

def _pbd_simulate(x, y, r, n_iters=200, pressure=0.002, max_r=0.35):
    """
    Position Based Dynamics simulation.
    Circles are particles; we iteratively:
      1. Grow radii by a small pressure step (global outward pressure).
      2. Resolve boundary violations (project centers inward).
      3. Resolve pairwise overlap violations (push centers apart, Gauss-Seidel).
    This keeps the configuration feasible at every step.
    Returns (x, y, r) after simulation.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    n = N

    for it in range(n_iters):
        # 1. Grow radii (pressure)
        r += pressure
        r = np.minimum(r, max_r)
        # Also cap so circle can fit in square
        r = np.minimum(r, 0.5)

        # 2. Resolve boundary: project centers to [r_i, 1 - r_i]
        for i in range(n):
            x[i] = np.clip(x[i], r[i], 1.0 - r[i])
            y[i] = np.clip(y[i], r[i], 1.0 - r[i])

        # 3. Resolve pairwise overlaps (Gauss-Seidel, multiple sweeps for convergence)
        for sweep in range(3):
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    min_d = r[i] + r[j]
                    if d < min_d - 1e-12:
                        if d < 1e-12:
                            # Degenerate: push apart along arbitrary direction
                            dx = 1e-6
                            dy = 0.0
                            d = 1e-6
                        overlap = min_d - d
                        # Split correction proportional to inverse mass (mass ~ r^2)
                        m_i = r[i] * r[i]
                        m_j = r[j] * r[j]
                        total_m = m_i + m_j
                        w_i = m_j / total_m  # heavier moves less
                        w_j = m_i / total_m
                        nx = dx / d
                        ny = dy / d
                        x[i] += w_i * overlap * nx
                        y[i] += w_i * overlap * ny
                        x[j] -= w_j * overlap * nx
                        y[j] -= w_j * overlap * ny
                        # Re-project to boundaries after push
                        x[i] = np.clip(x[i], r[i], 1.0 - r[i])
                        y[i] = np.clip(y[i], r[i], 1.0 - r[i])
                        x[j] = np.clip(x[j], r[j], 1.0 - r[j])
                        y[j] = np.clip(y[j], r[j], 1.0 - r[j])

    # Final validity check; if invalid, shrink radii slightly
    if not _is_valid(x, y, r, tol=1e-8):
        # Reduce radii to make feasible
        for _ in range(50):
            r *= 0.99
            # Re-project
            for i in range(n):
                x[i] = np.clip(x[i], r[i], 1.0 - r[i])
                y[i] = np.clip(y[i], r[i], 1.0 - r[i])
            if _is_valid(x, y, r, tol=1e-9):
                break
    return x, y, r

def _grid_init(n_cols, n_rows, jitter=0.0):
    """Grid initialization with optional jitter."""
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    count = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if count >= N:
                break
            x[count] = (col + 0.5) / n_cols
            y[count] = (row + 0.5) / n_rows
            r[count] = 0.5 / max(n_cols, n_rows) * 0.9
            if jitter > 0:
                x[count] += np.random.uniform(-jitter, jitter)
                y[count] += np.random.uniform(-jitter, jitter)
            count += 1
    if count < N:
        for i in range(count, N):
            angle = 2 * np.pi * (i - count) / max(1, N - count)
            x[i] = 0.5 + 0.2 * np.cos(angle)
            y[i] = 0.5 + 0.2 * np.sin(angle)
            r[i] = 0.03
    return x, y, r

def _random_init():
    """Random initialization with small radii."""
    x = np.random.uniform(0.1, 0.9, N)
    y = np.random.uniform(0.1, 0.9, N)
    r = np.random.uniform(0.02, 0.08, N)
    return x, y, r

def _slsqp_polish(x, y, r, deadline, max_iter=100):
    """
    SLSQP polish with explicit inequality constraints for non-overlap and boundaries.
    Maximizes sum(r) subject to constraints.
    """
    n = N
    p0 = np.concatenate([x, y, r])

    def objective(p):
        # Negative sum of radii (minimize)
        return -np.sum(p[2*n:])

    def objective_grad(p):
        g = np.zeros(3 * n)
        g[2*n:] = -1.0
        return g

    constraints = []
    # Boundary constraints: x_i >= r_i, x_i <= 1 - r_i, etc.
    # As inequality: g(p) >= 0
    for i in range(n):
        # x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[i] - p[2*n + i],
            'jac': lambda p, i=i: _grad_x_minus_r(p, i, 0)
        })
        # 1 - x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: 1.0 - p[i] - p[2*n + i],
            'jac': lambda p, i=i: _grad_1_minus_x_minus_r(p, i, 0)
        })
        # y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[n + i] - p[2*n + i],
            'jac': lambda p, i=i: _grad_x_minus_r(p, i, 1)
        })
        # 1 - y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: 1.0 - p[n + i] - p[2*n + i],
            'jac': lambda p, i=i: _grad_1_minus_x_minus_r(p, i, 1)
        })

    # Pairwise non-overlap: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    # Equivalent: d^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: _pair_dist_sq_minus_r_sum_sq(p, i, j),
                'jac': lambda p, i=i, j=j: _pair_grad(p, i, j)
            })

    bounds = [(0.0, 1.0)] * (2 * n) + [(0.01, 0.5)] * n

    try:
        result = minimize(
            objective,
            p0,
            jac=objective_grad,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}
        )
        if result.success or result.fun < 0:
            p = result.x
            x_new = p[:n].copy()
            y_new = p[n:2*n].copy()
            r_new = p[2*n:].copy()
            # Enforce validity
            if _is_valid(x_new, y_new, r_new):
                return x_new, y_new, r_new
    except Exception:
        pass
    return x, y, r

def _grad_x_minus_r(p, i, axis):
    """Gradient of (pos_i - r_i) where pos is x if axis=0, y if axis=1."""
    n = N
    g = np.zeros(3 * n)
    if axis == 0:
        g[i] = 1.0
    else:
        g[n + i] = 1.0
    g[2*n + i] = -1.0
    return g

def _grad_1_minus_x_minus_r(p, i, axis):
    """Gradient of (1 - pos_i - r_i)."""
    n = N
    g = np.zeros(3 * n)
    if axis == 0:
        g[i] = -1.0
    else:
        g[n + i] = -1.0
    g[2*n + i] = -1.0
    return g

def _pair_dist_sq_minus_r_sum_sq(p, i, j):
    """(x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2"""
    n = N
    dx = p[i] - p[j]
    dy = p[n + i] - p[n + j]
    d2 = dx * dx + dy * dy
    rs = p[2*n + i] + p[2*n + j]
    return d2 - rs * rs

def _pair_grad(p, i, j):
    """Gradient of pair constraint."""
    n = N
    g = np.zeros(3 * n)
    dx = p[i] - p[j]
    dy = p[n + i] - p[n + j]
    rs = p[2*n + i] + p[2*n + j]
    # d/dx_i of d^2 = 2*dx
    g[i] = 2 * dx
    g[j] = -2 * dx
    g[n + i] = 2 * dy
    g[n + j] = -2 * dy
    # d/dr_i of -(r_i + r_j)^2 = -2*(r_i + r_j)
    g[2*n + i] = -2 * rs
    g[2*n + j] = -2 * rs
    return g

def _enforce_validity(x, y, r):
    """Clip to ensure validity."""
    r = np.clip(r, 1e-6, 0.5)
    x = np.clip(x, r, 1.0 - r)
    y = np.clip(y, r, 1.0 - r)
    # If still overlapping, shrink
    for _ in range(100):
        if _is_valid(x, y, r):
            break
        r *= 0.98
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
    return x, y, r

def construct():
    """Construct 26 circles in unit square maximizing sum of radii using PBD + SLSQP."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    best_x, best_y, best_r = None, None, None
    best_score = -np.inf

    # Phase 1: PBD multi-start from diverse initializations
    inits = []
    # Grid configurations
    for nc, nr in [(5, 6), (6, 5), (4, 7), (7, 4), (5, 5)]:
        inits.append(_grid_init(nc, nr, jitter=0.0))
        inits.append(_grid_init(nc, nr, jitter=0.03))
    # Random inits
    for _ in range(8):
        inits.append(_random_init())

    for x0, y0, r0 in inits:
        if time.time() > deadline - 30:
            break
        # PBD simulation with varying pressure
        for pressure in [0.001, 0.002, 0.003]:
            if time.time() > deadline - 25:
                break
            x1, y1, r1 = _pbd_simulate(x0, y0, r0, n_iters=150, pressure=pressure, max_r=0.35)
            x1, y1, r1 = _enforce_validity(x1, y1, r1)
            sc = _score(x1, y1, r1)
            if sc > best_score:
                best_score = sc
                best_x, best_y, best_r = x1.copy(), y1.copy(), r1.copy()

    # Phase 2: SLSQP polish on best solutions
    if best_x is not None and best_score > 0:
        # Polish the best
        if time.time() < deadline - 10:
            xp, yp, rp = _slsqp_polish(best_x, best_y, best_r, deadline, max_iter=80)
            sc = _score(xp, yp, rp)
            if sc > best_score:
                best_score = sc
                best_x, best_y, best_r = xp, yp, rp

        # Try SLSQP on a few other good PBD results (re-run top inits)
        for x0, y0, r0 in inits[:6]:
            if time.time() > deadline - 15:
                break
            x1, y1, r1 = _pbd_simulate(x0, y0, r0, n_iters=100, pressure=0.002, max_r=0.35)
            x1, y1, r1 = _enforce_validity(x1, y1, r1)
            if time.time() < deadline - 10:
                xp, yp, rp = _slsqp_polish(x1, y1, r1, deadline, max_iter=60)
                sc = _score(xp, yp, rp)
                if sc > best_score:
                    best_score = sc
                    best_x, best_y, best_r = xp, yp, rp

    # Final validity enforcement
    if best_x is not None and best_score > 0:
        best_x, best_y, best_r = _enforce_validity(best_x, best_y, best_r)
        return np.column_stack([best_x, best_y, best_r])

    # Fallback: valid grid
    x, y, r = _grid_init(5, 6)
    x, y, r = _enforce_validity(x, y, r)
    return np.column_stack([x, y, r])

# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.spatial import Delaunay
import time
import sys

N = 26

def _is_valid(x, y, r, tol=1e-9):
    """Check if configuration is valid (no overlaps, all inside unit square)."""
    for i in range(N):
        if r[i] < 0:
            return False
        if x[i] < r[i] - tol or x[i] > 1.0 - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > 1.0 - r[i] + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < r[i] + r[j] - tol:
                return False
    return True

def _score(x, y, r):
    """Return sum of radii if valid, else -inf."""
    if not _is_valid(x, y, r):
        return -np.inf
    return np.sum(r)

def _build_contact_graph(x, y, r, contact_tol=0.05):
    """
    Build contact graph using Delaunay triangulation.
    Two circles are in contact if their Delaunay edge length is close to r_i + r_j.
    Returns list of (i, j) pairs that are in contact.
    """
    points = np.column_stack([x, y])
    try:
        tri = Delaunay(points)
    except Exception:
        # Fallback: use all pairs within some distance
        contacts = []
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < r[i] + r[j] + contact_tol:
                    contacts.append((i, j))
        return contacts
    
    contacts = []
    for simplex in tri.simplices:
        for k in range(3):
            i = simplex[k]
            j = simplex[(k+1) % 3]
            if i > j:
                i, j = j, i
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < r[i] + r[j] + contact_tol:
                contacts.append((i, j))
    
    # Remove duplicates
    seen = set()
    unique_contacts = []
    for i, j in contacts:
        key = (min(i,j), max(i,j))
        if key not in seen:
            seen.add(key)
            unique_contacts.append(key)
    return unique_contacts

def _contact_graph_optimize(x, y, r, max_iter=50, growth_rate=0.003):
    """
    Optimize using contact graph constraints.
    Iteratively grow radii and solve for positions that satisfy contact constraints.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    n = N
    
    for it in range(max_iter):
        # Build contact graph
        contacts = _build_contact_graph(x, y, r, contact_tol=0.08)
        
        if len(contacts) == 0:
            # No contacts: just grow and reposition
            r += growth_rate
            r = np.minimum(r, 0.35)
            x = np.clip(x, r, 1.0 - r)
            y = np.clip(y, r, 1.0 - r)
            continue
        
        # Define residual: for each contact edge, distance - (r_i + r_j) should be ~0
        # Also boundary residuals
        def residuals(p):
            px = p[:n]
            py = p[n:2*n]
            pr = p[2*n:]
            res = []
            
            # Contact constraints: d(i,j) = r_i + r_j
            for i, j in contacts:
                dx = px[i] - px[j]
                dy = py[i] - py[j]
                d = np.sqrt(dx*dx + dy*dy) + 1e-12
                res.append(d - (pr[i] + pr[j]))
            
            # Boundary constraints (soft): push centers inward
            for i in range(n):
                # Left boundary
                if px[i] < pr[i]:
                    res.append(px[i] - pr[i])
                # Right boundary
                if px[i] > 1.0 - pr[i]:
                    res.append(px[i] - (1.0 - pr[i]))
                # Bottom boundary
                if py[i] < pr[i]:
                    res.append(py[i] - pr[i])
                # Top boundary
                if py[i] > 1.0 - pr[i]:
                    res.append(py[i] - (1.0 - pr[i]))
            
            # Non-contact separation: ensure no overlaps for non-contact pairs
            for i in range(n):
                for j in range(i+1, n):
                    if (i, j) in contacts or (j, i) in contacts:
                        continue
                    dx = px[i] - px[j]
                    dy = py[i] - py[j]
                    d = np.sqrt(dx*dx + dy*dy) + 1e-12
                    min_d = pr[i] + pr[j]
                    if d < min_d:
                        res.append(d - min_d)
            
            return np.array(res)
        
        p0 = np.concatenate([x, y, r])
        
        try:
            result = least_squares(residuals, p0, method='trf', max_nfev=100, ftol=1e-10, xtol=1e-10)
            if result.success or len(result.fun) > 0:
                p = result.x
                x_new = p[:n]
                y_new = p[n:2*n]
                r_new = p[2*n:]
                
                # Validate and enforce
                x_new = np.clip(x_new, r_new, 1.0 - r_new)
                y_new = np.clip(y_new, r_new, 1.0 - r_new)
                
                if _is_valid(x_new, y_new, r_new, tol=1e-8):
                    x, y, r = x_new, y_new, r_new
                    # Grow radii slightly
                    r += growth_rate * 0.5
                    r = np.minimum(r, 0.35)
                    x = np.clip(x, r, 1.0 - r)
                    y = np.clip(y, r, 1.0 - r)
                else:
                    # Shrink slightly to recover
                    r *= 0.995
                    x = np.clip(x, r, 1.0 - r)
                    y = np.clip(y, r, 1.0 - r)
        except Exception:
            # Fallback: simple growth
            r += growth_rate * 0.3
            r = np.minimum(r, 0.35)
            x = np.clip(x, r, 1.0 - r)
            y = np.clip(y, r, 1.0 - r)
    
    return x, y, r

def _grid_init(n_cols, n_rows, jitter=0.0):
    """Grid initialization with optional jitter."""
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    count = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if count >= N:
                break
            x[count] = (col + 0.5) / n_cols
            y[count] = (row + 0.5) / n_rows
            r[count] = 0.5 / max(n_cols, n_rows) * 0.9
            if jitter > 0:
                x[count] += np.random.uniform(-jitter, jitter)
                y[count] += np.random.uniform(-jitter, jitter)
            count += 1
    if count < N:
        for i in range(count, N):
            angle = 2 * np.pi * (i - count) / max(1, N - count)
            x[i] = 0.5 + 0.2 * np.cos(angle)
            y[i] = 0.5 + 0.2 * np.sin(angle)
            r[i] = 0.03
    return x, y, r

def _random_init():
    """Random initialization with small radii."""
    x = np.random.uniform(0.1, 0.9, N)
    y = np.random.uniform(0.1, 0.9, N)
    r = np.random.uniform(0.02, 0.08, N)
    return x, y, r

def _slsqp_polish(x, y, r, max_iter=80):
    """
    SLSQP polish with contact-graph-aware constraints.
    """
    n = N
    p0 = np.concatenate([x, y, r])
    
    def objective(p):
        return -np.sum(p[2*n:])
    
    def objective_grad(p):
        g = np.zeros(3 * n)
        g[2*n:] = -1.0
        return g
    
    constraints = []
    # Boundary constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2*n+i], 'jac': lambda p, i=i: _grad_xm(p, i, 0)})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[i] - p[2*n+i], 'jac': lambda p, i=i: _grad_1xm(p, i, 0)})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[n+i] - p[2*n+i], 'jac': lambda p, i=i: _grad_xm(p, i, 1)})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[n+i] - p[2*n+i], 'jac': lambda p, i=i: _grad_1xm(p, i, 1)})
    
    # Pairwise non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i, j=j: _pair_d2_rs(p, i, j), 'jac': lambda p, i=i, j=j: _pair_g(p, i, j)})
    
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.01, 0.5)] * n
    
    try:
        result = minimize(objective, p0, jac=objective_grad, method='SLSQP', bounds=bounds,
                         constraints=constraints, options={'maxiter': max_iter, 'ftol': 1e-12})
        if result.success or result.fun < 0:
            p = result.x
            x_new, y_new, r_new = p[:n], p[n:2*n], p[2*n:]
            if _is_valid(x_new, y_new, r_new):
                return x_new, y_new, r_new
    except Exception:
        pass
    return x, y, r

def _grad_xm(p, i, axis):
    n = N
    g = np.zeros(3*n)
    if axis == 0:
        g[i] = 1.0
    else:
        g[n+i] = 1.0
    g[2*n+i] = -1.0
    return g

def _grad_1xm(p, i, axis):
    n = N
    g = np.zeros(3*n)
    if axis == 0:
        g[i] = -1.0
    else:
        g[n+i] = -1.0
    g[2*n+i] = -1.0
    return g

def _pair_d2_rs(p, i, j):
    n = N
    dx = p[i] - p[j]
    dy = p[n+i] - p[n+j]
    rs = p[2*n+i] + p[2*n+j]
    return dx*dx + dy*dy - rs*rs

def _pair_g(p, i, j):
    n = N
    g = np.zeros(3*n)
    dx = p[i] - p[j]
    dy = p[n+i] - p[n+j]
    rs = p[2*n+i] + p[2*n+j]
    g[i] = 2*dx
    g[j] = -2*dx
    g[n+i] = 2*dy
    g[n+j] = -2*dy
    g[2*n+i] = -2*rs
    g[2*n+j] = -2*rs
    return g

def _enforce_validity(x, y, r):
    """Clip to ensure validity."""
    r = np.clip(r, 1e-6, 0.5)
    x = np.clip(x, r, 1.0 - r)
    y = np.clip(y, r, 1.0 - r)
    for _ in range(100):
        if _is_valid(x, y, r):
            break
        r *= 0.98
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
    return x, y, r

def construct():
    """Construct 26 circles using contact-graph-based optimization."""
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    best_x, best_y, best_r = None, None, None
    best_score = -np.inf
    
    # Diverse initializations
    inits = []
    for nc, nr in [(5, 6), (6, 5), (4, 7), (7, 4), (5, 5), (6, 6)]:
        inits.append(_grid_init(nc, nr, jitter=0.0))
        inits.append(_grid_init(nc, nr, jitter=0.02))
    for _ in range(6):
        inits.append(_random_init())
    
    # Phase 1: Contact graph optimization from multiple starts
    for x0, y0, r0 in inits:
        if time.time() > deadline - 40:
            break
        for growth in [0.002, 0.003, 0.004]:
            if time.time() > deadline - 35:
                break
            x1, y1, r1 = _contact_graph_optimize(x0, y0, r0, max_iter=40, growth_rate=growth)
            x1, y1, r1 = _enforce_validity(x1, y1, r1)
            sc = _score(x1, y1, r1)
            if sc > best_score:
                best_score = sc
                best_x, best_y, best_r = x1.copy(), y1.copy(), r1.copy()
    
    # Phase 2: SLSQP polish on best
    if best_x is not None and best_score > 0 and time.time() < deadline - 15:
        xp, yp, rp = _slsqp_polish(best_x, best_y, best_r, max_iter=60)
        sc = _score(xp, yp, rp)
        if sc > best_score:
            best_score = sc
            best_x, best_y, best_r = xp, yp, rp
    
    # Final enforcement
    if best_x is not None and best_score > 0:
        best_x, best_y, best_r = _enforce_validity(best_x, best_y, best_r)
        return np.column_stack([best_x, best_y, best_r])
    
    # Fallback
    x, y, r = _grid_init(5, 6)
    x, y, r = _enforce_validity(x, y, r)
    return np.column_stack([x, y, r])

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
