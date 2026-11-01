"""Circle packing in perimeter-4 rectangle: Direct-constraint NLP with multi-resolution grid seeding."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
_TIME_LIMIT = 145.0  # Safety margin before 150s kill


def _make_constraints(w0, h0):
    """Create constraint functions for SLSQP with fixed aspect ratio."""
    def constraints_factory():
        cons = []
        for i in range(N):
            # x[i] >= r[i]
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i] - z[3*i+2],
                         'jac': lambda z, i=i: np.concatenate([np.eye(3*N+1)[3*i], np.zeros(1)])})
            # x[i] + r[i] <= w0
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: w0 - z[3*i] - z[3*i+2],
                         'jac': lambda z, i=i: np.concatenate([-np.eye(3*N+1)[3*i], np.zeros(1)])})
            # y[i] >= r[i]
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i+1] - z[3*i+2],
                         'jac': lambda z, i=i: np.concatenate([np.eye(3*N+1)[3*i+1], np.zeros(1)])})
            # y[i] + r[i] <= h0
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: h0 - z[3*i+1] - z[3*i+2],
                         'jac': lambda z, i=i: np.concatenate([-np.eye(3*N+1)[3*i+1], np.zeros(1)])})
        
        for i in range(N):
            for j in range(i+1, N):
                cons.append({'type': 'ineq', 'fun': lambda z, i=i, j=j:
                             (z[3*i]-z[3*j])**2 + (z[3*i+1]-z[3*j+1])**2 - (z[3*i+2]+z[3*j+2])**2,
                             'jac': lambda z, i=i, j=j: _pairwise_jac(z, i, j)})
        return cons
    
    def _pairwise_jac(z, i, j):
        jac = np.zeros(3*N+1)
        dx = z[3*i] - z[3*j]
        dy = z[3*i+1] - z[3*j+1]
        ri = z[3*i+2]
        rj = z[3*j+2]
        jac[3*i] = 2*dx
        jac[3*i+1] = 2*dy
        jac[3*i+2] = -2*(ri + rj)
        jac[3*j] = -2*dx
        jac[3*j+1] = -2*dy
        jac[3*j+2] = -2*(ri + rj)
        return jac
    
    return constraints_factory


def _objective(z):
    """Minimize negative sum of radii."""
    r = z[2::3][:N]
    return -np.sum(r)


def _objective_jac(z):
    """Gradient of objective."""
    jac = np.zeros(3*N+1)
    jac[2::3][:N] = -1.0
    return jac


def _multi_resolution_grid(w, h, n, rng):
    """
    Generate a feasible initial configuration using multi-resolution grid.
    Returns z array of shape (3*N+1,) with z[-1] = w.
    """
    # Try different grid resolutions
    best_z = None
    best_sum_r = -1
    
    for cols in range(2, 8):
        rows = int(np.ceil(n / cols))
        if rows < 2:
            continue
        
        cw = w / cols
        ch = h / rows
        
        # Base inradius with small margin
        base_r = min(cw, ch) / 2.0 * 0.95
        
        x = np.zeros(n)
        y = np.zeros(n)
        r = np.zeros(n)
        
        for i in range(n):
            row, col = divmod(i, cols)
            if row >= rows:
                break
            # Add small random jitter to break symmetry
            jx = rng.uniform(-0.1, 0.1) * cw
            jy = rng.uniform(-0.1, 0.1) * ch
            x[i] = (col + 0.5) * cw + jx
            y[i] = (row + 0.5) * ch + jy
            r[i] = base_r
        
        # Clamp positions to ensure boundary feasibility
        for i in range(n):
            r[i] = min(r[i], x[i], w - x[i], y[i], h - y[i])
            if r[i] <= 0:
                r[i] = 0.01
                x[i] = r[i] + 0.001
                y[i] = r[i] + 0.001
        
        # Check pairwise feasibility (approximate)
        feasible = True
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < r[i] + r[j] - 1e-6:
                    # Shrink both slightly
                    shrink = (r[i] + r[j] - d) / 2.0 + 0.001
                    r[i] = max(0.01, r[i] - shrink)
                    r[j] = max(0.01, r[j] - shrink)
                    feasible = False
        
        if feasible or np.sum(r) > best_sum_r:
            z = np.zeros(3*N+1)
            z[0::3][:n] = x[:n]
            z[1::3][:n] = y[:n]
            z[2::3][:n] = r[:n]
            z[-1] = w
            if np.sum(r[:n]) > best_sum_r:
                best_sum_r = np.sum(r[:n])
                best_z = z
    
    if best_z is None:
        # Fallback: simple grid
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        cw = w / cols
        ch = h / rows
        r_cell = min(cw, ch) / 2.0 * 0.9
        z = np.zeros(3*N+1)
        for i in range(n):
            row, col = divmod(i, cols)
            z[3*i] = (col + 0.5) * cw
            z[3*i+1] = (row + 0.5) * ch
            z[3*i+2] = r_cell
        z[-1] = w
        best_z = z
    
    return best_z


def _slsqp_solve(z0, w, h, maxiter, ftol, maxls):
    """Run SLSQP with hard constraints."""
    constraints = _make_constraints(w, h)()
    bounds = [(0, 2.0)] * (3*N) + [(w, w)]  # Fix aspect ratio
    
    try:
        res = minimize(_objective, z0, method='SLSQP',
                       jac=_objective_jac,
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': ftol, 'maxls': maxls})
        return res.x
    except Exception:
        return z0


def _check_valid(x, y, r, w, h, tol=1e-9):
    """Check validity of circles in a w x h rectangle."""
    if w <= 0.05 or h <= 0.05:
        return False
    for i in range(len(r)):
        if r[i] <= 0:
            return False
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum * r_sum - tol:
                return False
    return True


def _greedy_expand(x, y, r, w, h, time_budget):
    """Iteratively expand each circle to its maximum possible radius."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    improved = True
    rounds = 0
    while improved and rounds < 50:
        if time.time() > time_budget:
            break
        improved = False
        rounds += 1
        for i in range(N):
            # Max radius from walls
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            # Max radius from other circles
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)
            if r_max > r[i] + 1e-8:
                r[i] = r_max
                improved = True
    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    rng = np.random.default_rng(42)

    # Try multiple aspect ratios
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8, 0.9, 1.1]
    
    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        
        if time.time() > deadline - 15:
            break
        
        # Generate initial guess using multi-resolution grid
        z0 = _multi_resolution_grid(w0, h0, N, rng)
        
        # SLSQP optimization with hard constraints
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(400, max(100, remaining * 40)))
        
        z_opt = _slsqp_solve(z0, w0, h0, maxiter=maxiter, ftol=1e-12, maxls=80)
        
        x = z_opt[0::3][:N]
        y = z_opt[1::3][:N]
        r = z_opt[2::3][:N]
        
        if _check_valid(x, y, r, w0, h0):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = z_opt.copy()
        
        # Greedy expansion polish
        if time.time() < deadline - 10:
            x2, y2, r2 = _greedy_expand(x, y, r, w0, h0, deadline - 5)
            if _check_valid(x2, y2, r2, w0, h0):
                val2 = float(np.sum(r2))
                if val2 > best_value:
                    best_value = val2
                    z_new = np.empty(3*N+1)
                    z_new[0::3][:N] = x2
                    z_new[1::3][:N] = y2
                    z_new[2::3][:N] = r2
                    z_new[-1] = w0
                    best_result = z_new

    # Final polish on best result if time permits
    if best_result is not None and time.time() < deadline - 8:
        w = best_result[-1]
        h = 2.0 - w
        x = best_result[0::3][:N]
        y = best_result[1::3][:N]
        r = best_result[2::3][:N]
        
        # One more SLSQP pass with tighter tolerance
        z_final = _slsqp_solve(best_result, w, h, maxiter=200, ftol=1e-14, maxls=100)
        x_f = z_final[0::3][:N]
        y_f = z_final[1::3][:N]
        r_f = z_final[2::3][:N]
        
        if _check_valid(x_f, y_f, r_f, w, h):
            val_f = float(np.sum(r_f))
            if val_f > best_value:
                best_value = val_f
                best_result = z_final.copy()
        
        # Final greedy expansion
        if time.time() < deadline - 3:
            x2, y2, r2 = _greedy_expand(x_f, y_f, r_f, w, h, deadline - 1)
            if _check_valid(x2, y2, r2, w, h):
                val2 = float(np.sum(r2))
                if val2 > best_value:
                    best_value = val2
                    z_new = np.empty(3*N+1)
                    z_new[0::3][:N] = x2
                    z_new[1::3][:N] = y2
                    z_new[2::3][:N] = r2
                    z_new[-1] = w
                    best_result = z_new

    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Final safety: ensure strict validity with minimal shrink
    for _ in range(20):
        r_trial = r * (1 - 1e-10)
        if _check_valid(x, y, r_trial, w, h, tol=1e-12):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
