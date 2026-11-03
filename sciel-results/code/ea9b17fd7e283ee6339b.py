"""Packing 50 equal circles in a unit square [-0.5, 0.5]^2.
Uses active-set QCQP: identify binding constraints (touching pairs, wall
contacts), solve the resulting nonlinear system for a local optimum."""
import numpy as np
import time
from scipy.optimize import fsolve, minimize

N = 50
DEADLINE_FRAC = 0.92


def _radius(centers):
    """Compute the valid radius for a set of centers in [-0.5,0.5]^2."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min()
    wall = np.minimum(0.5 - np.abs(centers), axis=1)
    min_wall = wall.min()
    return min(min_pair / 2.0, min_wall)


def _radius_fast(centers):
    """Faster radius computation."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min()
    wall_x = np.minimum(0.5 - centers[:, 0], centers[:, 0])
    wall_y = np.minimum(0.5 - centers[:, 1], centers[:, 1])
    min_wall = np.minimum(wall_x.min(), wall_y.min())
    return min(min_pair / 2.0, min_wall)


def _contact_graph(centers, r, tol=1e-4):
    """Find pairs of circles that are touching (distance ≈ 2r) and
    circles touching walls (distance to wall ≈ r)."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Pairs touching: |dist - 2r| < tol
    pair_mask = np.abs(dist - 2 * r) < tol
    np.fill_diagonal(pair_mask, False)
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N) if pair_mask[i, j]]
    
    # Wall contacts
    wall_contacts = []
    for i in range(N):
        if abs(centers[i, 0] - 0.5 + r) < tol:
            wall_contacts.append((i, 'x_pos'))
        if abs(centers[i, 0] + 0.5 + r) < tol:
            wall_contacts.append((i, 'x_neg'))
        if abs(centers[i, 1] - 0.5 + r) < tol:
            wall_contacts.append((i, 'y_pos'))
        if abs(centers[i, 1] + 0.5 + r) < tol:
            wall_contacts.append((i, 'y_neg'))
    
    return pairs, wall_contacts


def _solve_active_set(centers, r, pairs, wall_contacts, max_iter=50, tol=1e-12):
    """Solve the system of equations for a given contact graph.
    Unknowns: 2*N center coordinates + 1 radius = 2N+1 unknowns.
    Equations: one per touching pair + one per wall contact.
    We solve by iterating: fix the contact graph, solve for centers+r."""
    n_unknowns = 2 * N + 1
    n_eqs = len(pairs) + len(wall_contacts)
    
    # We need at least as many equations as unknowns for a determined system.
    # If n_eqs < n_unknowns, we can't uniquely determine; use least squares or
    # add constraints. For now, if underdetermined, skip.
    if n_eqs < n_unknowns:
        return centers, r, False
    
    # Build the system: F(x) = 0 where x = [c_0, c_1, ..., c_{N-1}, r]
    # For each pair (i,j): ||c_i - c_j||^2 - (2r)^2 = 0
    # For each wall contact (i, wall): c_i,wall - (±0.5) + r = 0 (sign depends on wall)
    
    def equations(x):
        c = x[:2 * N].reshape(N, 2)
        rr = x[2 * N]
        eqs = []
        for i, j in pairs:
            eqs.append(np.sum((c[i] - c[j]) ** 2) - 4 * rr ** 2)
        for i, wall in wall_contacts:
            if wall == 'x_pos':
                eqs.append(c[i, 0] - 0.5 + rr)
            elif wall == 'x_neg':
                eqs.append(c[i, 0] + 0.5 + rr)
            elif wall == 'y_pos':
                eqs.append(c[i, 1] - 0.5 + rr)
            elif wall == 'y_neg':
                eqs.append(c[i, 1] + 0.5 + rr)
        return np.array(eqs[:n_unknowns])  # Take first n_unknowns equations
    
    x0 = np.concatenate([centers.flatten(), [r]])
    
    try:
        sol, info, ier, msg = fsolve(equations, x0, full_output=True, maxfev=10000)
        if ier != 1:
            return centers, r, False
        
        new_centers = sol[:2 * N].reshape(N, 2)
        new_r = sol[2 * N]
        
        # Check validity
        if new_r <= 0 or not np.all(np.isfinite(new_centers)):
            return centers, r, False
        if np.any(np.abs(new_centers) > 0.5 + 1e-6):
            return centers, r, False
        
        # Verify the solution is valid (all constraints satisfied)
        actual_r = _radius_fast(new_centers)
        if actual_r < new_r - 1e-8:
            return centers, r, False
        
        return new_centers, new_r, True
    except Exception:
        return centers, r, False


def _active_set_optimize(centers, r, deadline, max_rounds=200):
    """Iteratively refine the contact graph and solve the active set system."""
    centers = centers.copy()
    best_centers = centers.copy()
    best_r = r
    tol = 1e-4
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Find current contact graph
        pairs, wall_contacts = _contact_graph(centers, best_r, tol)
        
        if len(pairs) + len(wall_contacts) < 2 * N + 1:
            # Underdetermined; try a slightly different approach
            # Use a penalty-based optimization to maximize r while keeping
            # the current contacts tight
            def objective(x):
                c = x[:2 * N].reshape(N, 2)
                rr = x[2 * N]
                # Penalty for violating constraints
                penalty = 0.0
                for i, j in pairs:
                    d = np.sum((c[i] - c[j]) ** 2)
                    if d < 4 * rr ** 2:
                        penalty += (4 * rr ** 2 - d) ** 2
                for i, wall in wall_contacts:
                    if wall == 'x_pos':
                        v = c[i, 0] - 0.5 + rr
                        penalty += v ** 2
                    elif wall == 'x_neg':
                        v = c[i, 0] + 0.5 + rr
                        penalty += v ** 2
                    elif wall == 'y_pos':
                        v = c[i, 1] - 0.5 + rr
                        penalty += v ** 2
                    elif wall == 'y_neg':
                        v = c[i, 1] + 0.5 + rr
                        penalty += v ** 2
                return -rr + 1e4 * penalty
            
            x0 = np.concatenate([centers.flatten(), [best_r]])
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.x is not None and np.all(np.isfinite(res.x)):
                    new_centers = res.x[:2 * N].reshape(N, 2)
                    new_r = res.x[2 * N]
                    if new_r > 0 and np.all(np.abs(new_centers) <= 0.5 + 1e-6):
                        actual_r = _radius_fast(new_centers)
                        if actual_r > best_r + 1e-10:
                            best_r = actual_r
                            best_centers = new_centers.copy()
                            centers = new_centers.copy()
            except Exception:
                pass
            continue
        
        # Solve the active set system
        new_centers, new_r, success = _solve_active_set(centers, best_r, pairs, wall_contacts)
        
        if success and new_r > best_r + 1e-12:
            best_r = new_r
            best_centers = new_centers.copy()
            centers = new_centers.copy()
        elif success:
            # No improvement; try perturbing slightly to escape local optimum
            idx = np.random.randint(N)
            centers[idx] += (np.random.rand(2) - 0.5) * 1e-4
            centers = np.clip(centers, -0.499, 0.499)
        else:
            # Failed to solve; try a small perturbation
            idx = np.random.randint(N)
            centers[idx] += (np.random.rand(2) - 0.5) * 1e-3
            centers = np.clip(centers, -0.499, 0.499)
    
    return best_centers, best_r


def _initial_guess():
    """Create a good initial guess: hexagonal packing."""
    # 7 rows, 8 columns with hexagonal offset
    n_rows = 7
    n_cols = 8
    pts = []
    for row in range(n_rows):
        for col in range(n_cols):
            x = (col + 0.5) / n_cols - 0.5
            if row % 2 == 1:
                x += 0.5 / n_cols
            y = (row + 0.5) / n_rows - 0.5
            if abs(x) < 0.5 and abs(y) < 0.5:
                pts.append([x, y])
    
    pts = np.array(pts[:N])
    if len(pts) < N:
        extra = (np.random.rand(N - len(pts), 2) - 0.5) * 0.8
        pts = np.vstack([pts, extra])
    return pts[:N].astype(np.float64)


def _grid_init():
    """Square grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].astype(np.float64)


def _random_init():
    """Random initialization inside the square."""
    return (np.random.rand(N, 2) - 0.5) * 0.95


def _force_relax(centers, max_iter=3000, dt=0.001, deadline=None):
    """Simple force-directed relaxation to spread out circles."""
    centers = centers.copy()
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, 1.0)
        
        # Repulsive force
        inv_d3 = 1.0 / (dist ** 3 + 1e-10)
        np.fill_diagonal(inv_d3, 0.0)
        force = np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)
        
        # Wall repulsion
        for k in range(2):
            wall_pos = centers[:, k] - 0.45
            wall_neg = centers[:, k] + 0.45
            force[:, k] -= 10.0 * np.maximum(wall_pos, 0) ** 2
            force[:, k] += 10.0 * np.maximum(wall_neg, 0) ** 2
        
        # Weak centering
        force -= 0.005 * centers
        
        centers += dt * force
        centers[:, 0] = np.clip(centers[:, 0], -0.495, 0.495)
        centers[:, 1] = np.clip(centers[:, 1], -0.495, 0.495)
    return centers


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 * DEADLINE_FRAC
    
    np.random.seed(42)
    
    best_centers = None
    best_r = 0.0
    
    # Multiple initializations
    inits = [
        _initial_guess(),
        _grid_init(),
        _random_init(),
        _random_init(),
    ]
    
    for init in inits:
        if time.time() > deadline:
            break
        
        # Force relaxation to spread out
        relaxed = _force_relax(init, max_iter=2000, dt=0.0005, deadline=deadline)
        
        # Compute initial radius
        r = _radius_fast(relaxed)
        
        # Active set optimization
        opt_centers, opt_r = _active_set_optimize(relaxed, r, deadline, max_rounds=100)
        
        if opt_r > best_r:
            best_r = opt_r
            best_centers = opt_centers.copy()
    
    # Final local polish: coordinate descent on the best solution
    if best_centers is not None and time.time() < deadline:
        current = best_centers.copy()
        current_r = best_r
        for polish_iter in range(50):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                for k in range(2):
                    for step in [1e-4, -1e-4, 5e-5, -5e-5, 2e-5, -2e-5]:
                        new_c = current.copy()
                        new_c[i, k] += step
                        new_c[i, k] = np.clip(new_c[i, k], -0.4999, 0.4999)
                        r_new = _radius_fast(new_c)
                        if r_new > current_r + 1e-12:
                            current = new_c
                            current_r = r_new
                            improved = True
                            break
            if not improved:
                break
        
        if current_r > best_r:
            best_r = current_r
            best_centers = current
    
    # Ensure validity
    if best_centers is None:
        best_centers = _grid_init()
    
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    best_centers = np.nan_to_num(best_centers, nan=0.0, posinf=0.0, neginf=0.0)
    
    return best_centers
