"""Circle packing in perimeter-4 rectangle: Voronoi Inradius + L-BFGS-B + Force Relaxation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
_TIME_LIMIT = 145.0  # Safety margin before 150s kill


def _compute_voronoi_radii(x, y, w, h):
    """
    Compute the inradius for each circle center (x_i, y_i) within the rectangle [0,w]x[0,h].
    The radius is the minimum distance to:
    1. Rectangle boundaries
    2. Other circle centers (distance / 2)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    
    # Wall distances
    r_wall = np.minimum(np.minimum(x, w - x), np.minimum(y, h - y))
    
    # Neighbor distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, np.inf)
    
    r_neighbor = np.min(d, axis=1) / 2.0
    
    r = np.minimum(r_wall, r_neighbor)
    return r


def _objective(z):
    """
    Minimize -sum(r_i).
    """
    n = N
    x = z[0:n]
    y = z[n:2*n]
    w = z[2*n]
    h = 2.0 - w
    
    if w < 0.3 or h < 0.3:
        return 1e6
    
    r = _compute_voronoi_radii(x, y, w, h)
    return -np.sum(r)


def _objective_grad(z):
    """
    Numerical gradient for the non-smooth objective.
    """
    eps = 1e-6
    g = np.zeros_like(z)
    f0 = _objective(z)
    for i in range(len(z)):
        z_plus = z.copy()
        z_plus[i] += eps
        f_plus = _objective(z_plus)
        g[i] = (f_plus - f0) / eps
    return g


def _force_relaxation(x, y, w, h, iterations=200):
    """
    Simple force-directed relaxation using gradient descent on the objective.
    """
    n = N
    z = np.concatenate([x, y, [w]])
    
    for _ in range(iterations):
        grad = _objective_grad(z)
        # We want to minimize _objective, so move opposite to gradient
        lr = 0.01
        z_new = z - lr * grad
        
        # Clip w
        z_new[-1] = np.clip(z_new[-1], 0.3, 1.7)
        
        # Clip x, y to be within [0, w] and [0, h]
        x_new = z_new[0:n]
        y_new = z_new[n:2*n]
        w_new = z_new[-1]
        h_new = 2.0 - w_new
        
        x_new = np.clip(x_new, 0.0, w_new)
        y_new = np.clip(y_new, 0.0, h_new)
        
        z_new[0:n] = x_new
        z_new[n:2*n] = y_new
        z = z_new
        
    return z


def _polish_with_l_bfgs(z0, max_time=10.0):
    """
    Run L-BFGS-B to polish the configuration.
    """
    start = time.time()
    
    bounds = []
    for i in range(2 * N):
        bounds.append((0.0, 2.0))
    bounds.append((0.3, 1.7))
    
    try:
        res = minimize(
            _objective,
            z0,
            method='L-BFGS-B',
            jac=_objective_grad,
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return res.x
    except Exception:
        return z0


def _generate_initial_guesses():
    """Generate diverse initial configurations."""
    guesses = []
    rng = np.random.default_rng(42)
    
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.55, 1.8, 0.5]
    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        cols = max(1, int(np.ceil(np.sqrt(N * w0 / h0))))
        rows = max(1, int(np.ceil(N / cols)))
        z = np.zeros(2 * N + 1)
        for i in range(N):
            row, col = divmod(i, cols)
            cx = (col + 0.5) * (w0 / cols)
            cy = (row + 0.5) * (h0 / rows)
            z[i] = cx
            z[N + i] = cy
        z[-1] = w0
        guesses.append(z)
        
    for seed in range(10):
        rng2 = np.random.default_rng(seed)
        w0 = rng2.uniform(0.6, 1.4)
        h0 = 2.0 - w0
        z = np.zeros(2 * N + 1)
        for i in range(N):
            z[i] = rng2.uniform(0.0, w0)
            z[N + i] = rng2.uniform(0.0, h0)
        z[-1] = w0
        guesses.append(z)
        
    # Hexagonal packing approximation
    w0 = 1.0
    h0 = 1.0
    z = np.zeros(2 * N + 1)
    spacing = 0.25
    x_coords = []
    y_coords = []
    row = 0
    while len(x_coords) < N:
        for col in range(int(w0 / spacing) + 1):
            x = col * spacing
            y = row * spacing * np.sqrt(3) / 2
            if row % 2 == 1:
                x += spacing / 2
            if x < w0 and y < h0:
                x_coords.append(x)
                y_coords.append(y)
        row += 1
    indices = rng.permutation(len(x_coords))[:N]
    for i in range(N):
        z[i] = x_coords[indices[i]]
        z[N + i] = y_coords[indices[i]]
    z[-1] = w0
    guesses.append(z)
    
    return guesses


def _extract_result(z):
    """Extract x, y, r from z."""
    x = z[0:N]
    y = z[N:2*N]
    w = z[-1]
    h = 2.0 - w
    r = _compute_voronoi_radii(x, y, w, h)
    r = np.maximum(r, 1e-6)
    x = np.clip(x, r, w - r)
    y = np.clip(y, r, h - r)
    r = _compute_voronoi_radii(x, y, w, h)
    r = np.maximum(r, 1e-6)
    return x, y, r, w, h


def _is_valid(x, y, r, w, h, tol=1e-8):
    """Check validity."""
    if w <= 0.05 or h <= 0.05:
        return False
    for i in range(N):
        if r[i] <= 0:
            return False
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum * r_sum - tol:
                return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT

    best_value = 0.0
    best_result = None
    best_z = None

    guesses = _generate_initial_guesses()

    for idx, z0 in enumerate(guesses):
        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed
        if remaining < 10:
            break

        x0 = z0[0:N]
        y0 = z0[N:2*N]
        w0 = z0[-1]
        h0 = 2.0 - w0
        
        # 1. Force Relaxation
        z_relaxed = _force_relaxation(x0, y0, w0, h0, iterations=100)
        x_r, y_r, r_r, w_r, h_r = _extract_result(z_relaxed)
        valid = _is_valid(x_r, y_r, r_r, w_r, h_r)
        val = np.sum(r_r) if valid else 0.0
        
        if valid and val > best_value:
            best_value = val
            best_result = np.column_stack((x_r, y_r, r_r))
            best_z = z_relaxed.copy()

        # 2. L-BFGS Polish
        if remaining > 5:
            z_polished = _polish_with_l_bfgs(z_relaxed, max_time=min(remaining * 0.4, 15.0))
            x_p, y_p, r_p, w_p, h_p = _extract_result(z_polished)
            valid = _is_valid(x_p, y_p, r_p, w_p, h_p)
            val = np.sum(r_p) if valid else 0.0
            
            if valid and val > best_value:
                best_value = val
                best_result = np.column_stack((x_p, y_p, r_p))
                best_z = z_polished.copy()

    # Final polish on the best result found
    if best_z is not None and time.time() < deadline - 5:
        z_final = _polish_with_l_bfgs(best_z, max_time=min(deadline - time.time() - 2, 10.0))
        x_f, y_f, r_f, w_f, h_f = _extract_result(z_final)
        valid = _is_valid(x_f, y_f, r_f, w_f, h_f)
        val = np.sum(r_f) if valid else 0.0
        
        if valid and val > best_value:
            best_value = val
            best_result = np.column_stack((x_f, y_f, r_f))

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Safety: tiny shrink to ensure strict validity
    x = best_result[:, 0].copy()
    y = best_result[:, 1].copy()
    r = best_result[:, 2].copy()
    
    # We need w and h for validity check. 
    # Reconstruct w from best_z if available, else estimate
    if best_z is not None:
        w = best_z[-1]
        h = 2.0 - w
    else:
        w = 1.0
        h = 1.0

    for _ in range(30):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h, tol=1e-10):
            r = r_trial
        else:
            break
    best_result[:, 2] = r

    return best_result
