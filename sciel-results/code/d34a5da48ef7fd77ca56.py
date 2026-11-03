"""Circle packing in perimeter-4 rectangle.
Pack 21 circles with disjoint interiors, maximize sum of radii.
Perimeter constraint: 2*(w+h) <= 4  =>  w+h <= 2.
"""
N = 21
import time
import numpy as np
from scipy.optimize import minimize

# Safety margin before 120s deadline
DEADLINE = 115.0


def _validate(w, h, x, y, r):
    """Check all constraints; return True if valid."""
    # Perimeter constraint
    if 2.0 * (w + h) > 4.0 + 1e-7:
        return False
    # Wall constraints
    for i in range(N):
        if x[i] < r[i] - 1e-7 or x[i] > w - r[i] + 1e-7:
            return False
        if y[i] < r[i] - 1e-7 or y[i] > h - r[i] + 1e-7:
            return False
    # Pairwise separation
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            min_d = r[i] + r[j]
            if dist_sq < min_d * min_d - 1e-7:
                return False
    return True


def _penalty_objective(z, w, h, lam=50.0):
    """Penalty-based objective for N circles in rectangle [0,w]x[0,h].
    z = [x_0..x_{N-1}, y_0..y_{N-1}, r_0..r_{N-1}]
    """
    x = z[:N]
    y = z[N:2*N]
    r = z[2*N:3*N]

    # Objective: maximize sum(r) => minimize -sum(r) + penalties
    obj = -np.sum(r)

    # Wall penalty
    for i in range(N):
        # x in [r_i, w - r_i]
        if x[i] < r[i]:
            obj += lam * (r[i] - x[i]) ** 2
        if x[i] > w - r[i]:
            obj += lam * (x[i] - (w - r[i])) ** 2
        # y in [r_i, h - r_i]
        if y[i] < r[i]:
            obj += lam * (r[i] - y[i]) ** 2
        if y[i] > h - r[i]:
            obj += lam * (y[i] - (h - r[i])) ** 2

    # Pairwise separation penalty
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_d = r[i] + r[j]
            if dist < min_d:
                obj += lam * (min_d - dist) ** 2

    return obj


def _make_initial(w, h, seed):
    """Create a random initial configuration for N circles in [0,w]x[0,h]."""
    rng = np.random.default_rng(seed)
    # Place circles on a rough grid with jitter
    cols = int(np.ceil(np.sqrt(N * w / h)))
    rows = int(np.ceil(N / cols))
    dx = w / cols
    dy = h / rows

    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    r_max = 0.45 / max(w, h) * min(w, h)  # rough max radius

    for i in range(N):
        ci = i % cols
        ri = i // cols
        if ri >= rows:
            ri = rows - 1
        x[i] = (ci + 0.5) * dx + rng.uniform(-0.2 * dx, 0.2 * dx)
        y[i] = (ri + 0.5) * dy + rng.uniform(-0.2 * dy, 0.2 * dy)
        r[i] = 0.35 * min(dx, dy) * 0.5

    # Clamp to be inside
    for i in range(N):
        x[i] = np.clip(x[i], r[i], w - r[i])
        y[i] = np.clip(y[i], r[i], h - r[i])

    z = np.concatenate([x, y, r])
    return z


def _optimize_single(w, h, seed):
    """Run one optimization for fixed aspect ratio."""
    z0 = _make_initial(w, h, seed)
    
    # Bounds: x in [0, w], y in [0, h], r in [0, min(w,h)/2]
    r_max = 0.45 * min(w, h)
    bounds = [(0, w)] * N + [(0, h)] * N + [(0, r_max)] * N
    
    best_obj = _penalty_objective(z0, w, h)
    best_z = z0.copy()
    
    # Try multiple lambda values for annealing
    for lam in [50.0, 100.0, 200.0]:
        res = minimize(_penalty_objective, best_z, args=(w, h, lam),
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000})
        if res.fun < best_obj:
            best_obj = res.fun
            best_z = res.x.copy()
    
    x = best_z[:N]
    y = best_z[N:2*N]
    r = best_z[2*N:3*N]
    
    if _validate(w, h, x, y, r):
        return np.sum(r), x, y, r
    return None


def construct():
    start_time = time.time()
    
    best_score = -np.inf
    best_result = None
    
    # Grid of aspect ratios to try
    # w + h <= 2, so w in (0, 2), h = 2 - w
    # Try a range of w values
    w_values = np.linspace(0.5, 1.5, 11)  # 11 aspect ratios
    
    seed_counter = 0
    
    for w in w_values:
        if time.time() - start_time > DEADLINE:
            break
        h = 2.0 - w
        
        # Try a few random seeds per aspect ratio
        n_seeds = 3
        for s in range(n_seeds):
            if time.time() - start_time > DEADLINE:
                break
            seed_counter += 1
            result = _optimize_single(w, h, seed_counter)
            if result is not None:
                score, x, y, r = result
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([x, y, r])
    
    # If no valid result found, fall back to a simple grid
    if best_result is None:
        # Simple grid: 5x5 in a 1x1 square (perimeter 4)
        side = 5
        r = 0.5 / side - 1e-4
        x = np.zeros(N)
        y = np.zeros(N)
        rr = np.zeros(N)
        for i in range(N):
            ci = i % side
            ri = i // side
            x[i] = (ci + 0.5) / side
            y[i] = (ri + 0.5) / side
            rr[i] = r
        best_result = np.column_stack([x, y, rr])
    
    return best_result
