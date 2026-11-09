import time
import numpy as np
from scipy.optimize import minimize

N = 21
_DEADLINE = time.time() + 140.0  # safety margin under 150s kill


def _validate(arr):
    if arr is None or arr.shape != (N, 3):
        return None
    if not np.all(np.isfinite(arr)):
        return None
    w, h, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r < 0):
        return None
    if np.any(w < 0) or np.any(w > 1) or np.any(h < 0) or np.any(h > 1):
        return None
    # non-overlap: dist >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(w[i] - w[j], h[i] - h[j])
            if d < r[i] + r[j] - 1e-7:
                return None
    return arr


def _rect_dims(a):
    # perimeter 4 => 2(w+h)=4 => w+h=2 ; aspect a = w/h
    h = 2.0 / (a + 1.0)
    w = 2.0 - h
    return w, h


def _force_relax(x0, w, h, n_iter=500, dt=0.01, repulsion=1.0, attraction=0.5):
    """Force-directed relaxation to find a good local packing."""
    n = N
    pos = x0[:2 * n].copy().reshape(n, 2)
    radii = x0[2 * n:].copy()
    
    # Initial radii from a simple estimate
    min_dim = min(w, h)
    max_r = min_dim / (2 * np.ceil(np.sqrt(N)))
    radii[:] = max_r * 0.8
    
    for it in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        forces = np.zeros_like(pos)
        
        # Repulsive forces between circles
        for i in range(n):
            for j in range(i + 1, n):
                diff = pos[i] - pos[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    diff = np.array([0.1, 0.1])
                    dist = np.linalg.norm(diff)
                
                overlap = radii[i] + radii[j] - dist
                if overlap > 0:
                    force_mag = repulsion * overlap / dist
                    forces[i] += diff * force_mag
                    forces[j] -= diff * force_mag
        
        # Attractive forces to expand radii (indirectly via position spread)
        # Boundary forces
        for i in range(n):
            # Left wall
            if pos[i, 0] < radii[i]:
                forces[i, 0] += attraction * (radii[i] - pos[i, 0])
            # Right wall
            if pos[i, 0] > w - radii[i]:
                forces[i, 0] -= attraction * (pos[i, 0] - (w - radii[i]))
            # Bottom wall
            if pos[i, 1] < radii[i]:
                forces[i, 1] += attraction * (radii[i] - pos[i, 1])
            # Top wall
            if pos[i, 1] > h - radii[i]:
                forces[i, 1] -= attraction * (pos[i, 1] - (h - radii[i]))
        
        # Update positions with damping
        pos += dt * forces
        # Clamp to boundaries
        pos[:, 0] = np.clip(pos[:, 0], 0, w)
        pos[:, 1] = np.clip(pos[:, 1], 0, h)
        
        # Gradually increase radii if no overlaps
        min_overlap = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(pos[i] - pos[j])
                overlap = radii[i] + radii[j] - dist
                if overlap < min_overlap:
                    min_overlap = overlap
        
        if min_overlap < 0:
            # Can increase radii slightly
            growth = min(0.001, -min_overlap * 0.5)
            radii += growth
    
    # Build result
    result = np.empty(3 * n)
    result[:2 * n] = pos.flatten()
    result[2 * n:] = radii
    return result


def _slsqp_solve(x0, w, h, maxiter=300):
    """SLSQP optimization with tight constraints."""
    n = N
    
    def objective(x):
        return -np.sum(x[2 * n:])
    
    def jac(x):
        g = np.zeros(3 * n)
        g[2 * n:] = -1.0
        return g
    
    def con_overlap(x):
        pos = x[:2 * n].reshape(n, 2)
        r = x[2 * n:]
        vals = np.empty(n * (n - 1) // 2)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                d2 = (pos[i, 0] - pos[j, 0]) ** 2 + (pos[i, 1] - pos[j, 1]) ** 2
                s = r[i] + r[j]
                vals[idx] = d2 - s * s
                idx += 1
        return vals
    
    def con_left(x):
        return x[:2 * n:2] - x[2 * n:]
    
    def con_right(x):
        return (w - x[:2 * n:2]) - x[2 * n:]
    
    def con_bottom(x):
        return x[1:2 * n:2] - x[2 * n:]
    
    def con_top(x):
        return (h - x[1:2 * n:2]) - x[2 * n:]
    
    cons = [
        {"type": "ineq", "fun": con_overlap},
        {"type": "ineq", "fun": con_left},
        {"type": "ineq", "fun": con_right},
        {"type": "ineq", "fun": con_bottom},
        {"type": "ineq", "fun": con_top},
    ]
    bounds = [(0.0, w)] * n + [(0.0, h)] * n + [(1e-4, 1.0)] * n
    
    try:
        res = minimize(objective, x0, jac=jac, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": maxiter, "ftol": 1e-12})
        if res.fun < 0:
            arr = np.zeros((n, 3))
            arr[:, 0] = res.x[:2 * n:2]
            arr[:, 1] = res.x[1:2 * n:2]
            arr[:, 2] = res.x[2 * n:]
            v = _validate(arr)
            if v is not None:
                return v
    except Exception:
        pass
    return None


def _hexagonal_init(w, h, rng):
    """Generate hexagonal lattice initialization."""
    n = N
    # Estimate optimal radius
    area = w * h
    est_r = np.sqrt(area / (N * np.pi * 1.1))  # ~10% inefficiency
    
    # Hexagonal spacing
    dx = 2 * est_r
    dy = dx * np.sqrt(3) / 2
    
    pos = np.empty((n, 2))
    count = 0
    row = 0
    while count < n:
        col = 0
        while count < n:
            x = col * dx + (row % 2) * dx / 2 + est_r
            y = row * dy + est_r
            if x < w and y < h:
                pos[count, 0] = x
                pos[count, 1] = y
                count += 1
            col += 1
        row += 1
    
    # Add some randomness
    pos += rng.uniform(-est_r * 0.1, est_r * 0.1, size=pos.shape)
    pos[:, 0] = np.clip(pos[:, 0], est_r, w - est_r)
    pos[:, 1] = np.clip(pos[:, 1], est_r, h - est_r)
    
    x0 = np.empty(3 * n)
    x0[:2 * n] = pos.flatten()
    x0[2 * n:] = est_r
    return x0


def _solve_aspect_x0(a, x0, rng):
    w, h = _rect_dims(a)
    n = N
    
    def time_left_fn():
        return _DEADLINE - time.time()
    
    if time_left_fn() < 2:
        return None
    
    # Stage 1: Force-directed relaxation
    relaxed = _force_relax(x0, w, h, n_iter=300, dt=0.005, repulsion=1.0, attraction=0.3)
    
    # Stage 2: SLSQP polish
    result = _slsqp_solve(relaxed, w, h, maxiter=200)
    if result is not None:
        return result
    
    # If SLSQP failed, try with original x0
    result = _slsqp_solve(x0, w, h, maxiter=100)
    if result is not None:
        return result
    
    # Last resort: validate relaxed directly (may have slight violations)
    arr = np.zeros((n, 3))
    arr[:, 0] = relaxed[:2 * n:2]
    arr[:, 1] = relaxed[1:2 * n:2]
    arr[:, 2] = relaxed[2 * n:]
    v = _validate(arr)
    if v is not None:
        return v
    
    return None


def _seed_grid(a, rng):
    # a simple grid-like feasible-ish start for the aspect
    w, h = _rect_dims(a)
    cols = int(np.ceil(np.sqrt(N * w / h)))
    rows = int(np.ceil(N / cols))
    xs = np.linspace(0.1, 0.9, cols) * w
    ys = np.linspace(0.1, 0.9, rows) * h
    x0 = np.empty(3 * N)
    for k in range(N):
        i = k % cols
        j = k // cols
        x0[2 * k] = xs[i]
        x0[2 * k + 1] = ys[j]
        x0[2 * N + k] = 0.05
    return x0


def construct():
    t_start = time.time()
    rng = np.random.default_rng(12345)
    best = None
    best_val = -np.inf

    aspects = [1.0, 1.3, 0.77, 1.6, 0.62, 1.15, 0.87, 1.4, 0.7]
    starts_per_aspect = 2

    for a in aspects:
        for s in range(starts_per_aspect):
            if time.time() > _DEADLINE:
                break
            
            # Mix initialization strategies
            if s == 0:
                # Hexagonal lattice
                x0 = _hexagonal_init(*_rect_dims(a), rng)
            else:
                # Random
                w, h = _rect_dims(a)
                x0 = np.empty(3 * N)
                for k in range(N):
                    x0[2 * k] = rng.uniform(0.05, 0.95) * w
                    x0[2 * k + 1] = rng.uniform(0.05, 0.95) * h
                    x0[2 * N + k] = rng.uniform(0.02, 0.12)
            
            arr = _solve_aspect_x0(a, x0, rng)
            if arr is not None:
                val = np.sum(arr[:, 2])
                if val > best_val:
                    best_val = val
                    best = arr
        if time.time() > _DEADLINE:
            break

    if best is None:
        # fallback: seed-like uniform grid (always feasible)
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    return best
