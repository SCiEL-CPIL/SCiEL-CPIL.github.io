"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 26
TIME_LIMIT = 145.0  # safety margin from 150s

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-8):
    """Check if circles are valid (inside unit square, no overlaps)."""
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
    return True

def _score(pos, radii):
    """Return sum of radii if valid, else -inf."""
    if _is_valid(pos, radii):
        return np.sum(radii)
    return -np.inf

def _post_process(pos, radii, tol=1e-9):
    """Fix small violations by shrinking radii."""
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(30):
        valid = True
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r + tol:
                radii[i] = max(max_r * 0.99999, 1e-6)
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r + tol:
                    k = max((dist - tol) / (sum_r + 1e-12), 0.99)
                    radii[i] = max(radii[i] * k, 1e-6)
                    radii[j] = max(radii[j] * k, 1e-6)
                    valid = False
        if valid:
            break
    return pos, radii

def _max_radius_at(pos, radii, i):
    """Find maximum radius for circle i given all other circles fixed."""
    x, y = pos[i]
    max_r = min(x, 1.0 - x, y, 1.0 - y)
    for j in range(N):
        if i == j:
            continue
        dx = x - pos[j, 0]
        dy = y - pos[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        possible_r = dist - radii[j]
        if possible_r < max_r:
            max_r = possible_r
    return max(0.0, max_r)

def _greedy_init(n_circles=N):
    """Greedy best-fit initialization: place circles one by one at best position."""
    pos = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    
    # First circle: center
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, n_circles):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        rng = np.random.RandomState(i * 7919 + 13)
        
        # Sample candidate positions
        for _ in range(20000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                possible_r = dist - radii[j]
                if possible_r < max_r:
                    max_r = possible_r
            if max_r > best_r:
                best_r = max_r
                best_x = x
                best_y = y
        
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.999
        if best_r < 0.002:
            # Fallback: place with tiny radius
            pos[i] = [rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)]
            radii[i] = 0.002
    
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _hexagonal_init():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Try different hexagonal spacings
    best_pos, best_radii, best_score = None, None, -np.inf
    
    for r in [0.10, 0.11, 0.12, 0.13, 0.14]:
        dx = 2 * r * np.sqrt(3) / 2
        dy = 2 * r * 1.5
        k = 0
        p = np.zeros((N, 2))
        rr = np.zeros(N)
        for row in range(6):
            offset = dx / 2 if row % 2 == 1 else 0
            for col in range(6):
                if k >= N:
                    break
                x = r + offset + col * dx
                y = r + row * dy
                if x <= 1 - r and y <= 1 - r:
                    p[k] = [x, y]
                    rr[k] = r
                    k += 1
        while k < N:
            p[k] = [0.5, 0.5]
            rr[k] = 0.01
            k += 1
        p, rr = _post_process(p, rr)
        s = _score(p, rr)
        if s > best_score:
            best_score = s
            best_pos = p
            best_radii = rr
    
    return best_pos, best_radii

def _random_init(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.05, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _force_directed(pos, radii, n_steps=3000, dt=0.005, rng=None):
    """Force-directed relaxation with adaptive parameters."""
    if rng is None:
        rng = np.random.RandomState(0)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    
    for step in range(n_steps):
        progress = step / n_steps
        current_dt = dt * (1.0 - 0.8 * progress)
        forces = np.zeros_like(pos)
        
        # Repulsion between overlapping circles
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy) + 1e-12
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    overlap = sum_r - dist
                    f_mag = overlap * (1.0 + 3.0 * overlap)
                    fx = (dx / dist) * f_mag
                    fy = (dy / dist) * f_mag
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Wall repulsion
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            if x - r < 0:
                forces[i, 0] += (r - x) * 2.0
            if x + r > 1:
                forces[i, 0] -= (x + r - 1) * 2.0
            if y - r < 0:
                forces[i, 1] += (r - y) * 2.0
            if y + r > 1:
                forces[i, 1] -= (y + r - 1) * 2.0
        
        pos += forces * current_dt
        pos[:, 0] = np.clip(pos[:, 0], 0, 1)
        pos[:, 1] = np.clip(pos[:, 1], 0, 1)
        
        # Grow radii toward maximum
        if step > n_steps * 0.3:
            for i in range(N):
                max_r = _max_radius_at(pos, radii, i)
                if max_r > radii[i] * 1.001:
                    radii[i] = min(radii[i] * 1.002, max_r * 0.999)
                elif max_r < radii[i] * 0.998:
                    radii[i] = max_r * 0.998
    
    pos, radii = _post_process(pos, radii, tol=1e-9)
    return pos, radii

def _slsqp_optimize(pos, radii, maxiter=300):
    """SLSQP optimization with hard constraints."""
    def objective(x):
        return -np.sum(x[2 * N:])
    
    def constraints_func(x):
        pos_x = x[:2 * N].reshape(N, 2)
        r_x = x[2 * N:]
        cons = np.concatenate([
            pos_x[:, 0] - r_x,
            1.0 - pos_x[:, 0] - r_x,
            pos_x[:, 1] - r_x,
            1.0 - pos_x[:, 1] - r_x
        ])
        dx = pos_x[:, 0, np.newaxis] - pos_x[np.newaxis, :, 0]
        dy = pos_x[:, 1, np.newaxis] - pos_x[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        sum_r = r_x[:, np.newaxis] + r_x[np.newaxis, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        cons = np.concatenate([cons, (dist_sq - sum_r * sum_r)[mask]])
        return cons
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-6, 0.5))
    
    x0 = np.concatenate([pos.flatten(), radii])
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          constraints={'type': 'ineq', 'fun': constraints_func},
                          options={'maxiter': maxiter, 'ftol': 1e-10})
        if result.success or result.fun < objective(x0):
            pos_new = result.x[:2 * N].reshape(N, 2)
            radii_new = result.x[2 * N:]
            pos_pp, radii_pp = _post_process(pos_new, radii_new)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass
    return pos, radii

def _penalty_optimize(pos, radii, beta=50.0, maxiter=200):
    """Smooth penalty optimization."""
    def objective(x):
        pos_x = x[:2 * N].reshape(N, 2)
        r_x = x[2 * N:]
        b_cons = np.concatenate([
            pos_x[:, 0] - r_x,
            1.0 - pos_x[:, 0] - r_x,
            pos_x[:, 1] - r_x,
            1.0 - pos_x[:, 1] - r_x
        ])
        dx = pos_x[:, 0, np.newaxis] - pos_x[np.newaxis, :, 0]
        dy = pos_x[:, 1, np.newaxis] - pos_x[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        sum_r = r_x[:, np.newaxis] + r_x[np.newaxis, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        o_cons = (dist_sq - sum_r * sum_r)[mask]
        all_cons = np.concatenate([b_cons, o_cons])
        z = -beta * all_cons
        penalty = np.sum(np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))) / beta
        return -np.sum(r_x) + penalty
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-6, 0.5))
    
    x0 = np.concatenate([pos.flatten(), radii])
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-8})
        pos_new = result.x[:2 * N].reshape(N, 2)
        radii_new = result.x[2 * N:]
        pos_pp, radii_pp = _post_process(pos_new, radii_new)
        if _is_valid(pos_pp, radii_pp):
            return pos_pp, radii_pp
    except Exception:
        pass
    return pos, radii

def _coordinate_descent(pos, radii, n_rounds=5):
    """Coordinate descent: for each circle, find optimal position and radius."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(n_rounds):
        improved = False
        for i in range(N):
            # Try to expand radius
            old_r = radii[i]
            max_r = _max_radius_at(pos, radii, i)
            if max_r > old_r * 1.001:
                radii[i] = max_r * 0.999
                improved = True
            
            # Try small position moves
            best_pos_i = pos[i].copy()
            best_r_i = radii[i]
            rng = np.random.RandomState(i * 100 + 42)
            for _ in range(500):
                dx = rng.uniform(-0.03, 0.03)
                dy = rng.uniform(-0.03, 0.03)
                new_x = np.clip(pos[i, 0] + dx, 0.01, 0.99)
                new_y = np.clip(pos[i, 1] + dy, 0.01, 0.99)
                # Check if valid
                valid = True
                for j in range(N):
                    if i == j:
                        continue
                    ddx = new_x - pos[j, 0]
                    ddy = new_y - pos[j, 1]
                    dist = np.sqrt(ddx * ddx + ddy * ddy)
                    if dist < best_r_i + radii[j] + 1e-9:
                        valid = False
                        break
                if valid:
                    # Can we expand?
                    max_r_new = min(new_x, 1.0 - new_x, new_y, 1.0 - new_y)
                    for j in range(N):
                        if i == j:
                            continue
                        ddx = new_x - pos[j, 0]
                        ddy = new_y - pos[j, 1]
                        dist = np.sqrt(ddx * ddx + ddy * ddy)
                        possible_r = dist - radii[j]
                        if possible_r < max_r_new:
                            max_r_new = possible_r
                    if max_r_new > best_r_i:
                        best_r_i = max_r_new * 0.999
                        best_pos_i = [new_x, new_y]
                        improved = True
        
        pos[i] = best_pos_i
        radii[i] = best_r_i
        if not improved:
            break
    
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _simulated_annealing(pos, radii, n_steps=5000, t_start=0.05, t_end=0.001, rng=None):
    """Simulated annealing with position and radius perturbations."""
    if rng is None:
        rng = np.random.RandomState(42)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = _score(pos, radii)
    current_score = best_score
    
    for step in range(n_steps):
        t = t_start * (t_end / t_start) ** (step / n_steps)
        
        # Perturb
        new_pos = pos.copy()
        new_radii = radii.copy()
        i = rng.randint(N)
        new_pos[i, 0] += rng.normal(0, t * 2)
        new_pos[i, 1] += rng.normal(0, t * 2)
        new_pos[i, 0] = np.clip(new_pos[i, 0], 0.01, 0.99)
        new_pos[i, 1] = np.clip(new_pos[i, 1], 0.01, 0.99)
        new_radii[i] = max(1e-5, new_radii[i] * (1 + rng.normal(0, 0.05)))
        
        # Try to expand all radii
        for j in range(N):
            max_r = _max_radius_at(new_pos, new_radii, j)
            if max_r > new_radii[j] * 1.001:
                new_radii[j] = min(new_radii[j] * 1.005, max_r * 0.999)
        
        new_pos, new_radii = _post_process(new_pos, new_radii)
        new_score = _score(new_pos, new_radii)
        
        if new_score > current_score:
            pos, radii = new_pos, new_radii
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pos = pos.copy()
                best_radii = radii.copy()
        elif new_score > -np.inf and rng.random() < np.exp((new_score - current_score) / t):
            pos, radii = new_pos, new_radii
            current_score = new_score
    
    return best_pos, best_radii

def _final_polish(pos, radii, time_budget=10.0):
    """Final high-precision polish."""
    start_t = time.time()
    
    # SLSQP with tight bounds
    bounds = []
    for i in range(N):
        bounds.append((max(0.0, pos[i, 0] - 0.02), min(1.0, pos[i, 0] + 0.02)))
        bounds.append((max(0.0, pos[i, 1] - 0.02), min(1.0, pos[i, 1] + 0.02)))
    for i in range(N):
        bounds.append((max(1e-6, radii[i] * 0.995), min(0.5, radii[i] * 1.005)))
    
    def objective(x):
        return -np.sum(x[2 * N:])
    
    def constraints_func(x):
        pos_x = x[:2 * N].reshape(N, 2)
        r_x = x[2 * N:]
        cons = np.concatenate([
            pos_x[:, 0] - r_x,
            1.0 - pos_x[:, 0] - r_x,
            pos_x[:, 1] - r_x,
            1.0 - pos_x[:, 1] - r_x
        ])
        dx = pos_x[:, 0, np.newaxis] - pos_x[np.newaxis, :, 0]
        dy = pos_x[:, 1, np.newaxis] - pos_x[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        sum_r = r_x[:, np.newaxis] + r_x[np.newaxis, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        cons = np.concatenate([cons, (dist_sq - sum_r * sum_r)[mask]])
        return cons
    
    x0 = np.concatenate([pos.flatten(), radii])
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          constraints={'type': 'ineq', 'fun': constraints_func},
                          options={'maxiter': 100, 'ftol': 1e-12})
        if result.success or result.fun < objective(x0):
            pos_new = result.x[:2 * N].reshape(N, 2)
            radii_new = result.x[2 * N:]
            pos_pp, radii_pp = _post_process(pos_new, radii_new, tol=1e-10)
            if _is_valid(pos_pp, radii_pp, tol=1e-9):
                pos, radii = pos_pp, radii_pp
    except Exception:
        pass
    
    # Coordinate descent on radii only
    if time.time() - start_t < time_budget * 0.5:
        for _ in range(3):
            improved = False
            for i in range(N):
                max_r = _max_radius_at(pos, radii, i)
                if max_r > radii[i] + 1e-10:
                    radii[i] = max_r * 0.999999
                    improved = True
            if not improved:
                break
    
    # Final SLSQP
    if time.time() - start_t < time_budget * 0.7:
        bounds = []
        for i in range(N):
            bounds.append((max(0.0, pos[i, 0] - 0.005), min(1.0, pos[i, 0] + 0.005)))
            bounds.append((max(0.0, pos[i, 1] - 0.005), min(1.0, pos[i, 1] + 0.005)))
        for i in range(N):
            bounds.append((max(1e-6, radii[i] * 0.999), min(0.5, radii[i] * 1.001)))
        
        x0 = np.concatenate([pos.flatten(), radii])
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints_func},
                              options={'maxiter': 50, 'ftol': 1e-14})
            if result.success or result.fun < objective(x0):
                pos_new = result.x[:2 * N].reshape(N, 2)
                radii_new = result.x[2 * N:]
                pos_pp, radii_pp = _post_process(pos_new, radii_new, tol=1e-11)
                if _is_valid(pos_pp, radii_pp, tol=1e-10):
                    pos, radii = pos_pp, radii_pp
        except Exception:
            pass
    
    pos, radii = _post_process(pos, radii, tol=1e-10)
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    def update_best(pos, radii):
        nonlocal best_pos, best_radii, best_score
        s = _score(pos, radii)
        if s > best_score:
            best_score = s
            best_pos = pos.copy()
            best_radii = radii.copy()
    
    # Phase 1: Multiple initializations
    initializers = [
        _greedy_init,
        _hexagonal_init,
        lambda: _random_init(1),
        lambda: _random_init(2),
        lambda: _random_init(3),
        lambda: _random_init(4),
        lambda: _random_init(5),
        lambda: _random_init(6),
        lambda: _random_init(7),
        lambda: _random_init(8),
        lambda: _random_init(9),
        lambda: _random_init(10),
    ]
    
    for idx, init_func in enumerate(initializers):
        if time.time() > deadline - 30:
            break
        try:
            pos_init, rad_init = init_func()
        except Exception:
            continue
        
        update_best(pos_init, rad_init)
        
        # Force-directed relaxation
        if time.time() < deadline - 15:
            try:
                pos_fd, radii_fd = _force_directed(pos_init, rad_init, n_steps=2000, dt=0.005,
                                                    rng=np.random.RandomState(idx * 13 + 7))
                update_best(pos_fd, radii_fd)
            except Exception:
                pass
        
        # SLSQP
        if time.time() < deadline - 10:
            try:
                pos_s, radii_s = _slsqp_optimize(pos_fd if 'pos_fd' in dir() else pos_init,
                                                  radii_fd if 'radii_fd' in dir() else rad_init, maxiter=200)
                update_best(pos_s, radii_s)
            except Exception:
                pass
    
    # Phase 2: Multi-start local optimization from best
    if best_pos is not None:
        # Penalty optimization with increasing beta
        if time.time() < deadline - 20:
            for beta in [30.0, 80.0, 150.0]:
                if time.time() > deadline - 15:
                    break
                try:
                    pos_p, radii_p = _penalty_optimize(best_pos, best_radii, beta=beta, maxiter=150)
                    update_best(pos_p, radii_p)
                except Exception:
                    pass
        
        # Coordinate descent
        if time.time() < deadline - 10:
            try:
                pos_cd, radii_cd = _coordinate_descent(best_pos, best_radii, n_rounds=3)
                update_best(pos_cd, radii_cd)
            except Exception:
                pass
        
        # Simulated annealing
        if time.time() < deadline - 10:
            try:
                pos_sa, radii_sa = _simulated_annealing(best_pos, best_radii, n_steps=3000,
                                                         t_start=0.03, t_end=0.001,
                                                         rng=np.random.RandomState(123))
                update_best(pos_sa, radii_sa)
            except Exception:
                pass
        
        # More SLSQP from different starting points
        if time.time() < deadline - 8:
            for seed in [42, 123, 456]:
                if time.time() > deadline - 5:
                    break
                rng = np.random.RandomState(seed)
                pos_pert = best_pos + rng.normal(0, 0.01, (N, 2))
                pos_pert[:, 0] = np.clip(pos_pert[:, 0], 0.01, 0.99)
                pos_pert[:, 1] = np.clip(pos_pert[:, 1], 0.01, 0.99)
                pos_pert, radii_pert = _post_process(pos_pert, best_radii)
                if _is_valid(pos_pert, radii_pert):
                    try:
                        pos_s, radii_s = _slsqp_optimize(pos_pert, radii_pert, maxiter=150)
                        update_best(pos_s, radii_s)
                    except Exception:
                        pass
    
    # Phase 3: Final polish
    if best_pos is not None and time.time() < deadline - 5:
        try:
            pos_pol, radii_pol = _final_polish(best_pos, best_radii, time_budget=5.0)
            update_best(pos_pol, radii_pol)
        except Exception:
            pass
    
    # Fallback
    if best_pos is None or not _is_valid(best_pos, best_radii):
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N - 1:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = _post_process(pos, radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
