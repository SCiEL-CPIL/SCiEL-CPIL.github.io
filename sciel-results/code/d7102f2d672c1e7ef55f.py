"""Morse cluster global minima search with structured lattice seeds and robust local optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_TIME_LIMIT = 290.0  # safety margin before 330s kill


def _remaining_time():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _morse_energy(pos, rho):
    """Compute Morse potential energy efficiently."""
    n = len(pos)
    if n < 2:
        return 0.0
    d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(d ** 2, axis=2))
    np.fill_diagonal(r, 1e10)
    u = rho * (1.0 - r)
    e = np.exp(u)
    energy = np.sum(e * (e - 2.0))
    return 0.5 * energy


def _morse_gradient(pos, rho):
    """Compute gradient of Morse energy (fully vectorized)."""
    n = len(pos)
    grad = np.zeros_like(pos)
    if n < 2:
        return grad
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, 1e10)
    
    exp_val = np.exp(rho * (1.0 - dists))
    # d/dr [exp(rho(1-r)) * (exp(rho(1-r)) - 2)] = -2*rho*exp(rho(1-r))*(exp(rho(1-r))-1)
    pair_force = -2.0 * rho * exp_val * (exp_val - 1.0)
    np.fill_diagonal(pair_force, 0.0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        W = pair_force / dists
    W = np.where(dists > 1e-10, W, 0.0)
    Ws = W.sum(axis=1)
    grad = Ws[:, None] * pos - W @ pos
    return grad


def _fix_min_dist(pos, min_d=0.501, max_iter=200):
    """Separate any atoms closer than min_d using iterative push."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < min_d:
                    if dist > 1e-10:
                        push = (min_d - dist) / dist * d
                        pos[i] += 0.5 * push
                        pos[j] -= 0.5 * push
                    else:
                        # Random separation
                        pos[i] += np.array([0.1, 0.1, 0.1])
                        pos[j] -= np.array([0.1, 0.1, 0.1])
                    moved = True
        if not moved:
            break
    return pos


def _fcc_points(n, spacing=1.0):
    """Generate FCC lattice points, sorted by distance from origin."""
    pts = []
    a = spacing
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    pts.append(p)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()


def _hcp_points(n, spacing=1.0):
    """Generate HCP lattice points, sorted by distance from origin."""
    pts = []
    a = spacing
    c_over_a = np.sqrt(8.0 / 3.0)
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k * c_over_a]) * a
                p2 = np.array([i + 1/3.0, j + 2/3.0, (k + 0.5) * c_over_a]) * a
                pts.append(p1)
                pts.append(p2)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()


def _bcc_points(n, spacing=1.0):
    """Generate BCC lattice points, sorted by distance from origin."""
    pts = []
    a = spacing
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k]) * a
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                pts.append(p1)
                pts.append(p2)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()


def _random_dense(n, rng, radius=None, min_d=0.55):
    """Generate random dense packing."""
    if radius is None:
        radius = n ** (1.0 / 3.0) * 0.8
    pts = []
    attempts = 0
    max_attempts = n * 100
    while len(pts) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in pts:
                if np.linalg.norm(p - q) < min_d:
                    valid = False
                    break
            if valid:
                pts.append(p)
        attempts += 1
    
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
        
    return np.array(pts[:n])


def _optimize(pos, rho, time_budget=15.0):
    """Locally optimize a configuration using L-BFGS-B."""
    start = time.time()
    pos = _fix_min_dist(pos)
    
    def objective(x):
        p = x.reshape(-1, 3)
        return _morse_energy(p, rho)
    
    def gradient(x):
        p = x.reshape(-1, 3)
        try:
            g = _morse_gradient(p, rho)
            return g.flatten()
        except:
            # Numerical gradient fallback
            eps = 1e-6
            g = np.zeros_like(x)
            for i in range(len(x)):
                xp = x.copy()
                xm = x.copy()
                xp[i] += eps
                xm[i] -= eps
                g[i] = (objective(xp) - objective(xm)) / (2 * eps)
            return g
    
    try:
        result = minimize(
            objective,
            pos.flatten(),
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 300, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        
        elapsed = time.time() - start
        if result.success or (time.time() - start) < time_budget:
            return result.fun, result.x.reshape(-1, 3), elapsed
        else:
            return result.fun, result.x.reshape(-1, 3), elapsed
    except Exception:
        return _morse_energy(pos, rho), pos, time.time() - start


def construct(instance):
    """Multi-start local optimization for Morse cluster global minima."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 3))
    
    rng = np.random.default_rng(42)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting configurations
    starts = []
    
    # 1. FCC lattice
    starts.append(_fcc_points(size, spacing=1.0))
    
    # 2. HCP lattice
    starts.append(_hcp_points(size, spacing=1.0))
    
    # 3. BCC lattice
    starts.append(_bcc_points(size, spacing=1.0))
    
    # 4. Random dense starts with different radii
    for i in range(4):
        radius = size ** (1.0 / 3.0) * (0.6 + 0.15 * i)
        starts.append(_random_dense(size, rng, radius=radius))
    
    # 5. Perturbed FCC
    fcc_base = _fcc_points(size, spacing=1.0)
    for i in range(2):
        perturbed = fcc_base + rng.normal(0, 0.1, (size, 3))
        starts.append(perturbed)
    
    # Optimize each start, tracking time
    for start_pos in starts:
        if _remaining_time() < 5.0:
            break
        
        # Allocate time budget based on remaining time
        time_budget = min(20.0, _remaining_time() / max(1, len(starts)))
        
        try:
            energy, positions, elapsed = _optimize(start_pos, rho, time_budget)
            positions = _fix_min_dist(positions)
            
            # Verify energy and min distance
            min_d = np.inf
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < min_d:
                        min_d = d
            
            if min_d >= 0.5 - 1e-9 and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception as e:
            pass
    
    # If we still have time, do a few more random restarts
    while _remaining_time() > 10.0:
        start_pos = _random_dense(size, rng, radius=size ** (1.0 / 3.0) * 0.7)
        try:
            energy, positions, elapsed = _optimize(start_pos, rho, time_budget=10.0)
            positions = _fix_min_dist(positions)
            
            min_d = np.inf
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < min_d:
                        min_d = d
            
            if min_d >= 0.5 - 1e-9 and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception as e:
            pass
    
    if best_positions is None:
        # Fallback: return the first start
        best_positions = _fix_min_dist(starts[0])
    
    return best_positions
# EVOLVE-BLOCK-END
