"""Morse cluster global minima finder - Hybrid structural templates + multi-start optimization.

Strategy:
1. Generate diverse structural templates (FCC, HCP, BCC, icosahedral, random) matched to rho regime.
2. Multi-start L-BFGS-B relaxation with perturbation-based basin hopping.
3. Global time management to ensure all 6 instances finish within 300s.
"""
import numpy as np
import time
from scipy.optimize import minimize

# Global time tracking across all 6 instances
_GLOBAL_START_TIME = None
_GLOBAL_BUDGET = 290.0  # Leave 10s safety margin for 300s limit


def _get_global_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    elapsed = time.time() - _GLOBAL_START_TIME
    return max(0.0, _GLOBAL_BUDGET - elapsed)


def _morse_energy(pos, rho):
    """Compute Morse potential energy: sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij))-2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    x = rho * (1.0 - dists)
    x = np.clip(x, -50, 50)
    ex = np.exp(x)
    energy = np.sum(ex * (ex - 2.0)) / 2.0
    return float(energy)


def _morse_gradient(pos, rho):
    """Compute gradient of Morse energy (fully vectorized)."""
    n = len(pos)
    grad = np.zeros_like(pos)
    if n < 2:
        return grad
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    x = rho * (1.0 - dists)
    x = np.clip(x, -50, 50)
    exp_val = np.exp(x)
    pair_force = -2.0 * rho * exp_val * (exp_val - 1.0)
    np.fill_diagonal(pair_force, 0.0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        W = pair_force / dists
    W = np.where(dists > 1e-10, W, 0.0)
    Ws = W.sum(axis=1)
    grad = Ws[:, None] * pos - W @ pos
    return grad


def _min_dist_check(pos, min_d=0.5):
    """Check if all pairs are at least min_d apart."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_d - 1e-9


def _ensure_min_dist(pos, min_d=0.51, max_iter=200):
    """Separate atoms that are too close using a simple repulsion scheme."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        if not _min_dist_check(pos, min_d):
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=-1))
            np.fill_diagonal(dists, np.inf)
            idx_i, idx_j = np.unravel_index(np.argmin(dists), dists.shape)
            if idx_i > idx_j:
                idx_i, idx_j = idx_j, idx_i
            rij = pos[idx_i] - pos[idx_j]
            r = np.linalg.norm(rij)
            if r < 1e-10:
                rij = np.array([1.0, 0, 0])
                r = 1.0
            push = (min_d - r + 0.02) / 2.0
            pos[idx_i] += push * rij / r
            pos[idx_j] -= push * rij / r
        else:
            break
    return pos


def _fcc_points(n, spacing=1.0):
    """Generate FCC lattice points."""
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
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()


def _hcp_points(n, spacing=1.0):
    """Generate HCP lattice points."""
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
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()


def _bcc_points(n, spacing=1.0):
    """Generate BCC lattice points."""
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
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()


def _mckay_icosahedron(n, spacing=1.0):
    """Generate Mackay icosahedron cluster."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    pts = [np.zeros(3)]
    
    for v in verts:
        pts.append(v.copy())
    
    faces = [
        [0, 4, 8], [0, 9, 4], [0, 1, 5], [0, 5, 10],
        [0, 10, 9], [0, 8, 7], [0, 7, 11], [0, 11, 9],
        [1, 6, 8], [1, 8, 4], [1, 2, 3], [1, 3, 5],
        [2, 6, 7], [2, 7, 3], [2, 3, 11], [2, 11, 6],
        [3, 7, 11], [3, 11, 5], [3, 5, 2], [4, 9, 8]
    ]
    for f in faces:
        center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        r = np.linalg.norm(center)
        if r > 1e-10:
            center = center / r * 1.15
            pts.append(center)
    
    if n > len(pts):
        verts2 = verts * 2.0
        for v in verts2:
            pts.append(v.copy())
        for f in faces:
            center = (verts2[f[0]] + verts2[f[1]] + verts2[f[2]]) / 3.0
            pts.append(center)
            
    while len(pts) < n:
        p = pts[np.random.randint(len(pts))] + np.random.normal(0, 0.1, 3)
        pts.append(p)
        
    pts = np.array(pts[:n])
    pts -= pts.mean(axis=0)
    if len(pts) > 1:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        if min_d > 1e-10:
            pts = pts * (spacing / min_d)
            
    return pts


def _icosahedral_shell(n, rng=None):
    """Generate icosahedral shell configuration with multiple shells."""
    if rng is None:
        rng = np.random.default_rng()
    
    phi = (1 + np.sqrt(5)) / 2
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico_vertices = ico_vertices / np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    
    # Generate more directions by rotating
    all_dirs = [ico_vertices.copy()]
    for angle in [np.pi / 5, 2 * np.pi / 5, np.pi / 3, 2 * np.pi / 3]:
        for axis in [[0, 0, 1], [0, 1, 0], [1, 0, 0]]:
            axis = np.array(axis, dtype=float)
            axis = axis / np.linalg.norm(axis)
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            rotated = ico_vertices @ R.T
            all_dirs.append(rotated)
    
    all_dirs = np.vstack(all_dirs)
    all_dirs = all_dirs / np.linalg.norm(all_dirs, axis=1, keepdims=True)
    
    # Determine shell radii based on n
    if n <= 13:
        shell_radii = [0, 1.0]
    elif n <= 55:
        shell_radii = [0, 1.0, 1.7]
    elif n <= 147:
        shell_radii = [0, 1.0, 1.7, 2.4]
    else:
        shell_radii = [0, 1.0, 1.7, 2.4, 3.1]
    
    points = [np.zeros(3)]
    
    for shell_idx, radius in enumerate(shell_radii):
        if radius == 0:
            continue
        n_shell = min(30, n - len(points))
        for i in range(n_shell):
            if len(points) >= n:
                break
            d = all_dirs[i % len(all_dirs)]
            d = d + rng.normal(0, 0.02, 3)
            d = d / np.linalg.norm(d)
            points.append(d * radius)
    
    while len(points) < n:
        d = all_dirs[len(points) % len(all_dirs)] + rng.normal(0, 0.05, 3)
        d = d / np.linalg.norm(d)
        r = shell_radii[-1] + rng.uniform(0, 0.3)
        points.append(d * r)
    
    pts = np.array(points[:n])
    pts -= pts.mean(axis=0)
    return pts


def _random_dense(n, rng, spacing=1.0):
    """Generate random dense packing."""
    pts = []
    radius = spacing * (n ** (1/3)) * 0.85
    min_d = 0.55 * spacing
    max_attempts = 5000
    attempts = 0
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
    
    # If not enough, relax constraint
    if len(pts) < n:
        min_d = 0.51 * spacing
        attempts = 0
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
    
    # Final fallback
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
        
    return np.array(pts)


def _relax_lbfgs(pos, rho, maxiter=1000):
    """Relax using L-BFGS-B."""
    n = len(pos)
    x0 = pos.flatten()

    def objective(x):
        p = x.reshape(n, 3)
        return _morse_energy(p, rho)

    def gradient(x):
        p = x.reshape(n, 3)
        return _morse_gradient(p, rho).flatten()

    try:
        result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'gtol': 1e-10, 'ftol': 1e-12, 'maxls': 50})
        if result.fun < _morse_energy(pos, rho):
            return result.x.reshape(n, 3), result.fun
    except Exception:
        pass

    return pos, _morse_energy(pos, rho)


def construct(instance):
    """Find low-energy Morse cluster configuration."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    remaining = _get_global_remaining_time()
    if remaining <= 0:
        # Fallback: return a simple configuration
        return _random_dense(size, np.random.default_rng(42))
    
    # Per-instance deadline with safety margin
    per_instance_budget = min(remaining, 45.0)
    start_time = time.time()
    deadline = start_time + per_instance_budget
    
    rng = np.random.default_rng(size * 13 + int(rho) * 7 + 42)
    
    best_energy = np.inf
    best_pos = None
    
    # Generate initial configurations based on rho
    initial_configs = []
    
    if rho <= 6:
        # Long/medium range: FCC, HCP, BCC favored
        initial_configs.append(_fcc_points(size))
        initial_configs.append(_hcp_points(size))
        initial_configs.append(_bcc_points(size))
        initial_configs.append(_random_dense(size, rng))
        initial_configs.append(_random_dense(size, rng))
    else:
        # Short range: Icosahedral, FCC favored
        initial_configs.append(_mckay_icosahedron(size))
        initial_configs.append(_icosahedral_shell(size, rng))
        initial_configs.append(_fcc_points(size))
        initial_configs.append(_hcp_points(size))
        initial_configs.append(_random_dense(size, rng))
        initial_configs.append(_random_dense(size, rng))
    
    # Add perturbed versions of the first few templates
    for base in initial_configs[:3]:
        perturbed = base + rng.normal(0, 0.15, base.shape)
        initial_configs.append(perturbed)
    
    # Phase 1: Relax each initial configuration
    for pos in initial_configs:
        if time.time() > deadline - 5.0:
            break
            
        pos = _ensure_min_dist(pos)
        if not _min_dist_check(pos):
            continue
            
        try:
            relaxed_pos, energy = _relax_lbfgs(pos, rho, maxiter=800)
            if _min_dist_check(relaxed_pos):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos.copy()
        except Exception:
            continue
            
    # Phase 2: Basin hopping from best configuration
    if best_pos is not None:
        while time.time() < deadline - 10.0:
            kick_size = 0.25
            kicked_pos = best_pos + rng.normal(0, kick_size, best_pos.shape)
            kicked_pos = _ensure_min_dist(kicked_pos)
            if not _min_dist_check(kicked_pos):
                continue
                
            try:
                relaxed_pos, energy = _relax_lbfgs(kicked_pos, rho, maxiter=500)
                if _min_dist_check(relaxed_pos):
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed_pos.copy()
            except Exception:
                continue
            
    else:
        # Fallback if no valid configuration found
        pos = _random_dense(size, rng)
        pos = _ensure_min_dist(pos)
        if _min_dist_check(pos):
            best_pos, best_energy = _relax_lbfgs(pos, rho, maxiter=1000)
            
    # Phase 3: Final polish with more iterations
    if best_pos is not None and time.time() < deadline - 2.0:
        try:
            relaxed_pos, energy = _relax_lbfgs(best_pos, rho, maxiter=2000)
            if _min_dist_check(relaxed_pos) and energy < best_energy:
                best_pos = relaxed_pos
                best_energy = energy
        except Exception:
            pass
            
    if best_pos is None:
        best_pos = _random_dense(size, rng)
        best_pos = _ensure_min_dist(best_pos)
        
    return best_pos
