"""Morse cluster global minima finder with global time management."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time tracking
_GLOBAL_START_TIME = None
_GLOBAL_BUDGET = 290.0  # Leave 10s safety margin for 300s limit

def _get_global_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    elapsed = time.time() - _GLOBAL_START_TIME
    return max(0.0, _GLOBAL_BUDGET - elapsed)

def _morse_energy(pos, rho):
    """Compute Morse energy for a configuration."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    upper = np.triu_indices(n, 1)
    r = dists[upper]
    val = rho * (1.0 - r)
    exp_val = np.exp(val)
    e = np.sum(exp_val * (exp_val - 2.0))
    return float(e)

def _morse_energy_flat(x, n, rho):
    """Compute Morse energy from flat array."""
    pos = x.reshape(n, 3)
    return _morse_energy(pos, rho)

def _morse_gradient_flat(x, n, rho):
    """Compute Morse gradient from flat array using morse_energy_gradient."""
    pos = x.reshape(n, 3)
    grad = morse_energy_gradient(pos, rho)
    return grad.ravel()

def _min_dist_check(pos, min_d=0.5):
    """Check minimum pairwise distance."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_d - 1e-9

def _fcc_cluster(n, spacing=1.0):
    """Generate an FCC-like cluster of n atoms."""
    pts = []
    a = spacing
    max_r = int(np.ceil(n ** (1/3))) + 2
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    pts.append(p)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]

def _hcp_cluster(n, spacing=1.0):
    """Generate an HCP-like cluster of n atoms."""
    pts = []
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    max_r = int(np.ceil(n ** (1/3))) + 2
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k]) * a
                p2 = np.array([i + 1/3, j + 2/3, k + 0.5]) * a
                p1[2] *= c / a
                p2[2] *= c / a
                pts.append(p1)
                pts.append(p2)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]

def _icosahedron_shell(n, spacing=1.0):
    """Generate icosahedral shell cluster."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * spacing
    
    pts = [np.zeros(3)]
    if n > 1:
        pts.extend([v.copy() for v in verts])
    if n > 13:
        faces = [
            [0, 4, 8], [0, 9, 4], [0, 1, 5], [0, 5, 10],
            [0, 10, 9], [0, 8, 7], [0, 7, 11], [0, 11, 9],
            [1, 6, 8], [1, 8, 4], [1, 2, 3], [1, 3, 5],
            [2, 6, 7], [2, 7, 3], [2, 3, 11], [2, 11, 6],
            [3, 7, 11], [3, 11, 5], [3, 5, 2], [4, 9, 8]
        ]
        for f in faces:
            center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
            center = center / np.linalg.norm(center) * spacing * 0.87
            pts.append(center)
    if n > 33:
        edges = set()
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if d < 1.5 * spacing:
                    mid = (verts[i] + verts[j]) / 2.0
                    mid = mid / np.linalg.norm(mid) * spacing * 0.76
                    pts.append(mid)
    
    pts = np.array(pts[:n])
    return pts

def _random_packed(n, rng, spacing=1.0):
    """Random packed cluster with minimum distance constraint."""
    pts = []
    radius = spacing * n ** (1/3) * 1.2
    min_d = 0.55 * spacing
    attempts = 0
    max_attempts = 50000
    while len(pts) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_d for q in pts):
                pts.append(p)
        attempts += 1
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts)

def _perturb_cluster(base, n, rng, scale=0.3):
    """Perturb a base cluster and resize to n atoms."""
    if len(base) >= n:
        centroid = base.mean(axis=0)
        dists = np.linalg.norm(base - centroid, axis=1)
        idx = np.argsort(dists)[:n]
        pts = base[idx].copy()
    else:
        pts = base.copy()
        centroid = pts.mean(axis=0)
        while len(pts) < n:
            i = rng.integers(len(pts))
            new_p = pts[i] + rng.normal(0, scale, 3)
            pts = np.vstack([pts, new_p])
    
    pts = pts + rng.normal(0, 0.05, pts.shape)
    return pts

def construct(instance):
    """Find low-energy Morse cluster configuration."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    # Get remaining global time
    remaining_time = _get_global_remaining_time()
    if remaining_time <= 5.0:
        # If we are out of time, return a quick valid guess
        try:
            pos = _fcc_cluster(size, spacing=1.0)
            if _min_dist_check(pos):
                return pos
        except:
            pass
        # Fallback
        pos = np.zeros((size, 3))
        for i in range(size):
            pos[i] = [i * 0.6, 0, 0]
        return pos

    # Allocate a portion of the remaining time to this instance
    # Assuming 6 instances total, we aim for ~45s each to fit in 270s total.
    instance_budget = min(45.0, remaining_time * 0.8)
    
    start_time = time.time()
    deadline = start_time + instance_budget
    
    rng = np.random.default_rng(size * 7 + 42)
    
    best_energy = np.inf
    best_pos = None
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. FCC-based
    try:
        fcc = _fcc_cluster(size, spacing=1.0)
        initial_configs.append(('fcc', fcc))
    except Exception:
        pass
    
    # 2. HCP-based
    try:
        hcp = _hcp_cluster(size, spacing=1.0)
        initial_configs.append(('hcp', hcp))
    except Exception:
        pass
    
    # 3. Icosahedral
    try:
        ico = _icosahedron_shell(size, spacing=1.0)
        initial_configs.append(('ico', ico))
    except Exception:
        pass
    
    # 4. Random packed (multiple variants)
    for i in range(3):
        try:
            rp = _random_packed(size, rng, spacing=1.0)
            initial_configs.append((f'rand{i}', rp))
        except Exception:
            pass
    
    # 5. Perturbed FCC
    try:
        fcc_base = _fcc_cluster(max(size, 20), spacing=1.0)
        pert = _perturb_cluster(fcc_base, size, rng, scale=0.4)
        initial_configs.append(('fcc_pert', pert))
    except Exception:
        pass
    
    # 6. Perturbed random
    try:
        rand_base = _random_packed(max(size, 15), rng, spacing=1.0)
        pert2 = _perturb_cluster(rand_base, size, rng, scale=0.5)
        initial_configs.append(('rand_pert', pert2))
    except Exception:
        pass
    
    # Additional random restarts for larger clusters
    if size >= 50:
        for i in range(2):
            try:
                rp = _random_packed(size, rng, spacing=1.0)
                initial_configs.append((f'rand_extra{i}', rp))
            except Exception:
                pass
    
    def _lbfgs_relax(pos, rho_val, time_limit=None):
        """Use L-BFGS-B with vectorized gradient for fast, accurate relaxation."""
        n = len(pos)
        x0 = pos.ravel()
        
        def func(x):
            return _morse_energy_flat(x, n, rho_val)
        
        def grad(x):
            return _morse_gradient_flat(x, n, rho_val)
        
        t0 = time.time()
        options = {
            'maxiter': 2000,
            'ftol': 1e-12,
            'gtol': 1e-8,
            'maxls': 20,
            'maxcor': 10
        }
        
        if time_limit is not None:
            # Check time before starting
            if time.time() - t0 > time_limit:
                return None, None
        
        try:
            result = minimize(
                func, x0, jac=grad, method='L-BFGS-B',
                options=options
            )
            if result.success or result.fun < np.inf:
                relaxed = result.x.reshape(n, 3)
                if _min_dist_check(relaxed):
                    return result.fun, relaxed
        except Exception:
            pass
        return None, None
    
    # Relax each initial configuration using L-BFGS-B
    for name, pos in initial_configs:
        if time.time() > deadline - 5:
            break
        try:
            if not _min_dist_check(pos):
                pos = pos + rng.normal(0, 0.1, pos.shape)
                if not _min_dist_check(pos):
                    continue
            
            energy, relaxed_pos = _lbfgs_relax(pos, rho, time_limit=30)
            
            if relaxed_pos is not None and energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos.copy()
        except Exception:
            continue
    
    # If we have time, do additional random restarts
    if best_pos is not None and time.time() < deadline - 10:
        time_per_restart = 3.0 if size < 50 else 8.0
        n_extra = int((deadline - time.time()) / time_per_restart)
        n_extra = max(0, min(15, n_extra))
        
        for i in range(n_extra):
            if time.time() > deadline - 5:
                break
            try:
                rp = _random_packed(size, rng, spacing=1.0)
                if not _min_dist_check(rp):
                    rp = rp + rng.normal(0, 0.15, rp.shape)
                energy, relaxed_pos = _lbfgs_relax(rp, rho, time_limit=25)
                if relaxed_pos is not None and energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos.copy()
            except Exception:
                continue
    
    # Final high-precision polish pass on the best configuration
    if best_pos is not None and time.time() < deadline - 3:
        try:
            n = len(best_pos)
            x0 = best_pos.ravel()
            
            def func(x):
                return _morse_energy_flat(x, n, rho)
            
            def grad(x):
                return _morse_gradient_flat(x, n, rho)
            
            result = minimize(
                func, x0, jac=grad, method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30, 'maxcor': 15}
            )
            if result.fun < best_energy:
                polished = result.x.reshape(n, 3)
                if _min_dist_check(polished):
                    best_energy = result.fun
                    best_pos = polished
        except Exception:
            pass
    
    # Final fallback
    if best_pos is None:
        spacing = 0.6
        best_pos = np.zeros((size, 3))
        for i in range(size):
            best_pos[i] = [i * spacing, 0, 0]
        if not _min_dist_check(best_pos):
            radius = spacing * size ** (1/3) * 1.5
            best_pos = rng.uniform(-radius, radius, (size, 3))
            valid = [best_pos[0]]
            for i in range(1, size):
                attempts = 0
                while attempts < 1000:
                    p = best_pos[i]
                    if all(np.linalg.norm(p - v) >= 0.5 for v in valid):
                        valid.append(p)
                        break
                    p = rng.uniform(-radius, radius, 3)
                    best_pos[i] = p
                    attempts += 1
                else:
                    valid.append(best_pos[i])
            best_pos = np.array(valid[:size])
    
    return best_pos
# EVOLVE-BLOCK-END
