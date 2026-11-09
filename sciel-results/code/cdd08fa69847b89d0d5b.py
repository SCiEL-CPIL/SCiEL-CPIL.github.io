"""Bootstrap spawn: direct approach for SC transition-metal cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _validate_positions(pos):
    """Check if positions are valid (finite, min distance > 0.35)."""
    if not np.all(np.isfinite(pos)):
        return False
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.35


def _generate_fcc_cluster(n, lattice=1.0):
    """Generate a compact FCC-like cluster."""
    fcc_points = []
    size = int(np.ceil(n ** (1.0 / 3.0))) + 2
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                fcc_points.append([i * lattice, j * lattice, k * lattice])
    offsets = [
        [0, 0, 0],
        [lattice / 2, lattice / 2, 0],
        [lattice / 2, 0, lattice / 2],
        [0, lattice / 2, lattice / 2],
    ]
    all_points = []
    for base in fcc_points:
        for off in offsets:
            all_points.append(np.array(base) + np.array(off))
    all_points = np.array(all_points)
    dists = np.linalg.norm(all_points, axis=1)
    max_r = 0.9 * n ** (1.0 / 3.0) + 1.0
    mask = dists <= max_r
    points = all_points[mask]
    if len(points) >= n:
        d = np.linalg.norm(points, axis=1)
        order = np.argsort(d)
        return points[order[:n]]
    rng = np.random.default_rng(42)
    pts = []
    while len(pts) < n:
        p = rng.uniform(-max_r, max_r, 3)
        pts.append(p)
    return np.array(pts)


def _generate_hcp_cluster(n, lattice=1.0):
    """Generate a compact HCP-like cluster."""
    a = lattice
    c = a * np.sqrt(8.0 / 3.0)
    points = []
    size = int(np.ceil(n ** (1.0 / 3.0))) + 2
    for k in range(-size, size + 1):
        z = k * c / 2
        layer = k % 2
        for i in range(-size, size + 1):
            for j in range(-size, size + 1):
                if layer == 0:
                    x = i * a + j * a / 2
                    y = j * a * np.sqrt(3) / 2
                else:
                    x = i * a + j * a / 2 + a / 2
                    y = j * a * np.sqrt(3) / 2 + a * np.sqrt(3) / 6
                points.append([x, y, z])
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    max_r = 0.9 * n ** (1.0 / 3.0) + 1.0
    mask = dists <= max_r
    pts = points[mask]
    if len(pts) >= n:
        d = np.linalg.norm(pts, axis=1)
        order = np.argsort(d)
        return pts[order[:n]]
    rng = np.random.default_rng(43)
    pts = []
    while len(pts) < n:
        p = rng.uniform(-max_r, max_r, 3)
        pts.append(p)
    return np.array(pts)


def _generate_bcc_cluster(n, lattice=1.0):
    """Generate a compact BCC-like cluster."""
    points = []
    size = int(np.ceil(n ** (1.0 / 3.0))) + 2
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                points.append([i * lattice, j * lattice, k * lattice])
                points.append([(i + 0.5) * lattice, (j + 0.5) * lattice, (k + 0.5) * lattice])
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    max_r = 0.9 * n ** (1.0 / 3.0) + 1.0
    mask = dists <= max_r
    pts = points[mask]
    if len(pts) >= n:
        d = np.linalg.norm(pts, axis=1)
        order = np.argsort(d)
        return pts[order[:n]]
    rng = np.random.default_rng(45)
    pts = []
    while len(pts) < n:
        p = rng.uniform(-max_r, max_r, 3)
        pts.append(p)
    return np.array(pts)


def _generate_icosahedral_cluster(n, scale=1.0):
    """Generate an icosahedral-like cluster with configurable scale."""
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0]) * scale
    
    points = [np.zeros(3)]
    for v in ico_verts:
        points.append(v)
    
    n_verts = len(ico_verts)
    shell2 = []
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            if d < 1.2 * scale:
                mid = (ico_verts[i] + ico_verts[j]) / 2
                norm = np.linalg.norm(mid)
                if norm > 1e-8:
                    mid = mid / norm * 1.35 * scale
                    shell2.append(mid)
    for p in shell2:
        points.append(p)
    
    if len(points) >= n:
        return np.array(points[:n])
    else:
        rng = np.random.default_rng(44)
        while len(points) < n:
            p = rng.normal(size=3)
            norm = np.linalg.norm(p)
            if norm > 1e-8:
                p = p / norm * 2.0 * scale
            points.append(p)
    return np.array(points[:n])


def _generate_random_cluster(n, seed, min_dist=0.8):
    """Generate a random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    max_radius = 0.85 * n ** (1.0 / 3.0)
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 500:
        p = rng.uniform(-max_radius, max_radius, 3)
        if np.linalg.norm(p) <= max_radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    ok = False
                    break
            if ok:
                points.append(p)
        attempts += 1
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-max_radius, max_radius, 3)
            points.append(p)
    return np.array(points)


def _try_relax(pos, potential, kind):
    """Try to relax a configuration, return (energy, positions) or (inf, None)."""
    try:
        if not _validate_positions(pos):
            return np.inf, None
        energy, relaxed = relax(pos, potential, kind)
        if np.isfinite(energy) and _validate_positions(relaxed):
            return energy, relaxed
    except Exception:
        pass
    return np.inf, None


def _soft_relax(pos, potential, kind, n_exp_soft, m_exp_soft, c_soft, time_limit=5.0):
    """
    Relax with a softened potential (lower exponents, scaled c) to escape sharp barriers.
    Uses scipy minimize directly with a custom energy/gradient function.
    """
    start = time.time()
    n = len(pos)
    pos = pos.copy()
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def soft_energy_gradient(x):
        if time.time() - start > time_limit:
            raise TimeoutError()
        x = x.reshape(n, 3)
        diff = x[i_idx] - x[j_idx]
        r2 = np.sum(diff * diff, axis=1)
        r2 = np.maximum(r2, 1e-10)
        
        r = np.sqrt(r2)
        
        dEdr = -n_exp_soft * r2 ** (-(n_exp_soft + 1) / 2.0) + \
               c_soft * m_exp_soft / 2.0 * r2 ** (-(m_exp_soft / 2.0 + 1) / 2.0)
        
        factor = dEdr / np.sqrt(r2)
        
        grad_pairs = factor[:, np.newaxis] * diff
        
        grad = np.zeros_like(x)
        np.add.at(grad, i_idx, grad_pairs)
        np.add.at(grad, j_idx, -grad_pairs)
        
        E = np.sum(r2 ** (-n_exp_soft / 2.0) - c_soft * r2 ** (-m_exp_soft / 4.0))
        
        return E, grad.reshape(-1)
    
    try:
        E0, g0 = soft_energy_gradient(pos.flatten())
        if not np.isfinite(E0):
            return np.inf, None
        
        res = minimize(
            fun=soft_energy_gradient,
            x0=pos.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 1000}
        )
        
        if res.success or np.isfinite(res.fun):
            relaxed = res.x.reshape(n, 3)
            if _validate_positions(relaxed):
                return res.fun, relaxed
    except (TimeoutError, Exception):
        pass
    return np.inf, None


def _anneal_relax(pos, potential, kind, n_exp_true, m_exp_true, c_true, time_limit=8.0):
    """
    Coarse-to-fine annealing: start with softened potential, ramp to true potential.
    """
    start = time.time()
    n = len(pos)
    pos = pos.copy()
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    # Define annealing schedule: 5 steps from soft to true
    steps = 5
    # Soft parameters
    n_exp_soft = max(4, int(n_exp_true * 0.5))
    m_exp_soft = max(4, int(m_exp_true * 0.5))
    c_soft = c_true * 0.4
    
    current_pos = pos
    for step in range(steps):
        if time.time() - start > time_limit:
            break
        
        # Linearly interpolate parameters
        t = (step + 1) / steps
        n_exp_cur = n_exp_soft + (n_exp_true - n_exp_soft) * t
        m_exp_cur = m_exp_soft + (m_exp_true - m_exp_soft) * t
        c_cur = c_soft + (c_true - c_soft) * t
        
        def anneal_energy_gradient(x, ne=n_exp_cur, me=m_exp_cur, cc=c_cur):
            x = x.reshape(n, 3)
            diff = x[i_idx] - x[j_idx]
            r2 = np.sum(diff * diff, axis=1)
            r2 = np.maximum(r2, 1e-10)
            
            r = np.sqrt(r2)
            
            dEdr = -ne * r2 ** (-(ne + 1) / 2.0) + \
                   cc * me / 2.0 * r2 ** (-(me / 2.0 + 1) / 2.0)
            
            factor = dEdr / np.sqrt(r2)
            
            grad_pairs = factor[:, np.newaxis] * diff
            
            grad = np.zeros_like(x)
            np.add.at(grad, i_idx, grad_pairs)
            np.add.at(grad, j_idx, -grad_pairs)
            
            E = np.sum(r2 ** (-ne / 2.0) - cc * r2 ** (-me / 4.0))
            
            return E, grad.reshape(-1)
        
        try:
            E0, g0 = anneal_energy_gradient(current_pos.flatten())
            if not np.isfinite(E0):
                break
            
            res = minimize(
                fun=anneal_energy_gradient,
                x0=current_pos.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 400}
            )
            
            if res.success or np.isfinite(res.fun):
                relaxed = res.x.reshape(n, 3)
                if _validate_positions(relaxed):
                    current_pos = relaxed
        except Exception:
            break
    
    return current_pos


def construct(instance):
    """
    Multi-start optimization with deadline guard for SC cluster global minima.
    Uses a coarse-to-fine annealing strategy.
    """
    start_time = time.time()
    deadline = start_time + 290.0
    
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    if kind == "12-6":
        n_exp_true, m_exp_true, c_true = 12, 6, 144.41
    elif kind == "9-6":
        n_exp_true, m_exp_true, c_true = 9, 6, 39.432
    elif kind == "10-8":
        n_exp_true, m_exp_true, c_true = 10, 8, 34.408
    else:
        n_exp_true, m_exp_true, c_true = 12, 6, 144.41
    
    # Generate diverse starting configurations
    candidates = []
    
    candidates.append(_generate_fcc_cluster(n, lattice=0.70))
    candidates.append(_generate_fcc_cluster(n, lattice=0.80))
    candidates.append(_generate_fcc_cluster(n, lattice=0.90))
    candidates.append(_generate_hcp_cluster(n, lattice=0.70))
    candidates.append(_generate_hcp_cluster(n, lattice=0.80))
    candidates.append(_generate_hcp_cluster(n, lattice=0.90))
    candidates.append(_generate_bcc_cluster(n, lattice=0.70))
    candidates.append(_generate_bcc_cluster(n, lattice=0.80))
    candidates.append(_generate_bcc_cluster(n, lattice=0.90))
    candidates.append(_generate_icosahedral_cluster(n, scale=0.9))
    candidates.append(_generate_icosahedral_cluster(n, scale=1.1))
    candidates.append(_generate_icosahedral_cluster(n, scale=1.3))
    
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    min_dists = [0.75, 0.85, 0.95, 1.05]
    
    for seed in seeds:
        for md in min_dists:
            if time.time() > deadline - 10:
                break
            pos = _generate_random_cluster(n, seed=seed, min_dist=md)
            if _validate_positions(pos):
                candidates.append(pos)
    
    best_energy = np.inf
    best_positions = None
    
    for pos in candidates:
        if time.time() > deadline - 10:
            break
        
        # Phase 1: Anneal from soft to true potential
        annealed = _anneal_relax(pos, potential, kind, n_exp_true, m_exp_true, c_true, time_limit=6.0)
        
        # Phase 2: True relax from the annealed position
        true_energy, true_relaxed = _try_relax(annealed, potential, kind)
        if true_energy < best_energy:
            best_energy = true_energy
            best_positions = true_relaxed.copy()
        
        # Also try direct relax from original
        direct_energy, direct_relaxed = _try_relax(pos, potential, kind)
        if direct_energy < best_energy:
            best_energy = direct_energy
            best_positions = direct_relaxed.copy()
    
    # Polish pass
    if best_positions is not None and time.time() < deadline - 5:
        try:
            energy2, relaxed2 = relax(best_positions, potential, kind)
            if np.isfinite(energy2) and _validate_positions(relaxed2) and energy2 < best_energy:
                best_energy = energy2
                best_positions = relaxed2.copy()
        except Exception:
            pass
    
    if best_positions is None:
        pos = _generate_random_cluster(n, seed=99, min_dist=0.8)
        energy, best_positions = _try_relax(pos, potential, kind)
        if best_positions is None:
            best_positions = pos
    
    return best_positions
# EVOLVE-BLOCK-END
