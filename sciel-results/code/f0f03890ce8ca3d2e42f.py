"""Sutton-Chen transition-metal cluster global minima: coarse-to-fine annealing + multi-start structured seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
from scipy.optimize import minimize


def _ensure_min_dist(pts, dmin=0.355):
    """Push apart any pair closer than dmin so the result is always valid."""
    pts = np.asarray(pts, dtype=float).copy()
    n = pts.shape[0]
    if n < 2:
        return pts
    for _ in range(200):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                r = np.linalg.norm(d)
                if r < dmin:
                    if r < 1e-9:
                        pts[j] += np.array([dmin, 0.0, 0.0])
                    else:
                        push = (dmin - r) * 0.5
                        pts[i] += push * d / r
                        pts[j] -= push * d / r
                    moved = True
        if not moved:
            break
    return pts


def _validate_positions(pos):
    """Check if positions are finite and no atoms are closer than 0.35."""
    if not np.all(np.isfinite(pos)):
        return False
    if pos.shape[0] < 2:
        return True
    for i in range(pos.shape[0]):
        for j in range(i + 1, pos.shape[0]):
            if np.linalg.norm(pos[i] - pos[j]) < 0.35:
                return False
    return True


def _fcc_cluster(n, spacing=0.72):
    """Generate an n-atom fcc-like cluster by filling fcc lattice sites nearest to origin."""
    if n <= 0:
        return np.empty((0, 3))
    basis = np.array([[0, 0, 0],
                      [0, 0.5, 0.5],
                      [0.5, 0, 0.5],
                      [0.5, 0.5, 0]], dtype=float)
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                cubic = np.array([i, j, k], dtype=float)
                for b in basis:
                    p = (cubic + b) * spacing
                    sites.append(p)
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    selected = sites[idx[:n]]
    selected = selected - selected.mean(axis=0)
    return selected


def _hcp_cluster(n, spacing=0.72):
    """Generate an n-atom hcp-like cluster by filling hcp lattice sites nearest to origin."""
    if n <= 0:
        return np.empty((0, 3))
    sites = []
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    for layer in range(-max_shell, max_shell + 1):
        z = layer * c
        for i in range(-max_shell, max_shell + 1):
            for j in range(-max_shell, max_shell + 1):
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0) / 2.0 * j
                sites.append([x, y, z])
                sites.append([x + a * 0.5, y + a * np.sqrt(3.0) / 6.0, z])
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    selected = sites[idx[:n]]
    selected = selected - selected.mean(axis=0)
    return selected


def _icosahedral_cluster(n, spacing=0.72):
    """Generate an n-atom icosahedral shell cluster."""
    if n <= 0:
        return np.empty((0, 3))
    phi = (1 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, -phi, 1/phi], [0, phi, -1/phi], [0, -phi, -1/phi],
        [phi, 1/phi, 0], [-phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, -1/phi, 0],
        [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)]
    
    for v in ico_verts[:12]:
        points.append(v * spacing * 1.0)
    
    if n <= 13:
        selected = np.array(points[:n])
        return selected - selected.mean(axis=0)
    
    num_second = min(42, n - 13)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for k in range(num_second):
        z = 1.0 - 2.0 * (k + 0.5) / num_second
        r = np.sqrt(1.0 - z * z)
        theta = golden_angle * k
        p = np.array([r * np.cos(theta), r * np.sin(theta), z])
        points.append(p * spacing * 1.8)
    
    remaining = n - len(points)
    if remaining > 0:
        num_third = remaining
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for k in range(num_third):
            z = 1.0 - 2.0 * (k + 0.5) / num_third
            r = np.sqrt(1.0 - z * z)
            theta = golden_angle * (k + 0.3)
            p = np.array([r * np.cos(theta), r * np.sin(theta), z])
            points.append(p * spacing * 2.6)
    
    selected = np.array(points[:n])
    return selected - selected.mean(axis=0)


def _compressed_icosahedral(n, spacing=0.72):
    """Icosahedral with a compressed inner region (mimics real TM clusters)."""
    base = _icosahedral_cluster(n, spacing)
    r = np.linalg.norm(base, axis=1)
    scale = np.where(r < spacing * 1.5, 0.85, 1.0)
    return base * scale[:, None]


def _random_compact(n, spacing, seed):
    """Generate a random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    pts = []
    attempts = 0
    while len(pts) < n and attempts < 20000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.7 * spacing for q in pts):
            pts.append(p)
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts)


def _jitter_cluster(base, n, spacing=0.72, seed=0, jitter_frac=0.15):
    """Add random jitter to a base cluster to create diverse starting points."""
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0, jitter_frac * spacing, size=base.shape)
    return base + jitter


def _soft_energy_gradient(pts, kind, n_scale, m_scale, c_scale):
    """Compute energy and gradient with softened Sutton-Chen parameters."""
    if kind == "12-6":
        n_exp, m_exp, c = 12.0, 6.0, 144.41
    elif kind == "9-6":
        n_exp, m_exp, c = 9.0, 6.0, 39.432
    else:  # 10-8
        n_exp, m_exp, c = 10.0, 8.0, 34.408
    
    n_exp = n_exp * n_scale
    m_exp = m_exp * m_scale
    c = c * c_scale
    
    n = pts.shape[0]
    energy = 0.0
    grad = np.zeros_like(pts)
    
    for i in range(n):
        pair_r = np.linalg.norm(pts[i] - pts, axis=1)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        rij = pair_r[mask]
        
        if len(rij) == 0:
            continue
        
        inv_r_n = np.power(rij, -n_exp)
        inv_r_m = np.power(rij, -m_exp)
        
        pair_e = 0.5 * np.sum(inv_r_n) - c * np.sqrt(np.sum(inv_r_m))
        energy += pair_e
        
        # Gradient with respect to pts[i]
        # d/dr of 0.5 * r^(-n) = -0.5*n*r^(-n-1)
        # d/dr of -c*sqrt(sum r^(-m)) = -c * (1/(2*sqrt(S))) * sum(-m*r^(-m-1))
        S = np.sum(inv_r_m)
        if S > 1e-30:
            sqrt_S = np.sqrt(S)
            d_dr_rep = -0.5 * n_exp * np.power(rij, -n_exp - 1)
            d_dr_att = -c * (1.0 / (2.0 * sqrt_S)) * (-m_exp * np.power(rij, -m_exp - 1))
            d_dr = d_dr_rep + d_dr_att
        else:
            d_dr = -0.5 * n_exp * np.power(rij, -n_exp - 1)
        
        # Vector from j to i
        vecs = pts[i] - pts[mask]
        vecs = vecs / rij[:, None]
        grad[i] -= np.sum(d_dr[:, None] * vecs, axis=0)
    
    return energy, grad


def _soft_relax(pts, kind, n_scale, m_scale, c_scale, max_iter=300, tol=1e-6):
    """Relax using softened potential parameters."""
    def obj(x):
        x = x.reshape(-1, 3)
        e, g = _soft_energy_gradient(x, kind, n_scale, m_scale, c_scale)
        return e, g.flatten()
    
    try:
        res = minimize(obj, pts.flatten(), method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'gtol': tol, 'maxls': 40})
        if res.success or np.isfinite(res.fun):
            p = res.x.reshape(-1, 3)
            if np.all(np.isfinite(p)):
                return float(res.fun), p
    except Exception:
        pass
    return None, None


def _anneal_relax(pts, kind, t0, deadline, n_iter=50):
    """Coarse-to-fine annealing: start with softened potential, ramp to full."""
    best_e = np.inf
    best_p = pts.copy()
    
    # Phase 1: Coarse basin hopping with softened potential
    rng = np.random.default_rng(int(time.time() * 1000) % 100000)
    coarse_steps = min(n_iter, max(10, int((deadline - 40.0 - time.time()) / max(0.1, (deadline - 40.0 - t0) * 0.4))))
    
    for step in range(coarse_steps):
        if time.time() > deadline - 40.0:
            break
        
        # Soften parameters: interpolate from very soft to moderate
        frac = step / max(1, coarse_steps - 1)
        n_scale = 0.4 + 0.4 * frac  # 0.4 -> 0.8
        m_scale = 0.4 + 0.4 * frac
        c_scale = 0.3 + 0.4 * frac
        
        # Perturb current best
        noise_amp = 0.15 * (1.0 - 0.5 * frac)
        perturbed = best_p + rng.normal(0, noise_amp, size=best_p.shape)
        
        # Relax with softened potential
        e_soft, p_soft = _soft_relax(perturbed, kind, n_scale, m_scale, c_scale, max_iter=200, tol=1e-5)
        if e_soft is not None and np.isfinite(e_soft):
            # Evaluate with TRUE potential
            e_true, p_true = relax(p_soft, 'sutton_chen', kind)
            if np.isfinite(e_true) and _validate_positions(p_true) and e_true < best_e:
                best_e = e_true
                best_p = p_true.copy()
    
    # Phase 2: Fine-tuning with full potential
    remaining_time = deadline - 20.0 - time.time()
    if remaining_time > 10.0:
        fine_steps = min(15, max(3, int(remaining_time / 5.0)))
        for step in range(fine_steps):
            if time.time() > deadline - 20.0:
                break
            noise_amp = 0.05 * (1.0 - step / max(1, fine_steps))
            perturbed = best_p + rng.normal(0, noise_amp, size=best_p.shape)
            e_full, p_full = relax(perturbed, 'sutton_chen', kind)
            if np.isfinite(e_full) and _validate_positions(p_full) and e_full < best_e:
                best_e = e_full
                best_p = p_full.copy()
    
    return best_e, best_p


def _nested_relax(pts, potential, parameter, t0, deadline):
    """Annealed relaxation: coarse-to-fine potential softening + final polish."""
    # First do a quick standard relax to get a reasonable starting point
    try:
        e1, p1 = relax(pts, potential, parameter)
    except Exception:
        return None, None
    if not np.all(np.isfinite(p1)):
        return None, None
    
    if time.time() > deadline - 30.0:
        return e1, p1
    
    # Now do annealed relaxation from the relaxed point
    try:
        e2, p2 = _anneal_relax(p1, parameter, t0, deadline, n_iter=40)
        if e2 is not None and np.isfinite(e2) and e2 < e1 and _validate_positions(p2):
            return e2, p2
    except Exception:
        pass
    
    # Final polish with tight tolerance
    if time.time() < deadline - 15.0:
        try:
            p_polish = p1 + 0.01 * np.random.default_rng(12345).standard_normal(p1.shape)
            e_polish, p_polish = relax(p_polish, potential, parameter)
            if np.isfinite(e_polish) and e_polish < e1 and _validate_positions(p_polish):
                return e_polish, p_polish
        except Exception:
            pass
    
    return e1, p1


def construct(instance):
    """Multi-start structured seed relaxation for Sutton-Chen clusters."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    t_start = time.time()
    deadline = t_start + 280.0  # 280s out of 330s, leave 50s safety margin
    
    # Generate diverse initial geometries
    spacing = 0.72
    geometries = []
    
    # 1. FCC-like
    try:
        fcc = _fcc_cluster(size, spacing)
        if fcc.shape[0] == size:
            geometries.append(fcc)
    except Exception:
        pass
    
    # 2. HCP-like
    try:
        hcp = _hcp_cluster(size, spacing)
        if hcp.shape[0] == size:
            geometries.append(hcp)
    except Exception:
        pass
    
    # 3. Icosahedral
    try:
        ico = _icosahedral_cluster(size, spacing)
        if ico.shape[0] == size:
            geometries.append(ico)
    except Exception:
        pass
    
    # 4. Compressed icosahedral
    try:
        comp_ico = _compressed_icosahedral(size, spacing)
        if comp_ico.shape[0] == size:
            geometries.append(comp_ico)
    except Exception:
        pass
    
    # 5. Random compact with a few seeds
    for s in [size, size + 101, size + 202]:
        try:
            rc = _random_compact(size, spacing, s)
            if rc.shape[0] == size:
                geometries.append(rc)
        except Exception:
            pass
    
    # 6. Jittered versions of structured geometries
    if geometries:
        base = geometries[0]
        for jseed in [10, 20, 30]:
            try:
                jit = _jitter_cluster(base, size, spacing, seed=jseed, jitter_frac=0.15)
                if jit.shape[0] == size:
                    geometries.append(jit)
            except Exception:
                pass
    
    # Ensure we have at least one geometry
    if not geometries:
        positions = np.zeros((size, 3))
        positions[:, 0] = np.arange(size) * 0.5
        geometries.append(positions)
    
    # Relax each geometry and keep the best
    best_energy = np.inf
    best_positions = None
    
    # Budget: spend more time on fewer high-quality annealed relaxations
    # instead of many quick ones
    num_geoms_to_try = min(len(geometries), 5)
    
    for idx, geom in enumerate(geometries[:num_geoms_to_try]):
        if time.time() > deadline - 30.0:
            break
        try:
            energy, positions = _nested_relax(geom, potential, kind, t_start, deadline)
            if energy is not None and positions is not None:
                if np.isfinite(energy) and energy < best_energy and _validate_positions(positions):
                    best_energy = energy
                    best_positions = positions
        except Exception:
            continue
    
    if best_positions is None:
        best_positions = geometries[0]
        best_energy = sutton_chen_energy_gradient(best_positions, kind)[0]
    
    # Phase 2: Additional basin hopping with annealing on the best found
    if time.time() < deadline - 15.0:
        remaining = deadline - 15.0 - time.time()
        if remaining > 15.0:
            try:
                e_bh, p_bh = _anneal_relax(best_positions, kind, t_start, deadline - 10.0, n_iter=30)
                if e_bh is not None and np.isfinite(e_bh) and e_bh < best_energy and _validate_positions(p_bh):
                    best_energy = e_bh
                    best_positions = p_bh
            except Exception:
                pass
    
    # Phase 3: Quick shake-and-relax with full potential for final polish
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(42)
        shake_count = 0
        while time.time() < deadline - 8.0 and shake_count < 20:
            noise = rng.normal(0, 0.08, size=best_positions.shape)
            new_pos = best_positions + noise
            
            if not _validate_positions(new_pos):
                shake_count += 1
                continue
            
            try:
                energy, positions = relax(new_pos, potential, kind)
                if np.isfinite(energy) and energy < best_energy and _validate_positions(positions):
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
            shake_count += 1
    
    # Final validity guarantee
    best_positions = _ensure_min_dist(best_positions)
    
    if not np.all(np.isfinite(best_positions)):
        best_positions = np.zeros((size, 3))
        best_positions[:, 0] = np.arange(size) * 0.5
        
    return best_positions
# EVOLVE-BLOCK-END
