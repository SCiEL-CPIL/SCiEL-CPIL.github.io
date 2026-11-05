"""Morse cluster global minima via structured seeds + diverse local relaxations."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(pos, rho):
    """Exact Morse energy."""
    n = len(pos)
    d = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(r, np.inf)
    x = rho * (1.0 - r)
    e = np.exp(x)
    return float(np.sum(e * (e - 2.0)) * 0.5)


def _morse_gradient(pos, rho):
    """Morse energy gradient (vectorized)."""
    n = len(pos)
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt((diff ** 2).sum(-1, keepdims=True))
    np.fill_diagonal(r, 1.0)
    x = rho * (1.0 - r[..., 0])
    e = np.exp(x)
    # dE/dr = 2*rho*exp(x)*(exp(x)-1)
    de_dr = 2.0 * rho * e * (e - 1.0)
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            dr = diff[i, j] / r[i, j, 0]
            grad[i] += de_dr[i, j] * dr
            grad[j] -= de_dr[i, j] * dr
    return grad


def _min_distance(pos):
    """Compute minimum pairwise distance (vectorized)."""
    n = len(pos)
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


def _is_valid(pos, min_dist=0.5):
    """Check if configuration is valid."""
    if len(pos) == 0:
        return False
    return _min_distance(pos) >= min_dist - 1e-9


def _generate_icosahedron():
    """12 vertices of regular icosahedron with NN distance = 1."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    v /= 2.0
    return v


def _generate_mackay_icosahedron(n):
    """Mackay icosahedral cluster with n atoms."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    if n <= 12:
        ico = _generate_icosahedron()
        if n == 1:
            return np.array([[0.0, 0.0, 0.0]])
        r = np.linalg.norm(ico, axis=1)
        idx = np.argsort(r)[:n]
        return ico[idx]
    
    # Mackay icosahedra: 1, 13, 55, 147, 309, 561, ...
    # Shell radii: 1, 1.902, 2.764, 3.544, ...
    shells = [1.0, 1.902, 2.764, 3.544, 4.286, 5.0]
    shell_counts = [1, 12, 42, 92, 162, 252]
    
    pts = [np.array([0.0, 0.0, 0.0])]
    ico = _generate_icosahedron()
    
    for shell_idx, (radius, count) in enumerate(zip(shells, shell_counts)):
        if len(pts) >= n:
            break
        # Generate points on this shell
        if shell_idx == 0:
            shell_pts = ico * radius
        else:
            # Use icosahedral directions scaled
            shell_pts = ico * radius
            # Add more points by subdividing
            if count > 12:
                # Add face-centered points
                extra = []
                for i in range(12):
                    for j in range(i + 1, 12):
                        if np.linalg.norm(ico[i] - ico[j]) < 1.1:  # adjacent
                            mid = (ico[i] + ico[j]) / 2.0
                            mid = mid / np.linalg.norm(mid) * radius
                            extra.append(mid)
                shell_pts = np.vstack([shell_pts, extra])
        
        # Take as many as needed
        needed = min(count, n - len(pts))
        if len(shell_pts) >= needed:
            r = np.linalg.norm(shell_pts, axis=1)
            idx = np.argsort(r)[:needed]
            pts.extend(list(shell_pts[idx]))
        else:
            pts.extend(list(shell_pts))
    
    pts = np.array(pts[:n])
    if len(pts) < n:
        # Fill with random points
        rng = np.random.default_rng(42)
        while len(pts) < n:
            p = rng.normal(0, 1.5, 3)
            pts = np.vstack([pts, p])
    return pts


def _generate_fcc_cluster(n):
    """FCC cluster fragment with n atoms."""
    if n <= 1:
        return np.zeros((n, 3))
    
    # FCC primitive vectors with NN distance = 1
    a = 1.0
    v1 = np.array([0, a/np.sqrt(2), a/np.sqrt(2)])
    v2 = np.array([a/np.sqrt(2), 0, a/np.sqrt(2)])
    v3 = np.array([a/np.sqrt(2), a/np.sqrt(2), 0])
    
    # Generate lattice points
    side = int(np.ceil(n ** (1/3))) + 3
    pts = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                p = i * v1 + j * v2 + k * v3
                pts.append(p)
    
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    r = np.linalg.norm(pts, axis=1)
    idx = np.argsort(r)[:n]
    return pts[idx]


def _generate_bcc_cluster(n):
    """BCC cluster fragment with n atoms."""
    if n <= 1:
        return np.zeros((n, 3))
    
    # BCC primitive vectors with NN distance = 1
    a = 1.0
    v1 = np.array([0, a/2, a/2])
    v2 = np.array([a/2, 0, a/2])
    v3 = np.array([a/2, a/2, 0])
    
    side = int(np.ceil(n ** (1/3))) + 3
    pts = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                p = i * v1 + j * v2 + k * v3
                pts.append(p)
                # Add body-center atom
                p2 = p + np.array([a/2, a/2, a/2]) / 2.0
                pts.append(p2)
    
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    r = np.linalg.norm(pts, axis=1)
    idx = np.argsort(r)[:n]
    return pts[idx]


def _generate_random_cluster(n, seed=0, min_dist=0.85):
    """Random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = 1.0 * n ** (1.0/3.0) * 0.9
    points = []
    max_tries = 100000
    tries = 0
    while len(points) < n and tries < max_tries:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    ok = False
                    break
            if ok:
                points.append(p)
        tries += 1
    
    # If we couldn't fit all, add remaining with looser constraint
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _perturb(pos, scale, seed=0):
    """Add random Gaussian perturbation."""
    rng = np.random.default_rng(seed)
    return pos + rng.normal(0, scale, pos.shape)


def _ensure_min_distance(pos, min_dist=0.5):
    """Nudge atoms apart if any pair is closer than min_dist."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(100):
        moved = False
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(dist, np.inf)
        
        for i in range(n):
            for j in range(i + 1, n):
                r = dist[i, j]
                if r < min_dist:
                    if r > 1e-10:
                        push = (diff[i, j] / r) * (min_dist - r) * 0.5
                        pos[i] += push
                        pos[j] -= push
                    else:
                        rng = np.random.default_rng(i * 1000 + j)
                        pos[i] += rng.uniform(-0.1, 0.1, 3)
                        pos[j] -= rng.uniform(-0.1, 0.1, 3)
                    moved = True
        if not moved:
            break
    return pos


def _basin_hop(pos, rho, potential, max_iter=10, max_step=0.3, min_step=0.05, deadline=None):
    """Basin hopping: perturb, relax, accept if lower energy."""
    best_pos = pos.copy()
    best_energy = _morse_energy(pos, rho)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Adaptive step size
        scale = max_step * (min_step / max_step) ** (it / max_iter)
        perturbed = _perturb(best_pos, scale, seed=it)
        perturbed = _ensure_min_distance(perturbed, 0.5)
        
        try:
            energy, new_pos = relax(perturbed, potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_pos = new_pos.copy()
        except Exception:
            continue
    
    return best_energy, best_pos


def _relax_and_validate(pos, potential, rho, deadline=None):
    """Relax and validate configuration."""
    try:
        energy, new_pos = relax(pos, potential, rho)
        if _is_valid(new_pos, 0.5):
            return energy, new_pos
    except Exception:
        pass
    return np.inf, pos


def construct(instance):
    """Find low-energy Morse cluster via structured seeds + basin hopping."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    potential = instance["potential"]
    t0 = time.time()
    deadline = t0 + 320.0  # safety margin before 330s kill
    
    # Generate diverse starting configurations
    candidates = []
    
    # 1. Mackay icosahedron
    try:
        ico = _generate_mackay_icosahedron(n)
        ico = _ensure_min_distance(ico, 0.5)
        candidates.append(ico)
    except Exception:
        pass
    
    # 2. FCC fragment
    try:
        fcc = _generate_fcc_cluster(n)
        fcc = _ensure_min_distance(fcc, 0.5)
        candidates.append(fcc)
    except Exception:
        pass
    
    # 3. BCC fragment
    try:
        bcc = _generate_bcc_cluster(n)
        bcc = _ensure_min_distance(bcc, 0.5)
        candidates.append(bcc)
    except Exception:
        pass
    
    # 4-11. Random clusters with different seeds
    for s in range(8):
        try:
            rc = _generate_random_cluster(n, seed=100 + s)
            rc = _ensure_min_distance(rc, 0.5)
            candidates.append(rc)
        except Exception:
            pass
    
    # Ensure all candidates have exactly n atoms
    valid_candidates = []
    for c in candidates:
        c = np.asarray(c, dtype=float)
        if len(c) != n:
            if len(c) > n:
                c = c[:n]
            else:
                rng = np.random.default_rng(999)
                while len(c) < n:
                    c = np.vstack([c, rng.uniform(-2, 2, 3)])
        c = _ensure_min_distance(c, 0.5)
        valid_candidates.append(c)
    
    # Phase 1: Relax all candidates
    best_energy = np.inf
    best_pos = valid_candidates[0] if valid_candidates else np.zeros((n, 3))
    relaxed_results = []
    
    for idx, start in enumerate(valid_candidates):
        if time.time() > deadline - 60:  # Reserve time for basin hopping
            break
        try:
            energy, positions = relax(start, potential, rho)
            if _is_valid(positions, 0.5):
                relaxed_results.append((energy, positions.copy()))
                if energy < best_energy:
                    best_energy = energy
                    best_pos = positions.copy()
        except Exception:
            continue
    
    # Phase 2: Basin hopping from top relaxed results
    relaxed_results.sort(key=lambda x: x[0])
    top_k = min(5, len(relaxed_results))
    
    for k in range(top_k):
        if time.time() > deadline - 10:
            break
        try:
            energy, pos = relaxed_results[k]
            bh_energy, bh_pos = _basin_hop(
                pos, rho, potential,
                max_iter=15, max_step=0.35, min_step=0.08,
                deadline=deadline - 10
            )
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos.copy()
        except Exception:
            continue
    
    # Phase 3: Final polish from best
    if time.time() < deadline - 5:
        try:
            energy, positions = relax(best_pos, potential, rho)
            if _is_valid(positions, 0.5) and energy < best_energy:
                best_energy = energy
                best_pos = positions.copy()
        except Exception:
            pass
    
    return np.asarray(best_pos, dtype=float)
# EVOLVE-BLOCK-END
