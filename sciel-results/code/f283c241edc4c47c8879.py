"""Structure-motif + basin-hopping search for Sutton-Chen cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


# ---- Motif generators (all return (n,3) arrays, reduced units, lattice ~1) ----

def _icosa(n):
    """Mackay icosahedral shells. Radii chosen so nearest-neighbor ~ 1.0."""
    # Icosahedron vertices (unit sphere)
    phi = (1 + 5**0.5) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    # shell radii (Mackay): 1, 1.701, 2.56, 3.31...
    shell_radii = [1.0, 1.701, 2.56, 3.31, 4.0, 4.6, 5.2, 5.8]
    shell_counts = [13, 55, 147, 309, 561, 923, 1431, 2109]
    # pick shells to reach n
    points = [verts[0] * 0.0]  # center
    used = 1
    for r, cap in zip(shell_radii, shell_counts):
        if used + cap <= n:
            points.append(verts * r)
            used += cap
    # if still short, add random points on a slightly larger shell
    while len(points) < n:
        v = np.random.default_rng(used).uniform(-1, 1, 3)
        v /= np.linalg.norm(v)
        r = shell_radii[-1] + 0.1
        points.append(v * r)
        used += 1
    pts = np.vstack(points)[:n]
    # center of mass to origin
    pts -= pts.mean(axis=0)
    return pts


def _fcc_fragment(n):
    """Fragment of an FCC lattice, trimmed to a compact shape."""
    a = 1.0  # lattice constant
    # generate FCC points in a cube
    L = int(np.ceil((n * 2)**(1/3)))
    pts = []
    for i in range(L):
        for j in range(L):
            for k in range(L):
                # FCC basis
                pts.append([i, j, k])
                pts.append([i + 0.5, j + 0.5, k])
                pts.append([i + 0.5, j, k + 0.5])
                pts.append([i, j + 0.5, k + 0.5])
    pts = np.array(pts, dtype=float) * (a / np.sqrt(2))
    # keep n closest to center
    pts -= pts.mean(axis=0)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    pts = pts[order[:n]]
    return pts


def _hcp_fragment(n):
    """Fragment of an HCP lattice."""
    a = 1.0
    c = a * (2 * 3 / 2)**0.5
    pts = []
    L = int(np.ceil((n)**(1/3))) + 2
    for layer in range(2 * L):
        z = layer * c / 2
        offset = (layer % 2) * 0.5
        for i in range(-L, L):
            for j in range(-L, L):
                x = (i + offset) * a
                y = j * a * np.sqrt(3) / 2
                if layer % 2 == 1:
                    x += a * 0.5
                pts.append([x, y, z])
    pts = np.array(pts, dtype=float)
    pts -= pts.mean(axis=0)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    pts = pts[order[:n]]
    return pts


def _bcc_fragment(n):
    """Fragment of a BCC lattice."""
    a = 1.0
    L = int(np.ceil((n)**(1/3))) + 2
    pts = []
    for i in range(-L, L):
        for j in range(-L, L):
            for k in range(-L, L):
                pts.append([i, j, k])
                pts.append([i + 0.5, j + 0.5, k + 0.5])
    pts = np.array(pts, dtype=float) * a / np.sqrt(3)
    pts -= pts.mean(axis=0)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    pts = pts[order[:n]]
    return pts


def _random_cluster(n, seed, spacing=0.8):
    """Random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n**(1/3) * 0.8
    points = []
    max_iter = 20000
    min_dist = 0.7 * spacing
    for _ in range(max_iter):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
    # if not enough, fill randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    pts = np.array(points, dtype=float)
    pts -= pts.mean(axis=0)
    return pts


def _random_shell(n, seed, shell_radius=None):
    """Random points on a spherical shell with minimum spacing."""
    rng = np.random.default_rng(seed)
    if shell_radius is None:
        shell_radius = 1.5 * n**(1/3)
    points = []
    min_dist = 0.6
    max_iter = 20000
    for _ in range(max_iter):
        if len(points) >= n:
            break
        # Random direction on sphere
        z = rng.uniform(-1, 1)
        theta = rng.uniform(0, 2 * np.pi)
        r = np.sqrt(1 - z**2)
        dir_vec = np.array([r * np.cos(theta), r * np.sin(theta), z])
        p = dir_vec * shell_radius
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
    while len(points) < n:
        z = rng.uniform(-1, 1)
        theta = rng.uniform(0, 2 * np.pi)
        r = np.sqrt(1 - z**2)
        dir_vec = np.array([r * np.cos(theta), r * np.sin(theta), z])
        p = dir_vec * shell_radius
        points.append(p)
    pts = np.array(points, dtype=float)
    pts -= pts.mean(axis=0)
    return pts


def _jitter(pts, seed, amount=0.05):
    """Add small random jitter to break symmetry."""
    rng = np.random.default_rng(seed)
    return pts + rng.normal(0, amount, pts.shape)


def _validate(pts, min_dist=0.35):
    """Check coordinates are finite and no pair closer than min_dist."""
    if not np.all(np.isfinite(pts)):
        return False
    n = len(pts)
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pts[i] - pts[j]) < min_dist:
                return False
    return True


def _basin_hopping(pts0, kind, n_steps=8, step_scale=0.15, seed=0, time_limit=None):
    """Simple basin hopping: local relax, perturb, relax, keep best."""
    rng = np.random.default_rng(seed)
    start_time = time.time()
    best_e, best_p = relax(pts0, 'sutton_chen', kind)
    if not _validate(best_p):
        return best_e, best_p
    current_p = best_p.copy()
    for step in range(n_steps):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        # perturb with adaptive scale
        scale = step_scale * (1.0 + 0.3 * np.sin(step * 0.5))
        noise = rng.normal(0, scale, best_p.shape)
        perturbed = current_p + noise
        if not _validate(perturbed):
            continue
        e, p = relax(perturbed, 'sutton_chen', kind)
        if not _validate(p):
            continue
        if e < best_e:
            best_e, best_p = e, p
            current_p = p.copy()
    return best_e, best_p


def _generate_starts(n, kind, seed_base=0):
    """Generate a diverse set of starting structures."""
    starts = []
    seeds = []
    # Motif-based
    motifs = [
        ('icosa', _icosa),
        ('fcc', _fcc_fragment),
        ('hcp', _hcp_fragment),
        ('bcc', _bcc_fragment),
    ]
    for mi, (name, gen) in enumerate(motifs):
        for jitter_seed in range(4):
            try:
                pts = gen(n)
                if len(pts) >= n:
                    pts = pts[:n]
                # center
                pts = pts - pts.mean(axis=0)
                if _validate(pts, 0.3):
                    starts.append(pts.copy())
                    seeds.append((seed_base + mi*10 + jitter_seed, 'motif'))
                    # jittered versions with different amounts
                    for jamt in [0.03, 0.08]:
                        jp = _jitter(pts, seed_base + mi*10 + jitter_seed + 100 + int(jamt*1000), jamt)
                        if _validate(jp, 0.3):
                            starts.append(jp.copy())
                            seeds.append((seed_base + mi*10 + jitter_seed + 100 + int(jamt*1000), 'jitter'))
            except Exception:
                pass
    # Random clusters
    for r in range(8):
        try:
            pts = _random_cluster(n, seed_base + 1000 + r)
            if _validate(pts, 0.3):
                starts.append(pts.copy())
                seeds.append((seed_base + 1000 + r, 'random'))
        except Exception:
            pass
    # Random shell structures
    for r in range(4):
        try:
            shell_r = 1.2 * n**(1/3) + 0.5 * r
            pts = _random_shell(n, seed_base + 2000 + r, shell_r)
            if _validate(pts, 0.3):
                starts.append(pts.copy())
                seeds.append((seed_base + 2000 + r, 'shell'))
        except Exception:
            pass
    return starts, seeds


def construct(instance):
    """Structure-motif + basin-hopping search for Sutton-Chen cluster global minima."""
    size = instance["n"]
    kind = instance.get("kind", "12-6")
    total_time_limit = 280  # Leave 20s buffer
    
    # Generate diverse starting structures
    starts, seeds = _generate_starts(size, kind, seed_base=42)
    
    if len(starts) == 0:
        # Fallback: single random
        pts = _random_cluster(size, 12345)
        e, p = relax(pts, 'sutton_chen', kind)
        return p
    
    # Use parallel basin hopping with time management
    best_e = np.inf
    best_p = None
    start_time = time.time()
    
    def _worker(args):
        idx, start_pts, seed = args
        try:
            # Adaptive step scale based on cluster size
            step_scale = 0.12 + 0.02 * np.log(size / 10)
            n_steps = max(4, min(10, int(20 / np.sqrt(size))))
            e, p = _basin_hopping(start_pts, kind, n_steps=n_steps, step_scale=step_scale, seed=seed)
            return e, p
        except Exception:
            return np.inf, start_pts
    
    tasks = [(i, s, seeds[i]) for i, s in enumerate(starts)]
    
    # Process in parallel (max 4 threads)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_worker, task): task for task in tasks}
        for future in as_completed(futures):
            if time.time() - start_time > total_time_limit:
                break
            e, p = future.result()
            if not _validate(p):
                continue
            if e < best_e:
                best_e = e
                best_p = p.copy()
    
    if best_p is None:
        # Fallback
        e, p = relax(starts[0], 'sutton_chen', kind)
        return p
    
    # If we have time, do a second round of refinement on the best structure
    elapsed = time.time() - start_time
    if elapsed < total_time_limit - 30:
        try:
            refined_e, refined_p = _basin_hopping(
                best_p, kind, n_steps=8, step_scale=0.08, seed=999, 
                time_limit=total_time_limit - elapsed - 10
            )
            if _validate(refined_p) and refined_e < best_e:
                best_e = refined_e
                best_p = refined_p
        except Exception:
            pass
    
    return best_p
# EVOLVE-BLOCK-END
