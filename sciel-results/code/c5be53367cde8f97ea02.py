"""Direct-structure bootstrap: build compact FCC clusters and relax them."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_DEADLINE = 320.0  # safety margin before 330 s kill

def _remaining():
    return _DEADLINE - time.time()

def _morse_energy(pos, rho):
    pos = np.asarray(pos, dtype=float)
    n = pos.shape[0]
    if n < 2:
        return 0.0
    r = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(r, 0.0)
    mask = r > 1e-12
    x = np.exp(rho * (1.0 - r[mask]))
    return float((x * (x - 2.0)).sum())

def _fcc_cluster(n, a=1.0):
    """Build a compact FCC cluster of exactly n atoms."""
    pts = []
    maxcoord = 3.0 * n ** (1.0 / 3.0) + 6.0
    for i in range(-int(maxcoord), int(maxcoord) + 1):
        for j in range(-int(maxcoord), int(maxcoord) + 1):
            for k in range(-int(maxcoord), int(maxcoord) + 1):
                r2 = i * i + j * j + k * k
                if r2 > maxcoord * maxcoord:
                    continue
                x, y, z = i, j, k
                if (i + j + k) % 2 == 0:
                    pts.append([x * a / 2.0, y * a / 2.0, z * a / 2.0])
    pts = np.array(pts, dtype=float)
    d2 = (pts ** 2).sum(1)
    order = np.argsort(d2)
    return pts[order[:n]].copy()

def _bcc_cluster(n, a=1.0):
    """Build a compact BCC cluster of exactly n atoms."""
    pts = []
    maxcoord = 3.0 * n ** (1.0 / 3.0) + 6.0
    for i in range(-int(maxcoord), int(maxcoord) + 1):
        for j in range(-int(maxcoord), int(maxcoord) + 1):
            for k in range(-int(maxcoord), int(maxcoord) + 1):
                x, y, z = i, j, k
                if (i + j + k) % 2 == 0 or (i == j == k == 0):
                    pts.append([x * a / 2.0, y * a / 2.0, z * a / 2.0])
    pts = np.array(pts, dtype=float)
    d2 = (pts ** 2).sum(1)
    order = np.argsort(d2)
    return pts[order[:n]].copy()

def _shell_cluster(n, a=1.0):
    """Build a spherical shell cluster."""
    pts = []
    core_n = max(1, int(n ** 0.5))
    if core_n < n:
        core = _fcc_cluster(core_n, a)
        pts.extend(core.tolist())
    shell_n = n - len(pts)
    if shell_n > 0:
        radius = a * (n ** (1/3) * 0.8)
        phi = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(shell_n):
            y = 1 - (i / float(shell_n - 1)) * 2
            radius_y = np.sqrt(1 - y**2)
            theta = phi * i
            x = np.cos(theta) * radius_y
            z = np.sin(theta) * radius_y
            pts.append([x * radius, y * radius, z * radius])
    return np.array(pts, dtype=float)

def _random_cluster(n, rng, spacing=1.0):
    """Random compact cluster with minimum separation."""
    radius = spacing * n ** (1.0 / 3.0) * 1.2
    pts = []
    minsep = 0.51
    maxtries = 50000
    t = 0
    while len(pts) < n and t < maxtries:
        t += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= minsep for q in pts):
            pts.append(p)
    if len(pts) < n:
        base = _fcc_cluster(n)
        base = base + rng.normal(0, 0.1, base.shape)
        return base
    return np.array(pts, dtype=float)

def _check_valid(pos, minsep=0.5):
    n = pos.shape[0]
    if n == 0:
        return False
    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    min_d = np.sqrt(d2.min())
    return min_d >= minsep - 1e-6 and np.all(np.isfinite(pos))

def construct(instance):
    """Build a compact FCC cluster of n atoms and relax it to a local minimum."""
    n = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10.0))

    seeds = []
    
    # 1. FCC
    seeds.append(_fcc_cluster(n))
    # 2. BCC
    seeds.append(_bcc_cluster(n))
    # 3. Shell
    seeds.append(_shell_cluster(n))
    
    # 4. Jittered FCC
    rng = np.random.default_rng(n + int(rho))
    fcc_jitter = _fcc_cluster(n) + rng.normal(0, 0.05, (n, 3))
    seeds.append(fcc_jitter)
    
    # 5. Random
    seeds.append(_random_cluster(n, rng))
    
    # 6. Another Random
    seeds.append(_random_cluster(n, rng))

    best_pos = None
    best_e = float("inf")
    
    def _relax_seed(seed):
        try:
            if _remaining() < 10.0:
                return None, float("inf")
            e, pos = relax(np.asarray(seed, dtype=float), potential, rho)
            e = float(e)
            if np.isfinite(e) and _check_valid(pos):
                return pos, e
            return None, float("inf")
        except Exception:
            return None, float("inf")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_relax_seed, s): s for s in seeds}
        for future in as_completed(futures):
            if _remaining() < 5.0:
                break
            try:
                pos, e = future.result()
                if pos is not None and e < best_e:
                    best_e = e
                    best_pos = pos
            except Exception:
                pass

    if best_pos is not None:
        return best_pos

    # Fallback: return the best seed (unrelaxed) if relax failed completely
    return seeds[0]

# EVOLVE-BLOCK-END
