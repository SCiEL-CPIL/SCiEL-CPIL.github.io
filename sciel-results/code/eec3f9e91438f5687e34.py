"""Search-portfolio solver for Morse cluster global minima (SCiEL EVOLVE-BLOCK)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax

_T0 = None
_HARD_DEADLINE = None


def _init_timer():
    global _T0, _HARD_DEADLINE
    if _T0 is None:
        _T0 = time.time()
        # Contract: six calls together < 300 s. Leave margin; hard cap under 330 s kill.
        _HARD_DEADLINE = _T0 + 292.0
    return _T0


def _time_left():
    global _HARD_DEADLINE
    return max(0.0, _HARD_DEADLINE - time.time())


def _enforce_min_dist(P, min_d=0.5):
    """Ensure all pair distances >= min_d by small local pushes."""
    P = np.array(P, dtype=float)
    n = P.shape[0]
    for _ in range(50):
        ok = True
        for i in range(n):
            for j in range(i + 1, n):
                d = P[i] - P[j]
                dist = np.linalg.norm(d)
                if dist < min_d:
                    if dist < 1e-12:
                        d = np.array([1.0, 0, 0])
                    else:
                        d = d / dist
                    push = (min_d - dist) / 2.0 + 0.01
                    P[i] += push * d
                    P[j] -= push * d
                    ok = False
        if ok:
            break
    return P


def _fcc_points(n, spacing=1.0):
    """Generate n points from an FCC lattice, centered."""
    pts = []
    a = spacing
    # FCC basis: (0,0,0), (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0)
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                base = np.array([i, j, k]) * a
                for b in [np.array([0, 0, 0]), np.array([0, a/2, a/2]),
                          np.array([a/2, 0, a/2]), np.array([a/2, a/2, 0])]:
                    pts.append(base + b)
    pts = np.array(pts)
    # Sort by distance from center and take closest n
    center = pts.mean(axis=0)
    d2 = np.sum((pts - center)**2, axis=1)
    idx = np.argsort(d2)[:n]
    pts = pts[idx] - pts[idx].mean(axis=0)
    return pts


def _hcp_points(n, spacing=1.0):
    """Generate n points from an HCP lattice, centered."""
    pts = []
    a = spacing
    c = a * np.sqrt(8.0/3.0)
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0)/2.0 * j
                z = c * (k + (1/3.0) * ((j % 2) + 1))
                pts.append(np.array([x, y, z]))
    pts = np.array(pts)
    center = pts.mean(axis=0)
    d2 = np.sum((pts - center)**2, axis=1)
    idx = np.argsort(d2)[:n]
    pts = pts[idx] - pts[idx].mean(axis=0)
    return pts


def _bcc_points(n, spacing=1.0):
    """Generate n points from a BCC lattice, centered."""
    pts = []
    a = spacing
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                base = np.array([i, j, k]) * a
                pts.append(base)
                pts.append(base + np.array([a/2, a/2, a/2]))
    pts = np.array(pts)
    center = pts.mean(axis=0)
    d2 = np.sum((pts - center)**2, axis=1)
    idx = np.argsort(d2)[:n]
    pts = pts[idx] - pts[idx].mean(axis=0)
    return pts


def _sc_points(n, spacing=1.0):
    """Generate n points from a simple cubic lattice, centered."""
    pts = []
    a = spacing
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                pts.append(np.array([i, j, k]) * a)
    pts = np.array(pts)
    center = pts.mean(axis=0)
    d2 = np.sum((pts - center)**2, axis=1)
    idx = np.argsort(d2)[:n]
    pts = pts[idx] - pts[idx].mean(axis=0)
    return pts


def _random_compact(n, spacing=1.0, seed=42):
    """Generate a random compact cluster with minimum distance enforcement."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    points = []
    max_tries = 50000
    tries = 0
    while len(points) < n and tries < max_tries:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 * spacing for q in points):
                points.append(p)
        tries += 1
    if len(points) < n:
        # Fallback: add points on a sphere shell
        for i in range(n - len(points)):
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(0, np.pi)
            r = radius * 0.9
            p = np.array([r * np.sin(phi) * np.cos(theta),
                          r * np.sin(phi) * np.sin(theta),
                          r * np.cos(phi)])
            points.append(p)
    return np.array(points)


def _relax_one(P, potential, parameter, deadline_check=None):
    """Relax a single configuration, returning (energy, positions)."""
    try:
        if deadline_check and deadline_check() < 2.0:
            # Not enough time; return current energy estimate
            return _morse_E(P, parameter), P
        energy, positions = relax(P, potential, parameter)
        return energy, positions
    except Exception:
        return np.inf, P


def _morse_E(P, rho):
    """Exact energy E = sum_{i<j} exp(rho(1-r))(exp(rho(1-r))-2)."""
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    iu = np.triu_indices(n, k=1)
    r = d[iu]
    t = np.exp(rho * (1.0 - r))
    return float(np.sum(t * (t - 2.0)))


def construct(instance):
    """Multi-start portfolio: lattice-based + random compact, all relaxed, best kept."""
    _init_timer()
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    spacing = {"morse": 1.0, "sutton_chen": 0.72, "dzugutov": 1.1}[potential]

    candidates = []

    # Lattice-based starts
    candidates.append(_fcc_points(size, spacing))
    candidates.append(_hcp_points(size, spacing))
    candidates.append(_bcc_points(size, spacing))
    candidates.append(_sc_points(size, spacing))

    # Random compact starts with different seeds
    for seed in [size, size + 100, size + 200, size + 300, size + 400]:
        candidates.append(_random_compact(size, spacing, seed))

    # Slightly perturbed FCC (often helps escape local minima)
    rng = np.random.default_rng(12345)
    fcc_pert = _fcc_points(size, spacing) + rng.normal(0, 0.05 * spacing, (size, 3))
    candidates.append(fcc_pert)

    best_energy = np.inf
    best_positions = None

    for i, cand in enumerate(candidates):
        if _time_left() < 3.0:
            break
        cand = _enforce_min_dist(cand, 0.5)
        energy, positions = _relax_one(cand, potential, parameter, deadline_check=_time_left)
        if energy < best_energy:
            best_energy = energy
            best_positions = positions

    if best_positions is None:
        # Fallback: return the first candidate as-is
        best_positions = _enforce_min_dist(candidates[0], 0.5)

    return best_positions
# EVOLVE-BLOCK-END
