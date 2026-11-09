"""Morse cluster global minima: structured seeds + basin hopping with deadline guard."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_TOTAL_BUDGET = 300.0
_DEADLINE = time.time() + (_TOTAL_BUDGET - 15.0)  # 15s safety margin


def _time_left():
    return _DEADLINE - time.time()


def _morse_energy(rho, coords):
    n = len(coords)
    if n < 2:
        return 0.0
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    if d.min() < 0.5:
        return 1e12
    u = np.exp(rho * (1.0 - d))
    e = (u * (u - 2.0)).sum()
    return 0.5 * e


def _morse_grad(rho, coords):
    n = len(coords)
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    u = np.exp(rho * (1.0 - d))
    grad = rho * u * (2.0 * u - 1.0)
    grad = (grad * (coords[:, None, :] - coords[None, :, :])).sum(axis=1)
    return 0.5 * grad


def _local_min(rho, x0, max_iter=600):
    """L-BFGS-B local minimization of Morse energy."""
    n = len(x0)
    if n < 2:
        return x0, 0.0

    def fun(p):
        return _morse_energy(rho, p.reshape(n, 3))

    def jac(p):
        return _morse_grad(rho, p.reshape(n, 3)).reshape(-1)

    res = minimize(fun, x0, jac=jac, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'maxls': 40})
    if res.success or res.fun < _morse_energy(rho, x0):
        return res.x.reshape(n, 3), res.fun
    return x0, _morse_energy(rho, x0)


def _icosahedron_shell(radius):
    """12 vertices of a regular icosahedron scaled to given radius."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    return verts * radius


def _fcc_positions(n, spacing=1.0):
    """Generate FCC-like positions for n atoms."""
    pts = []
    a = spacing
    # FCC lattice points
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]], dtype=float)
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                for b in basis:
                    p = np.array([i, j, k]) * a + b
                    pts.append(p)
    pts = np.array(pts)
    # Sort by distance from origin
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    pts = pts[idx]
    # Center
    pts -= pts.mean(axis=0)
    # Take closest n
    if len(pts) >= n:
        return pts[:n].copy()
    # Pad with small random offsets
    while len(pts) < n:
        pts = np.vstack([pts, pts[0] + 0.1 * np.random.randn(3)])
    return pts[:n]


def _bcc_positions(n, spacing=1.0):
    """Generate BCC-like positions for n atoms."""
    pts = []
    a = spacing
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5]], dtype=float)
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                for b in basis:
                    p = np.array([i, j, k]) * a + b
                    pts.append(p)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    pts = pts[idx]
    pts -= pts.mean(axis=0)
    if len(pts) >= n:
        return pts[:n].copy()
    while len(pts) < n:
        pts = np.vstack([pts, pts[0] + 0.1 * np.random.randn(3)])
    return pts[:n]


def _icosahedral_cluster(n, spacing=1.0):
    """Build an icosahedral-shell cluster: inner core + concentric shells."""
    pts = []
    # Core: 1 atom at center
    pts.append(np.zeros(3))
    # Shell 1: 12 vertices of icosahedron
    shell1_r = spacing
    shell1 = _icosahedron_shell(shell1_r)
    pts.extend(list(shell1))
    # If need more, add second shell (octahedral or larger icosahedron)
    if n > 13:
        # Second shell: 6 vertices of octahedron at larger radius
        shell2_r = spacing * 1.6
        octa = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                         [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float) * shell2_r
        pts.extend(list(octa))
    if n > 19:
        # Third shell: icosahedron at larger radius
        shell3_r = spacing * 2.2
        shell3 = _icosahedron_shell(shell3_r)
        pts.extend(list(shell3))
    pts = np.array(pts)
    if len(pts) >= n:
        return pts[:n].copy()
    while len(pts) < n:
        pts = np.vstack([pts, pts[0] + 0.15 * np.random.randn(3)])
    return pts[:n]


def _random_perturb(coords, sigma):
    return coords + sigma * np.random.randn(*coords.shape)


def construct(instance):
    """Structured seeds + basin hopping for Morse cluster global minima."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    rng = np.random.default_rng(42)

    # Generate diverse starting points
    starts = []

    # Icosahedral cluster
    starts.append(_icosahedral_cluster(n, 1.0))
    # FCC
    starts.append(_fcc_positions(n, 1.0))
    # BCC
    starts.append(_bcc_positions(n, 1.0))
    # Random compact (rejection sampling)
    radius = 1.0 * n ** (1.0 / 3.0) * 0.8
    pts = []
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in pts:
                if np.linalg.norm(p - q) < 0.55:
                    ok = False
                    break
            if ok:
                pts.append(p)
    starts.append(np.array(pts))
    # Second random with different seed
    rng2 = np.random.default_rng(n * 7 + 13)
    pts2 = []
    while len(pts2) < n:
        p = rng2.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in pts2:
                if np.linalg.norm(p - q) < 0.55:
                    ok = False
                    break
            if ok:
                pts2.append(p)
    starts.append(np.array(pts2))

    # Local minimize each start
    best_pos = None
    best_e = np.inf
    for s in starts:
        if _time_left() < 5.0:
            break
        pos, e = _local_min(rho, s, max_iter=400)
        if e < best_e:
            best_e = e
            best_pos = pos.copy()

    # Basin hopping: iterate perturb + local minimize
    # Budget: spend remaining time on basin hopping
    bh_budget = max(0.0, _time_left() - 3.0)
    if bh_budget > 2.0 and best_pos is not None:
        # Adaptive sigma: start moderate, shrink over time
        sigma0 = 0.3
        sigma_min = 0.05
        max_iter_bh = int(bh_budget / 1.0)  # rough: ~1s per iteration
        for it in range(max_iter_bh):
            if _time_left() < 3.0:
                break
            # Anneal sigma
            frac = it / max(1, max_iter_bh)
            sigma = sigma0 * (1.0 - frac) + sigma_min * frac
            # Perturb
            x_pert = _random_perturb(best_pos, sigma)
            # Local minimize
            pos, e = _local_min(rho, x_pert, max_iter=200)
            if e < best_e:
                best_e = e
                best_pos = pos.copy()
            # Occasional larger kick to escape deep local minima
            if it % 10 == 5:
                x_kick = _random_perturb(best_pos, sigma * 2.5)
                pos_k, e_k = _local_min(rho, x_kick, max_iter=150)
                if e_k < best_e:
                    best_e = e_k
                    best_pos = pos_k.copy()

    # Final polish
    if best_pos is not None and _time_left() > 2.0:
        best_pos, best_e = _local_min(rho, best_pos, max_iter=500)

    if best_pos is None:
        # Fallback: return first start
        best_pos = starts[0]

    # Ensure minimum distance constraint
    # If any pair < 0.5, nudge apart
    for _ in range(50):
        d = cdist(best_pos, best_pos)
        np.fill_diagonal(d, np.inf)
        if d.min() >= 0.5:
            break
        # Find closest pair and separate
        i, j = np.unravel_index(d.argmin(), d.shape)
        mid = (best_pos[i] + best_pos[j]) / 2.0
        diff = best_pos[i] - best_pos[j]
        norm = np.linalg.norm(diff)
        if norm < 1e-10:
            diff = np.array([1.0, 0.0, 0.0])
        diff /= np.linalg.norm(diff)
        sep = (0.5 - norm) / 2.0 + 0.01
        best_pos[i] = mid + diff * sep
        best_pos[j] = mid - diff * sep

    return best_pos
