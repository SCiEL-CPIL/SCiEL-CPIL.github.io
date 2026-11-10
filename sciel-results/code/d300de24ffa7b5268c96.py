"""Morse cluster global minima: basin hopping with structured crystalline seeds and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_DEADLINE = 320.0  # seconds; evaluator kills at 330s, keep safety margin


def _morse_energy(pos, rho):
    """E = sum_{i<j} exp(rho(1-r_ij)) (exp(rho(1-r_ij)) - 2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    x = rho * (1.0 - r_ij)
    e = np.exp(x) * (np.exp(x) - 2.0)
    return float(np.sum(e))


def _morse_energy_and_grad(pos, rho):
    """Energy and gradient of the Morse potential."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    x = rho * (1.0 - r_ij)
    ex = np.exp(x)
    e = ex * (ex - 2.0)
    energy = float(np.sum(e))
    # dE/dr_ij = rho * (1 - 2 ex) * ex
    # force on i from j: -dE/dr * (r_i - r_j)/r
    de_dr = rho * (1.0 - 2.0 * ex) * ex  # shape (n(n-1)/2,)
    grad = np.zeros_like(pos)
    # Build pair index arrays
    ii, jj = iu
    rij_vec = diff[ii, jj]  # pos[i] - pos[j]
    # contribution to grad[i] = +de_dr * rij_vec / r
    # contribution to grad[j] = -de_dr * rij_vec / r
    inv_r = 1.0 / r_ij
    f = (de_dr * inv_r)[:, None]  # (pairs, 1)
    np.add.at(grad, ii, f * rij_vec)
    np.add.at(grad, jj, -f * rij_vec)
    return energy, grad


def _is_valid(pos, min_dist=0.5):
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    return float(np.min(r)) >= min_dist


def _local_minimize(pos, rho, deadline):
    """Local minimization using L-BFGS-B on the gradient; fall back to relax."""
    pos = pos.astype(np.float64)

    def fun_g(x):
        if time.time() > deadline:
            return 1e30, np.zeros_like(x)
        e, g = _morse_energy_and_grad(x.reshape(-1, 3), rho)
        return e, g

    try:
        res = minimize(fun_g, pos.ravel(), method="L-BFGS-B",
                       options={"maxiter": 600, "ftol": 1e-12, "gtol": 1e-8,
                                "maxfun": 4000})
        if res.success or res.fun < _morse_energy(pos, rho):
            newpos = res.x.reshape(-1, 3)
            if _is_valid(newpos):
                return newpos
    except Exception:
        pass
    # fallback to provided relax
    try:
        e, newpos = relax(pos, "morse", rho)
        if _is_valid(newpos):
            return newpos
    except Exception:
        pass
    return pos


def _fcc_cluster(n, a=1.0):
    """Generate n points on an FCC lattice, centered."""
    pts = []
    # FCC basis offsets
    basis = [
        (0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5),
    ]
    # generate lattice points in a cube
    R = int(np.ceil((n / 4.0) ** (1.0 / 3.0))) + 1
    coords = []
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                for (dx, dy, dz) in basis:
                    coords.append((a * (i + dx), a * (j + dy), a * (k + dz)))
    coords = np.array(coords)
    # sort by distance from origin, take n
    d = np.linalg.norm(coords, axis=1)
    order = np.argsort(d)
    pts = coords[order[:n]]
    # center
    pts = pts - pts.mean(axis=0)
    return pts


def _bcc_cluster(n, a=1.0):
    """Generate n points on a BCC lattice, centered."""
    basis = [(0, 0, 0), (0.5, 0.5, 0.5)]
    R = int(np.ceil((n / 2.0) ** (1.0 / 3.0))) + 1
    coords = []
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                for (dx, dy, dz) in basis:
                    coords.append((a * (i + dx), a * (j + dy), a * (k + dz)))
    coords = np.array(coords)
    d = np.linalg.norm(coords, axis=1)
    order = np.argsort(d)
    pts = coords[order[:n]]
    pts = pts - pts.mean(axis=0)
    return pts


def _icosahedral(n):
    """Icosahedral shell construction (Mackay-like), centered."""
    # icosahedral vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
        [1, phi, 0],
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    pts = [np.zeros(3)]  # center
    shells = [1.0, np.sqrt(2.0 + 2.0 / phi), 2.0]  # radii for shells 1,2,3
    # shell 1: 12 vertices
    for v in verts:
        pts.append(shells[0] * v)
    if len(pts) >= n:
        return np.array(pts[:n])
    # shell 2: 30 points (edge midpoints of icosahedron scaled)
    # generate by rotating basis vectors
    # Use icosahedral symmetry: take 12 verts, 30 edges
    # Approximate: place 30 points as combinations
    # Simpler: generate 30 points as normalized (a,b,c) with small integers
    shell2 = []
    # icosahedral 30-point set: permutations of (0, phi, 1) scaled
    for perm in [(0, phi, 1), (0, -phi, 1), (0, phi, -1), (0, -phi, -1),
                 (phi, 1, 0), (-phi, 1, 0), (phi, -1, 0), (-phi, -1, 0),
                 (1, 0, phi), (-1, 0, phi), (1, 0, -phi), (-1, 0, -phi),
                 (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)]:
        p = np.array(perm, dtype=float)
        p = p / np.linalg.norm(p)
        shell2.append(shells[1] * p)
    # that's 16, need 30; add more via symmetry
    # For simplicity, use 30 points from a known set: (±1, ±phi, 0) perms etc.
    # We'll just take what we have and pad with shell 3 if needed.
    pts.extend(shell2)
    if len(pts) >= n:
        return np.array(pts[:n])
    # shell 3: 20 points (face centers)
    # face centers of icosahedron: normalize sums of 3 adjacent verts
    # approximate with 20 points
    shell3 = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                p = verts[i] + verts[j] + verts[k]
                if np.linalg.norm(p) > 1e-6:
                    p = p / np.linalg.norm(p)
                    shell3.append(shells[2] * p)
    pts.extend(shell3)
    pts = np.array(pts)
    if len(pts) < n:
        # pad with random near surface
        rng = np.random.default_rng(0)
        while len(pts) < n:
            d = rng.normal(size=3)
            d = d / np.linalg.norm(d)
            pts = np.vstack([pts, shells[2] * d * rng.uniform(0.9, 1.1)])
    return pts[:n]


def _spherical_shell(n):
    """Fibonacci sphere, centered, radius chosen for packing."""
    rng = np.random.default_rng(1)
    if n == 1:
        return np.zeros((1, 3))
    radius = 0.5 * n ** (1.0 / 3.0) + 0.5
    pts = [np.zeros(3)]
    # Fibonacci sphere for the rest
    m = n - 1
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    for i in range(m):
        z = 1.0 - 2.0 * (i + 0.5) / m
        r = np.sqrt(max(0.0, 1.0 - z * z))
        theta = 2.0 * np.pi * i / golden
        pts.append(radius * np.array([r * np.cos(theta), r * np.sin(theta), z]))
    pts = np.array(pts)
    return pts - pts.mean(axis=0)


def _random_compact(n, spacing=1.0, seed=0):
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    points = []
    tries = 0
    while len(points) < n and tries < 20000:
        tries += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
            points.append(p)
    # if not enough, fill randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def construct(instance):
    """Basin hopping over structured seeds with deadline guard."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 3.0))
    start = time.time()
    deadline = start + _DEADLINE

    # Build candidate seed generators
    seeds = []
    try:
        seeds.append(_fcc_cluster(n))
    except Exception:
        pass
    try:
        seeds.append(_bcc_cluster(n))
    except Exception:
        pass
    try:
        seeds.append(_icosahedral(n))
    except Exception:
        pass
    try:
        seeds.append(_spherical_shell(n))
    except Exception:
        pass
    for s in range(6):
        try:
            seeds.append(_random_compact(n, seed=s))
        except Exception:
            pass

    best_pos = None
    best_energy = np.inf

    def consider(pos):
        nonlocal best_pos, best_energy
        if not _is_valid(pos):
            return
        e = _morse_energy(pos, rho)
        if e < best_energy:
            best_energy = e
            best_pos = pos.copy()

    # Phase 1: relax each seed
    for idx, seed in enumerate(seeds):
        if time.time() > deadline - 30.0:
            break
        try:
            pos = _local_minimize(seed, rho, deadline)
            consider(pos)
        except Exception:
            pass

    # Phase 2: basin hopping from the best and a few good seeds
    # Collect the top few relaxed seeds
    candidates = []
    for idx, seed in enumerate(seeds):
        if time.time() > deadline - 60.0:
            break
        try:
            pos = _local_minimize(seed, rho, deadline)
            e = _morse_energy(pos, rho)
            if _is_valid(pos):
                candidates.append((e, pos))
        except Exception:
            pass
    candidates.sort(key=lambda x: x[0])

    # Basin hopping
    perturb_scale = 0.5
    rng = np.random.default_rng(42)
    bh_iter = 0
    max_bh = 200
    while time.time() < deadline - 45.0 and bh_iter < max_bh:
        bh_iter += 1
        # pick a starting point: best with some probability, else a random good one
        if best_pos is not None and rng.random() < 0.5:
            cur = best_pos.copy()
        elif candidates:
            cur = candidates[rng.integers(len(candidates))][1].copy()
        else:
            break
        # perturb
        noise = rng.normal(0.0, perturb_scale, size=cur.shape)
        perturbed = cur + noise
        if not _is_valid(perturbed):
            # shrink noise
            noise = rng.normal(0.0, perturb_scale * 0.3, size=cur.shape)
            perturbed = cur + noise
            if not _is_valid(perturbed):
                continue
        # relax
        try:
            newpos = _local_minimize(perturbed, rho, deadline)
        except Exception:
            continue
        e = _morse_energy(newpos, rho)
        if _is_valid(newpos):
            if e < best_energy:
                best_energy = e
                best_pos = newpos.copy()
                # anneal perturbation a bit
                perturb_scale = max(0.1, perturb_scale * 0.95)
            else:
                # occasionally keep going with larger perturbation
                if rng.random() < 0.1:
                    perturb_scale = min(1.5, perturb_scale * 1.1)

    # Phase 3: final high-precision polish on the best
    if best_pos is not None and time.time() < deadline - 10.0:
        try:
            polished = _local_minimize(best_pos, rho, deadline)
            e = _morse_energy(polished, rho)
            if _is_valid(polished) and e < best_energy:
                best_energy = e
                best_pos = polished.copy()
        except Exception:
            pass

    if best_pos is None:
        # last resort: a random compact cluster
        best_pos = _random_compact(n, seed=99)
    return best_pos
# EVOLVE-BLOCK-END
