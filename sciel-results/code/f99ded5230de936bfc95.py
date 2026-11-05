"""Morse cluster global minima via structured seeds + diverse local relaxations."""
# EVOLVE-BLOCK-START
import time
import numpy as np
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


def _icosahedron_shell():
    """12 vertices of icosahedron scaled so nearest-neighbor distance = 1."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    # nearest neighbor distance in this construction is 2
    v /= 2.0
    return v


def _fcc_lattice(n, spacing=1.0):
    """Generate n atoms on an FCC-like lattice, centered."""
    # FCC conventional cell: atoms at (0,0,0),(1,0,0),(0,1,0),(0,0,1),
    # (1,1,0),(1,0,1),(0,1,1),(1,1,1) in units of a/2 where a=sqrt(2)*spacing
    # Simplest: place on FCC primitive vectors
    a = spacing  # nearest neighbor = a
    v1 = np.array([0, a/np.sqrt(2), a/np.sqrt(2)])
    v2 = np.array([a/np.sqrt(2), 0, a/np.sqrt(2)])
    v3 = np.array([a/np.sqrt(2), a/np.sqrt(2), 0])
    # Enumerate lattice points
    pts = []
    side = int(np.ceil(n ** (1/3))) + 2
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                p = i*v1 + j*v2 + k*v3
                pts.append(p)
    pts = np.array(pts)
    # Center
    pts -= pts.mean(axis=0)
    # Sort by radius and take n
    r = np.linalg.norm(pts, axis=1)
    idx = np.argsort(r)[:n]
    return pts[idx]


def _hcp_lattice(n, spacing=1.0):
    """Generate n atoms on an HCP-like lattice."""
    a = spacing
    # HCP basis in hexagonal coords
    c = a * np.sqrt(8.0/3.0)
    pts = []
    side = int(np.ceil(n ** (1/3))) + 2
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                # A-layer
                x = a * (i + j/2.0)
                y = a * (np.sqrt(3)/2.0) * j
                z = c * k
                pts.append([x, y, z])
                # B-layer
                x2 = a * (i + j/2.0 + 1.0/np.sqrt(3.0))
                y2 = a * (np.sqrt(3)/2.0) * j - a * (1.0/np.sqrt(6.0))
                z2 = c * (k + 0.5)
                pts.append([x2, y2, z2])
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    r = np.linalg.norm(pts, axis=1)
    idx = np.argsort(r)[:n]
    return pts[idx]


def _decahedral_cluster(n, spacing=1.0):
    """5-fold symmetric decahedral cluster (Mackay-like)."""
    # 5-fold axis along z; 5-fold positions
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # Center
    pts = [np.array([0.0, 0.0, 0.0])]
    # First shell: 12 atoms (icosahedral)
    ico = _icosahedron_shell() * spacing
    pts.extend(list(ico))
    # Second shell: 42 atoms (Mackay 55)
    # Approximate with points on a slightly larger icosahedron + face centers
    ico2 = _icosahedron_shell() * spacing * 1.9
    pts.extend(list(ico2))
    # Add mid-edge points
    for i in range(12):
        for j in range(i+1, 12):
            mid = (ico[i] + ico2[j]) / 2.0
            pts.append(mid)
    pts = np.array(pts)
    # Deduplicate and take n
    if len(pts) > n:
        r = np.linalg.norm(pts, axis=1)
        idx = np.argsort(r)[:n]
        pts = pts[idx]
    elif len(pts) < n:
        # Fill with random close-packed
        rng = np.random.default_rng(42)
        while len(pts) < n:
            p = rng.uniform(-2.0, 2.0, 3) * spacing * n**(1/3) * 0.5
            if all(np.linalg.norm(p - q) >= 0.8 * spacing for q in pts):
                pts = np.vstack([pts, p]) if isinstance(pts, np.ndarray) else np.array(pts + [p])
        pts = np.array(pts)
    return pts


def _random_cluster(n, spacing=1.0, seed=0):
    """Random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    max_tries = 50000
    tries = 0
    while len(points) < n and tries < max_tries:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < 0.85 * spacing:
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


def _perturb_cluster(pos, spacing=1.0, seed=0, scale=0.3):
    """Add random perturbation to a cluster."""
    rng = np.random.default_rng(seed)
    return pos + rng.normal(0, scale * spacing, pos.shape)


def _ensure_min_distance(pos, min_dist=0.5):
    """Nudge atoms apart if any pair is closer than min_dist."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(50):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                d = pos[i] - pos[j]
                r = np.linalg.norm(d)
                if r < min_dist and r > 1e-10:
                    push = (d / r) * (min_dist - r) * 0.5
                    pos[i] += push
                    pos[j] -= push
                    moved = True
                elif r < min_dist:
                    # Random nudge
                    pos[i] += np.random.default_rng(i*1000+j).uniform(-0.1, 0.1, 3)
                    pos[j] -= np.random.default_rng(j*1000+i).uniform(-0.1, 0.1, 3)
                    moved = True
        if not moved:
            break
    return pos


def construct(instance):
    """Find low-energy Morse cluster via structured seeds + multi-start relaxation."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    potential = instance["potential"]
    t0 = time.time()
    deadline = t0 + 280.0  # safety margin

    # Generate diverse starting configurations
    candidates = []

    # 1. Icosahedral shell-based
    try:
        ico = _icosahedron_shell() * 1.0
        if n <= 12:
            # Take subset
            if n == 1:
                candidates.append(np.array([[0.0, 0.0, 0.0]]))
            else:
                r = np.linalg.norm(ico, axis=1)
                idx = np.argsort(r)[:n]
                candidates.append(ico[idx])
        else:
            # Multiple shells
            pts = [np.array([0.0, 0.0, 0.0])]
            for shell in range(1, 20):
                shell_pts = _icosahedron_shell() * shell * 0.55
                pts.extend(list(shell_pts))
                if len(pts) >= n:
                    break
            pts = np.array(pts[:n]) if len(pts) >= n else np.array(pts)
            if len(pts) < n:
                # Fill randomly
                rng = np.random.default_rng(1)
                while len(pts) < n:
                    p = rng.uniform(-3, 3, 3)
                    pts = np.vstack([pts, p])
            candidates.append(pts)
    except Exception:
        pass

    # 2. FCC fragment
    try:
        fcc = _fcc_lattice(n, 1.0)
        candidates.append(fcc)
    except Exception:
        pass

    # 3. HCP fragment
    try:
        hcp = _hcp_lattice(n, 1.0)
        candidates.append(hcp)
    except Exception:
        pass

    # 4. Decahedral
    try:
        dec = _decahedral_cluster(n, 1.0)
        candidates.append(dec)
    except Exception:
        pass

    # 5-12. Random clusters with different seeds
    for s in range(8):
        try:
            rc = _random_cluster(n, 1.0, seed=100 + s)
            candidates.append(rc)
        except Exception:
            pass

    # 13-20. Perturbed versions of best geometric seeds
    for s in range(4):
        try:
            if len(candidates) > 0:
                base = candidates[s % min(len(candidates), 4)]
                pert = _perturb_cluster(base, 1.0, seed=200 + s, scale=0.2 + 0.1*s)
                candidates.append(pert)
        except Exception:
            pass

    # Ensure all candidates have exactly n atoms and min distance
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

    # Multi-start relaxation
    best_energy = np.inf
    best_pos = valid_candidates[0] if valid_candidates else np.zeros((n, 3))

    for idx, start in enumerate(valid_candidates):
        if time.time() > deadline:
            break
        try:
            energy, positions = relax(start, potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_pos = positions.copy()
        except Exception:
            continue

    # If we have time, do a second pass with more perturbations of the best
    if time.time() < deadline - 30:
        for s in range(6):
            if time.time() > deadline:
                break
            try:
                pert = _perturb_cluster(best_pos, 1.0, seed=500 + s, scale=0.15 + 0.05*s)
                pert = _ensure_min_distance(pert, 0.5)
                energy, positions = relax(pert, potential, rho)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = positions.copy()
            except Exception:
                continue

    # Final polish: one more relax from best with tighter settings if available
    if time.time() < deadline - 5:
        try:
            energy, positions = relax(best_pos, potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_pos = positions.copy()
        except Exception:
            pass

    return np.asarray(best_pos, dtype=float)
# EVOLVE-BLOCK-END
