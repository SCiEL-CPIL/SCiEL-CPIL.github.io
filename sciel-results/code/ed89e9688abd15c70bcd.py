"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START = time.time()
_DEADLINE = 295.0


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _is_valid(positions, min_dist=0.35):
    """Vectorized validity: finite coords, no pair closer than min_dist."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    # Vectorized pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    return bool(np.all(dists >= min_dist))


def _make_fcc(n):
    """Compact FCC chunk of n atoms."""
    pts = []
    mc = int(np.ceil(n ** (1.0 / 3.0))) + 2
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                pts.append([i, j, k])
    pts = np.array(pts, dtype=float)
    # Keep only FCC sites: i+j+k even
    mask = (pts[:, 0].astype(int) + pts[:, 1].astype(int) + pts[:, 2].astype(int)) % 2 == 0
    pts = pts[mask]
    # Sort by distance from center, take n closest
    if len(pts) >= n:
        d = np.linalg.norm(pts, axis=1)
        idx = np.argsort(d)[:n]
        pts = pts[idx]
    else:
        # Pad with random points near the cluster
        while len(pts) < n:
            p = np.random.randn(3) * 0.5
            pts = np.vstack([pts, p])
    # Center
    pts -= pts.mean(axis=0)
    return pts[:n]


def _make_bcc(n):
    """Compact BCC chunk of n atoms."""
    pts = []
    mc = int(np.ceil(n ** (1.0 / 3.0))) + 2
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                pts.append([i, j, k])
                pts.append([i + 0.5, j + 0.5, k + 0.5])
    pts = np.array(pts, dtype=float)
    if len(pts) >= n:
        d = np.linalg.norm(pts, axis=1)
        idx = np.argsort(d)[:n]
        pts = pts[idx]
    else:
        while len(pts) < n:
            p = np.random.randn(3) * 0.5
            pts = np.vstack([pts, p])
    pts -= pts.mean(axis=0)
    return pts[:n]


def _make_mackay(n):
    """Mackay icosahedral shell structure, filled to n atoms."""
    # Icosahedron vertices (12), then 30 edge midpoints, then 20 face centers, etc.
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    pts = [verts]
    # Edge midpoints (30)
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if 1.5 < d < 1.6:  # nearest-neighbor edges of icosahedron
                edges.append((i, j))
    edge_pts = np.array([(verts[i] + verts[j]) / 2 for i, j in edges])
    edge_pts /= np.linalg.norm(edge_pts[0])
    pts.append(edge_pts)
    # Face centers (20) - use golden ratio construction
    face_pts = []
    for i in range(12):
        for j in range(12):
            if i == j:
                continue
            for k in range(12):
                if k == i or k == j:
                    continue
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[i] - verts[k])
                d3 = np.linalg.norm(verts[j] - verts[k])
                if 1.5 < d1 < 1.6 and 1.5 < d2 < 1.6 and 1.5 < d3 < 1.6:
                    fc = (verts[i] + verts[j] + verts[k]) / 3
                    fc /= np.linalg.norm(fc)
                    # Check not already added
                    if all(np.linalg.norm(fc - existing) > 0.01 for existing in face_pts):
                        face_pts.append(fc)
    if face_pts:
        pts.append(np.array(face_pts))
    # Flatten and take n
    all_pts = np.vstack(pts)
    if len(all_pts) >= n:
        d = np.linalg.norm(all_pts, axis=1)
        idx = np.argsort(d)[:n]
        all_pts = all_pts[idx]
    else:
        while len(all_pts) < n:
            p = np.random.randn(3)
            p /= np.linalg.norm(p)
            all_pts = np.vstack([all_pts, p])
    all_pts -= all_pts.mean(axis=0)
    return all_pts[:n]


def _make_deformed(template, scale=0.15):
    """Apply random affine shear to a template, then fix min-distance violations."""
    pts = template.copy()
    # Random shear
    A = np.eye(3)
    A[0, 1] += np.random.randn() * scale
    A[0, 2] += np.random.randn() * scale
    A[1, 0] += np.random.randn() * scale
    A[1, 2] += np.random.randn() * scale
    A[2, 0] += np.random.randn() * scale
    A[2, 1] += np.random.randn() * scale
    pts = pts @ A.T
    pts -= pts.mean(axis=0)
    # Fix collisions: push apart any pair closer than 0.36
    min_d = 0.36
    for _ in range(20):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        mask = dists < min_d
        if not np.any(mask):
            break
        # Push apart
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if mask[i, j]:
                    d = pts[i] - pts[j]
                    nd = np.linalg.norm(d)
                    if nd < 1e-8:
                        d = np.array([1.0, 0, 0])
                        nd = 1.0
                    push = d / nd * (min_d - nd) * 0.5
                    pts[i] += push
                    pts[j] -= push
    return pts


def _make_random_compact(n, spacing=0.72):
    """Random compact cluster with vectorized collision check."""
    rng = np.random.default_rng()
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    min_d = 0.7 * spacing
    attempts = 0
    while len(points) < n and attempts < n * 500:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if not points:
            points.append(p)
            continue
        arr = np.array(points)
        d = np.linalg.norm(arr - p, axis=1)
        if np.all(d >= min_d):
            points.append(p)
    if len(points) < n:
        # Fallback: fill with FCC
        pts = _make_fcc(n)
        return pts
    pts = np.array(points)
    pts -= pts.mean(axis=0)
    return pts


def _basin_hop(positions, kind, n_iter_max=30, T0=0.5, T_final=0.01):
    """Basin hopping: perturb, relax, accept/reject with SA."""
    best_pos = positions.copy()
    best_energy = None
    try:
        best_energy, _ = sutton_chen_energy_gradient(positions, kind)
    except Exception:
        best_energy = np.inf

    T = T0
    cooling = (T_final / T0) ** (1.0 / max(n_iter_max, 1))
    rng = np.random.default_rng()

    for it in range(n_iter_max):
        if _time_left() < 5.0:
            break
        # Perturb
        scale = 0.15 + 0.1 * (1 - it / n_iter_max)
        perturbed = positions + rng.normal(0, scale, positions.shape)
        # Relax
        try:
            e, pos = relax(perturbed, 'sutton_chen', kind)
        except Exception:
            continue
        if not np.isfinite(e) or not _is_valid(pos):
            continue
        # Accept/reject
        if best_energy is None or e < best_energy or rng.random() < np.exp(-(e - best_energy) / max(T, 1e-8)):
            positions = pos
            if e < best_energy:
                best_energy = e
                best_pos = pos.copy()
        T *= cooling

    return best_pos, best_energy


def construct(instance):
    """Portfolio basin-hopping search for Sutton-Chen cluster global minima."""
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    rng = np.random.default_rng(n * 13 + 7)

    # Generate diverse starting templates
    templates = []
    templates.append(_make_fcc(n))
    templates.append(_make_bcc(n))
    templates.append(_make_mackay(n))
    templates.append(_make_random_compact(n))
    # Deformed variants
    templates.append(_make_deformed(_make_fcc(n), scale=0.2))
    templates.append(_make_deformed(_make_bcc(n), scale=0.2))
    templates.append(_make_deformed(_make_fcc(n), scale=0.35))

    best_energy = np.inf
    best_pos = None

    for tmpl in templates:
        if _time_left() < 10.0:
            break
        # First: direct relax
        try:
            e, pos = relax(tmpl, 'sutton_chen', kind)
        except Exception:
            continue
        if np.isfinite(e) and _is_valid(pos) and e < best_energy:
            best_energy = e
            best_pos = pos.copy()

        # Then: basin hopping from this template
        if _time_left() < 15.0:
            break
        bh_pos, bh_energy = _basin_hop(tmpl, kind, n_iter_max=20)
        if bh_energy is not None and np.isfinite(bh_energy) and _is_valid(bh_pos) and bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos.copy()

    # If we still have time, do additional basin hops from the best
    if best_pos is not None and _time_left() > 20.0:
        for _ in range(3):
            if _time_left() < 15.0:
                break
            bh_pos, bh_energy = _basin_hop(best_pos, kind, n_iter_max=15)
            if bh_energy is not None and np.isfinite(bh_energy) and _is_valid(bh_pos) and bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos.copy()

    if best_pos is None:
        # Last-resort fallback
        best_pos = _make_fcc(n)
        try:
            _, best_pos = relax(best_pos, 'sutton_chen', kind)
        except Exception:
            pass

    # Final validity check
    if not _is_valid(best_pos):
        # Try to fix
        best_pos = _make_fcc(n)
        try:
            _, best_pos = relax(best_pos, 'sutton_chen', kind)
        except Exception:
            pass

    return best_pos
# EVOLVE-BLOCK-END
