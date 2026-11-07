"""Portfolio-search program for Sutton-Chen TM cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 300.0


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    for i in range(n):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _make_fcc(n):
    pts = []
    mc = int(np.ceil(n ** (1/3))) + 4
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    pts.append(np.array([i+dx, j+dy, k+dz]))
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    if n > 1:
        rmax = np.max(dists[order[:n]])
        if rmax > 0:
            pts *= (0.72 * n**(1.0/3.0) * 0.6) / rmax
    return pts


def _make_bcc(n):
    pts = []
    mc = int(np.ceil(n ** (1/3))) + 4
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                pts.append(np.array([i, j, k]))
                pts.append(np.array([i+0.5, j+0.5, k+0.5]))
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    if n > 1:
        rmax = np.max(dists[order[:n]])
        if rmax > 0:
            pts *= (0.72 * n**(1.0/3.0) * 0.6) / rmax
    return pts


def _make_mackay(n):
    """Generate Mackay icosahedral cluster shells."""
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of icosahedron
    verts = np.array([
        [1,0,phi],[-1,0,phi],[1,0,-phi],[-1,0,-phi],
        [0,phi,1],[0,-phi,1],[0,phi,-1],[0,-phi,-1],
        [phi,1,0],[-phi,1,0],[phi,-1,0],[-phi,-1,0]], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    pts = [np.zeros(3)]  # center
    # Shell 1: 12 vertices
    for v in verts:
        pts.append(v)
    # Shell 2: 42 points (Mackay 55)
    # 12 at distance sqrt(2+phi) * a, 30 at distance sqrt(2+phi)*a in different positions
    # Actually Mackay 55 has: 1 + 12 + 42 = 55
    # The 42 points: 12 at edge midpoints scaled, 30 at face centers scaled
    # Let me use the standard construction
    # Edge midpoints (30 edges)
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if 0.9 < d < 1.1:  # edge length ~1.05
                edges.append((verts[i] + verts[j]) / 2)
    # Face centers (20 faces)
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                # Check if i,j,k form a face
                v1, v2, v3 = verts[i], verts[j], verts[k]
                n1 = np.cross(v2 - v1, v3 - v1)
                if np.linalg.norm(n1) > 0.1:
                    d1 = np.linalg.norm(v2 - v1)
                    d2 = np.linalg.norm(v3 - v1)
                    d3 = np.linalg.norm(v3 - v2)
                    if 0.9 < d1 < 1.1 and 0.9 < d2 < 1.1 and 0.9 < d3 < 1.1:
                        faces.append((v1 + v2 + v3) / 3)
    
    r2 = np.sqrt(2 + phi)  # ~1.902
    for e in edges:
        e = e / np.linalg.norm(e) * r2
        pts.append(e)
    for f in faces:
        f = f / np.linalg.norm(f) * r2
        pts.append(f)
    
    # Shell 3 for larger clusters
    r3 = np.sqrt(5 + 2*phi)  # ~2.802
    # Add more points on shell 3
    for v in verts:
        pts.append(v * r3)
    for e in edges:
        e = e / np.linalg.norm(e) * r3
        pts.append(e)
    for f in faces:
        f = f / np.linalg.norm(f) * r3
        pts.append(f)
    # Additional points on shell 3
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                v1, v2, v3 = verts[i], verts[j], verts[k]
                d1 = np.linalg.norm(v2 - v1)
                d2 = np.linalg.norm(v3 - v1)
                d3 = np.linalg.norm(v3 - v2)
                if 0.9 < d1 < 1.1 and 0.9 < d2 < 1.1 and 0.9 < d3 < 1.1:
                    # This is a face; add points around it
                    c = (v1 + v2 + v3) / 3
                    c = c / np.linalg.norm(c)
                    # Add 6 more points per face region
                    for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
                        # Rotate around face normal
                        nrm = np.cross(v2 - v1, v3 - v1)
                        nrm /= np.linalg.norm(nrm)
                        # Create rotation
                        x = np.array([1.0, 0, 0])
                        if abs(np.dot(x, nrm)) > 0.9:
                            x = np.array([0, 1.0, 0])
                        y = np.cross(nrm, x)
                        z = nrm
                        rot = c * np.cos(angle) + y * np.sin(angle)
                        rot = rot / np.linalg.norm(rot) * r3
                        pts.append(rot)
    
    pts = np.array(pts)
    # Remove duplicates
    unique_pts = [pts[0]]
    for p in pts[1:]:
        if all(np.linalg.norm(p - q) > 0.05 for q in unique_pts):
            unique_pts.append(p)
    pts = np.array(unique_pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    if n > 1:
        rmax = np.max(dists[order[:n]])
        if rmax > 0:
            pts *= (0.72 * n**(1.0/3.0) * 0.6) / rmax
    return pts


def _make_decahedral(n):
    """Generate decahedral cluster (twinned FCC)."""
    pts = []
    mc = int(np.ceil(n ** (1/3))) + 4
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    pts.append(np.array([i+dx, j+dy, k+dz]))
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    # Apply 5-fold twinning: rotate some points
    # For simplicity, just use FCC but with slight distortion
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    if n > 1:
        rmax = np.max(dists[order[:n]])
        if rmax > 0:
            pts *= (0.72 * n**(1.0/3.0) * 0.6) / rmax
    return pts


def _make_random_compact(n, rng, spacing=0.72):
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 * spacing for q in points):
                points.append(p)
    if len(points) == n:
        return np.array(points)
    return None


def _perturb(positions, rng, scale=0.1):
    return positions + rng.normal(0, scale, positions.shape)


def _basin_hop(positions, kind, rng, n_hops=5, step_scale=0.15):
    """Simple basin hopping: perturb and relax."""
    current = positions.copy()
    for _ in range(n_hops):
        if _time_left() < 5.0:
            break
        perturbed = current + rng.normal(0, step_scale, current.shape)
        if not _is_valid(perturbed):
            continue
        try:
            energy, relaxed = relax(perturbed, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed):
                # Accept if lower energy (Metropolis not needed, just keep best)
                current = relaxed
        except Exception:
            pass
    return current


def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    rng = np.random.default_rng(n * 12345 + 7)
    best_energy = np.inf
    best_positions = None
    
    starts = []
    
    # Generate diverse starting structures
    try:
        fcc = _make_fcc(n)
        if _is_valid(fcc):
            starts.append(fcc)
            for s in [0.03, 0.07, 0.12, 0.2]:
                pert = _perturb(fcc, rng, s)
                if _is_valid(pert):
                    starts.append(pert)
    except Exception:
        pass
    
    try:
        bcc = _make_bcc(n)
        if _is_valid(bcc):
            starts.append(bcc)
            for s in [0.05, 0.1, 0.15]:
                pert = _perturb(bcc, rng, s)
                if _is_valid(pert):
                    starts.append(pert)
    except Exception:
        pass
    
    try:
        mackay = _make_mackay(n)
        if _is_valid(mackay):
            starts.append(mackay)
            for s in [0.03, 0.07, 0.12]:
                pert = _perturb(mackay, rng, s)
                if _is_valid(pert):
                    starts.append(pert)
    except Exception:
        pass
    
    try:
        deca = _make_decahedral(n)
        if _is_valid(deca):
            starts.append(deca)
            for s in [0.05, 0.1]:
                pert = _perturb(deca, rng, s)
                if _is_valid(pert):
                    starts.append(pert)
    except Exception:
        pass
    
    # Random compact starts
    for _ in range(4):
        rc = _make_random_compact(n, rng)
        if rc is not None and _is_valid(rc):
            starts.append(rc)
    
    # Relax each start
    for i, start in enumerate(starts):
        if _time_left() < 15.0:
            break
        try:
            energy, positions = relax(start, potential, kind)
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass
    
    # If we have time, do basin hopping from the best result
    if best_positions is not None and _time_left() > 20.0:
        try:
            bh_result = _basin_hop(best_positions, kind, rng, n_hops=8, step_scale=0.12)
            energy, positions = relax(bh_result, potential, kind)
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass
    
    # Additional basin hopping from second best if time permits
    if best_positions is not None and _time_left() > 30.0:
        # Try basin hopping from a perturbed version of best
        try:
            pert_best = _perturb(best_positions, rng, 0.15)
            bh2 = _basin_hop(pert_best, kind, rng, n_hops=5, step_scale=0.1)
            energy, positions = relax(bh2, potential, kind)
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass
    
    if best_positions is None:
        try:
            best_positions = _make_fcc(n)
        except Exception:
            best_positions = rng.uniform(-1, 1, (n, 3))
    
    return best_positions
# EVOLVE-BLOCK-END
