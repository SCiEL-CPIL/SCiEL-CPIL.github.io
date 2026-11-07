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
    for i in range(n):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _make_fcc(n):
    """Generate FCC-like cluster by taking nearest neighbors from FCC lattice."""
    pts = []
    mc = int(np.ceil(n ** (1/3))) + 3
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
            # Scale to typical interatomic spacing ~0.72
            target_r = 0.72 * n**(1.0/3.0) * 0.55
            pts *= target_r / rmax
    return pts


def _make_bcc(n):
    """Generate BCC-like cluster."""
    pts = []
    mc = int(np.ceil(n ** (1/3))) + 3
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
            target_r = 0.72 * n**(1.0/3.0) * 0.55
            pts *= target_r / rmax
    return pts


def _make_icosahedral(n):
    """Generate icosahedral cluster with concentric shells."""
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of icosahedron
    verts = np.array([
        [1,0,phi],[-1,0,phi],[1,0,-phi],[-1,0,-phi],
        [0,phi,1],[0,-phi,1],[0,phi,-1],[0,-phi,-1],
        [phi,1,0],[-phi,1,0],[phi,-1,0],[-phi,-1,0]], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    # 6 face centers
    faces = [
        [0,1,8],[0,8,4],[0,4,10],[0,10,2],[0,2,6],
        [1,3,8],[3,8,10],[8,10,4],[4,10,6],[6,10,2],[2,6,12],
        [3,5,8],[5,8,10],[10,6,12],[6,12,14],[14,12,2]
    ]
    # Actually, let's use a simpler approach: generate points on icosahedral shells
    pts = [np.zeros(3)]
    
    # Shell 1: 12 vertices at r=1
    shell1 = verts.copy()
    for v in shell1:
        pts.append(1.0 * v)
    
    # Shell 2: 30 edge midpoints at r ~ 1.618 (for proper icosahedral geometry)
    # The 30 edges of an icosahedron
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 1.05:  # connected vertices
                edges.append((i, j))
    
    shell2 = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2
        mid /= np.linalg.norm(mid)
        shell2.append(mid)
    for v in shell2:
        pts.append(1.618 * v)
    
    # Shell 3: 20 face centers at r ~ 2.236
    # Find face centers from triangles
    face_centers = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d_ij = np.linalg.norm(verts[i] - verts[j])
                d_jk = np.linalg.norm(verts[j] - verts[k])
                d_ki = np.linalg.norm(verts[k] - verts[i])
                if d_ij < 1.05 and d_jk < 1.05 and d_ki < 1.05:
                    center = (verts[i] + verts[j] + verts[k]) / 3
                    center /= np.linalg.norm(center)
                    face_centers.append(center)
    
    # Deduplicate face centers
    unique_fc = []
    for fc in face_centers:
        if all(np.linalg.norm(fc - q) > 0.1 for q in unique_fc):
            unique_fc.append(fc)
    
    for v in unique_fc:
        pts.append(2.236 * v)
    
    # Shell 4: more vertices at larger radius
    shell4 = verts.copy()
    for v in shell4:
        pts.append(2.828 * v)
    
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    
    # Remove duplicates
    unique_pts = [pts[0]]
    for p in pts[1:]:
        if all(np.linalg.norm(p - q) > 0.05 for q in unique_pts):
            unique_pts.append(p)
    pts = np.array(unique_pts)
    
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    if n > 1:
        rmax = np.max(dists[order[:n]])
        if rmax > 0:
            target_r = 0.72 * n**(1.0/3.0) * 0.55
            pts *= target_r / rmax
    return pts


def _make_decahedral(n):
    """Generate decahedral (Mackay) cluster."""
    # Use icosahedral but with different shell ratios for decahedral
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1,0,phi],[-1,0,phi],[1,0,-phi],[-1,0,-phi],
        [0,phi,1],[0,-phi,1],[0,phi,-1],[0,-phi,-1],
        [phi,1,0],[-phi,1,0],[phi,-1,0],[-phi,-1,0]], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    pts = [np.zeros(3)]
    for v in verts:
        pts.append(1.0 * v)
    
    # Deahedral shells have different radii
    # Shell 2 at ~1.5 (compressed)
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 1.05:
                edges.append((i, j))
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2
        mid /= np.linalg.norm(mid)
        pts.append(1.5 * mid)
    
    # Shell 3
    face_centers = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d_ij = np.linalg.norm(verts[i] - verts[j])
                d_jk = np.linalg.norm(verts[j] - verts[k])
                d_ki = np.linalg.norm(verts[k] - verts[i])
                if d_ij < 1.05 and d_jk < 1.05 and d_ki < 1.05:
                    center = (verts[i] + verts[j] + verts[k]) / 3
                    center /= np.linalg.norm(center)
                    face_centers.append(center)
    unique_fc = []
    for fc in face_centers:
        if all(np.linalg.norm(fc - q) > 0.1 for q in unique_fc):
            unique_fc.append(fc)
    for v in unique_fc:
        pts.append(2.0 * v)
    
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    unique_pts = [pts[0]]
    for p in pts[1:]:
        if all(np.linalg.norm(p - q) > 0.05 for q in unique_pts):
            unique_pts.append(p)
    pts = np.array(unique_pts)
    
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    if n > 1:
        rmax = np.max(dists[order[:n]])
        if rmax > 0:
            target_r = 0.72 * n**(1.0/3.0) * 0.55
            pts *= target_r / rmax
    return pts


def _make_random_compact(n, rng, spacing=0.72):
    """Generate random compact cluster with minimum distance constraint."""
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    points = []
    attempts = 0
    min_dist = 0.55 * spacing
    while len(points) < n and attempts < 20000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
    if len(points) == n:
        return np.array(points)
    return None


def _perturb(positions, rng, scale):
    return positions + rng.normal(0, scale, positions.shape)


def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    rng = np.random.default_rng(n * 12345 + 7)
    best_energy = np.inf
    best_positions = None
    starts = []

    # Generate structured templates
    templates = []
    try:
        fcc = _make_fcc(n)
        if _is_valid(fcc):
            templates.append(("fcc", fcc))
    except Exception:
        pass

    try:
        bcc = _make_bcc(n)
        if _is_valid(bcc):
            templates.append(("bcc", bcc))
    except Exception:
        pass

    try:
        ico = _make_icosahedral(n)
        if _is_valid(ico):
            templates.append(("ico", ico))
    except Exception:
        pass

    try:
        deca = _make_decahedral(n)
        if _is_valid(deca):
            templates.append(("deca", deca))
    except Exception:
        pass

    # Add perturbations of each template
    perturbation_scales = [0.0, 0.03, 0.06, 0.1, 0.15, 0.2]
    for name, template in templates:
        for scale in perturbation_scales:
            if scale == 0.0:
                starts.append(template)
            else:
                pert = _perturb(template, rng, scale)
                if _is_valid(pert):
                    starts.append(pert)

    # Add random compact starts
    n_random = min(4, max(2, n // 20))
    for _ in range(n_random):
        rc = _make_random_compact(n, rng)
        if rc is not None and _is_valid(rc):
            starts.append(rc)

    # Relax all starts with time guard
    for i, start in enumerate(starts):
        if _time_left() < 8.0:
            break
        try:
            energy, positions = relax(start, potential, kind)
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    # If we have time, try a few more random starts
    if _time_left() > 20.0:
        for _ in range(3):
            if _time_left() < 10.0:
                break
            rc = _make_random_compact(n, rng)
            if rc is not None and _is_valid(rc):
                try:
                    energy, positions = relax(rc, potential, kind)
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
