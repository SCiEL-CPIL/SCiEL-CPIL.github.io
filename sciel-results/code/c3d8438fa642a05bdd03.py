"""
Dzugutov cluster global minima: polytetrahedral seeds + parallel basin hopping.
Uses Mackay icosahedral geometry, decahedral seeds, and concurrent optimization.
"""
import numpy as np
import time
import math
import concurrent.futures

try:
    from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
except ImportError:
    def relax(positions, potential):
        return 0.0, positions.copy()
    def dzugutov_energy_gradient(positions):
        return 0.0, np.zeros_like(positions)

_START_TIME = time.time()
_DEADLINE = 320.0  # 10s safety margin

def _remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _get_time_remaining():
    return max(0.0, _remaining())

def _rotmat(axis, angle):
    """Rotation matrix around axis by angle."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    C = 1 - c
    return np.array([
        [c + axis[0]**2*C, axis[0]*axis[1]*C - axis[2]*s, axis[0]*axis[2]*C + axis[1]*s],
        [axis[1]*axis[0]*C + axis[2]*s, c + axis[1]**2*C, axis[1]*axis[2]*C - axis[0]*s],
        [axis[2]*axis[0]*C - axis[1]*s, axis[2]*axis[1]*C + axis[0]*s, c + axis[2]**2*C]
    ])

def _icosahedron_verts(r=1.0):
    """12 vertices of regular icosahedron at radius r."""
    phi = (1 + math.sqrt(5)) / 2
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms * r

def _generate_mackay_icosahedron(n, rng, d0=2.05):
    """
    Generate Mackay icosahedral cluster with shells.
    Shell k has radius k*d0 and 10k^2+1 sites (approx).
    """
    pts = [np.zeros(3)]
    if n <= 1:
        return np.array(pts[:n])
    
    # Shell 1: 12 vertices
    verts = _icosahedron_verts(d0)
    for v in verts:
        if len(pts) < n:
            pts.append(v)
    
    if len(pts) >= n:
        return np.array(pts[:n])
    
    # Higher shells: use icosahedral symmetry operations on face centers
    # For simplicity, generate points on concentric spheres with icosahedral distortion
    shell = 1
    while len(pts) < n:
        shell += 1
        r = shell * d0
        # Number of sites in shell ~ 10*shell^2 + 1
        n_sites = min(10 * shell * shell + 1, n - len(pts))
        
        # Generate n_sites points on sphere with icosahedral symmetry bias
        # Use golden angle spiral with slight icosahedral perturbation
        golden = math.pi * (3 - math.sqrt(5))
        for i in range(n_sites):
            if len(pts) >= n:
                break
            # Spherical coords
            y = 1 - 2 * (i + 0.5) / n_sites
            rad = math.sqrt(max(0, 1 - y*y))
            theta = golden * i
            p = np.array([math.cos(theta)*rad, y, math.sin(theta)*rad]) * r
            pts.append(p)
    
    pts = np.array(pts[:n])
    pts -= pts.mean(axis=0)
    return pts

def _generate_decahedral(n, rng, d0=2.05):
    """
    Generate a decahedral (five-fold twinned) seed.
    Five icosahedral fragments sharing a common center.
    """
    pts = [np.zeros(3)]
    if n <= 1:
        return np.array(pts[:n])
    
    # 5-fold axis: z-axis
    # Each fragment: 12 icosahedron vertices rotated by 72 deg around z
    verts_template = _icosahedron_verts(d0)
    
    # Keep only upper-hemisphere-ish vertices for each fragment to avoid overlap
    # Actually, just rotate all 12 and deduplicate
    all_pts = []
    for k in range(5):
        angle = 2 * math.pi * k / 5
        R = _rotmat([0, 0, 1], angle)
        for v in verts_template:
            p = R @ v
            all_pts.append(p)
    
    # Sort by distance from center, keep closest n-1
    all_pts = np.array(all_pts)
    dists = np.linalg.norm(all_pts, axis=1)
    order = np.argsort(dists)
    
    for idx in order:
        if len(pts) >= n:
            break
        p = all_pts[idx]
        # Check min distance to existing
        if len(pts) > 0:
            dmin = np.min(np.linalg.norm(np.array(pts) - p, axis=1))
            if dmin < 0.55:
                continue
        pts.append(p)
    
    # Fill remaining with random compact points
    while len(pts) < n:
        p = rng.uniform(-d0*2, d0*2, 3)
        if len(pts) > 0:
            dmin = np.min(np.linalg.norm(np.array(pts) - p, axis=1))
            if dmin < 0.55:
                continue
        pts.append(p)
    
    pts = np.array(pts[:n])
    pts -= pts.mean(axis=0)
    return pts

def _generate_bimodal(n, rng, d0=2.05):
    """
    Two overlapping icosahedral cores.
    """
    if n < 30:
        return _generate_mackay_icosahedron(n, rng, d0)
    
    n1 = n // 2
    n2 = n - n1
    ico1 = _generate_mackay_icosahedron(n1, rng, d0)
    ico2 = _generate_mackay_icosahedron(n2, rng, d0)
    
    r1 = np.max(np.linalg.norm(ico1, axis=1)) if n1 > 1 else d0
    r2 = np.max(np.linalg.norm(ico2, axis=1)) if n2 > 1 else d0
    
    # Overlap by ~1 shell
    d = r1 + r2 - d0 * 0.8
    ico2[:, 0] += d
    
    pts = np.vstack([ico1, ico2])
    pts -= pts.mean(axis=0)
    return pts

def _generate_fcc_seed(n, rng, a=2.1):
    """Generate FCC-like compact seed."""
    L = int((4.0 * n) ** (1.0/3.0)) + 2
    points = []
    for i in range(-L//2, L//2 + 1):
        for j in range(-L//2, L//2 + 1):
            for k in range(-L//2, L//2 + 1):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([i+dx, j+dy, k+dz]) * a
                    if np.linalg.norm(p) < 12.0:
                        points.append(p)
    points = np.array(points)
    if len(points) == 0:
        return _generate_mackay_icosahedron(n, rng)
    if len(points) > n:
        dists = np.linalg.norm(points, axis=1)
        points = points[np.argsort(dists)[:n]]
    elif len(points) < n:
        while len(points) < n:
            p = rng.uniform(-5, 5, 3)
            if np.linalg.norm(p) < 5.0:
                points = np.vstack([points, [p]])
    points -= points.mean(axis=0)
    return points

def _generate_random_seed(n, rng):
    """Random compact cluster."""
    radius = 0.9 * n ** (1.0/3.0)
    pts = rng.uniform(-radius, radius, (n, 3))
    pts -= pts.mean(axis=0)
    return pts

def _basin_hop(positions, potential, rng, max_time=25.0, step0=0.4):
    """
    Basin hopping with adaptive step size.
    Returns (best_energy, best_positions).
    """
    t0 = time.time()
    best_e = float('inf')
    best_p = positions.copy()
    
    try:
        e, p = relax(positions, potential)
        if e < best_e:
            best_e, best_p = e, p
    except Exception:
        return best_e, best_p
    
    cur_p = best_p
    cur_e = best_e
    step = step0
    stall = 0
    
    while (time.time() - t0) < max_time and _get_time_remaining() > 8.0:
        n = len(cur_p)
        # Move random subset
        k = max(1, n // 4)
        idx = rng.choice(n, size=k, replace=False)
        
        pert = cur_p.copy()
        pert[idx] += rng.normal(0, step, (k, 3))
        
        try:
            e2, p2 = relax(pert, potential)
        except Exception:
            stall += 1
            continue
        
        if e2 < cur_e:
            cur_e, cur_p = e2, p2
            stall = 0
            # Occasionally shrink step for fine-tuning
            if rng.random() < 0.1:
                step = max(0.1, step * 0.9)
        else:
            stall += 1
            if stall > 8:
                step = min(1.5, step * 1.3)
                stall = 0
            elif stall > 15:
                step = min(2.0, step * 1.6)
                stall = 0
        
        if cur_e < best_e:
            best_e, best_p = cur_e, cur_p
    
    return best_e, best_p

def _worker(seed, potential, seed_id):
    """Worker for parallel basin hopping."""
    rng = np.random.default_rng(42 + seed_id * 1000)
    max_time = min(25.0, _get_time_remaining() - 5.0)
    if max_time < 3.0:
        # Quick relaxation only
        try:
            e, p = relax(seed, potential)
            return e, p
        except Exception:
            return float('inf'), seed.copy()
    return _basin_hop(seed, potential, rng, max_time=max_time)

def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    
    if _get_time_remaining() < 5.0:
        rng = np.random.default_rng(n)
        radius = 0.8 * n ** (1.0/3.0)
        return rng.uniform(-radius, radius, (n, 3))
    
    rng = np.random.default_rng(n)
    
    # Generate diverse seeds
    seeds = []
    seeds.append(_generate_mackay_icosahedron(n, rng))
    seeds.append(_generate_decahedral(n, rng))
    seeds.append(_generate_bimodal(n, rng))
    seeds.append(_generate_fcc_seed(n, rng))
    seeds.append(_generate_random_seed(n, rng))
    # Deformed icosahedron
    ico2 = _generate_mackay_icosahedron(n, rng)
    ico2 += rng.normal(0, 0.25, ico2.shape)
    seeds.append(ico2)
    
    # Run parallel basin hopping
    best_e = float('inf')
    best_p = seeds[0]
    
    max_workers = min(4, len(seeds))
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, seed in enumerate(seeds):
            if _get_time_remaining() < 10.0:
                break
            f = executor.submit(_worker, seed, potential, i)
            futures.append(f)
        
        for f in futures:
            try:
                e, p = f.result(timeout=max(1.0, _get_time_remaining() - 2.0))
                if e < best_e:
                    best_e, best_p = e, p
            except Exception:
                pass
    
    # Final polish on best structure
    if _get_time_remaining() > 10.0:
        try:
            e, p = _basin_hop(best_p, potential, rng, max_time=_get_time_remaining() - 5.0, step0=0.3)
            if e < best_e:
                best_e, best_p = e, p
        except Exception:
            pass
    
    return best_p
