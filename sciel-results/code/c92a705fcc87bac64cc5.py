"""
Dzugutov cluster global minima search.
Uses icosahedral shell construction, FCC, random, and Markov-chain-based
diverse initial configurations, relaxed in parallel with a strict deadline.
"""
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE_OFFSET = 290.0  # Leave 40s margin for 330s kill (300s soft limit)

def _get_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME + _DEADLINE_OFFSET

def _energy(positions):
    """Compute Dzugutov energy."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(positions)
    E = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < a:
                E += A * (r**(-m) - B) * np.exp(c / (r - a))
            elif r < b:
                E += B * np.exp(d / (r - b))
    return E

def _generate_icosahedral(n, rng, shell_radius=2.05):
    """
    Generate an icosahedral cluster with multiple shells.
    Shell 0: 1 atom (center)
    Shell 1: 12 atoms (icosahedron vertices) at radius r1
    Shell 2: 30 atoms (edge midpoints) at radius ~r1*0.85
    Shell 3: 20 atoms (face centers) at radius ~r1*0.6
    Additional shells as needed.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0])
    
    r1 = shell_radius
    points = [np.zeros(3)]
    
    # Shell 1: 12 vertices
    for v in verts:
        if len(points) >= n:
            break
        points.append(v * r1)
    
    if n <= 13:
        return np.array(points[:n])
    
    # Shell 2: edge midpoints (30 edges)
    verts_scaled = verts * r1
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts_scaled[i] - verts_scaled[j])
            edges.append((d, i, j))
    edges.sort()
    # Take the 30 shortest (true edges)
    for _, i, j in edges[:30]:
        if len(points) >= n:
            break
        mid = (verts_scaled[i] + verts_scaled[j]) / 2.0
        if all(np.linalg.norm(mid - p) > 0.55 for p in points):
            points.append(mid)
    
    if n <= 43:
        return np.array(points[:n])
    
    # Shell 3: face centers (20 faces)
    # Find faces: triples of vertices forming equilateral triangles
    face_centers = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(verts_scaled[i] - verts_scaled[j])
                d2 = np.linalg.norm(verts_scaled[j] - verts_scaled[k])
                d3 = np.linalg.norm(verts_scaled[k] - verts_scaled[i])
                if abs(d1 - d2) < 0.01 and abs(d2 - d3) < 0.01:
                    fc = (verts_scaled[i] + verts_scaled[j] + verts_scaled[k]) / 3.0
                    face_centers.append(fc)
    
    # Deduplicate
    unique_fc = []
    for fc in face_centers:
        if all(np.linalg.norm(fc - ufc) > 0.1 for ufc in unique_fc):
            unique_fc.append(fc)
    
    for fc in unique_fc:
        if len(points) >= n:
            break
        if all(np.linalg.norm(fc - p) > 0.55 for p in points):
            points.append(fc)
    
    if n <= 63:
        return np.array(points[:n])
    
    # Additional: random surface points
    while len(points) < n:
        theta = rng.random() * 2 * np.pi
        phi_ang = np.arccos(2 * rng.random() - 1)
        r = r1 * (1.0 + rng.random() * 0.3)
        p = np.array([
            r * np.sin(phi_ang) * np.cos(theta),
            r * np.sin(phi_ang) * np.sin(theta),
            r * np.cos(phi_ang)
        ])
        if all(np.linalg.norm(p - q) >= 0.55 for q in points):
            points.append(p)
    
    points = np.array(points[:n])
    points = points + rng.normal(scale=0.015, size=points.shape)
    return points

def _generate_fcc(n, rng):
    """Generate an FCC-like cluster."""
    a = 2.0
    R = (3.0 * n / (4.0 * np.pi) * (a**3 / 4)) ** (1.0/3.0) * 1.15
    box_size = 2 * R + a
    nx = int(box_size / a) + 2
    ny = int(box_size / a) + 2
    nz = int(box_size / a) + 2
    
    center = np.array([box_size/2, box_size/2, box_size/2])
    points = []
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                base = np.array([i, j, k]) * a
                for offset in [np.array([0,0,0]), np.array([0.5,0.5,0])*a,
                               np.array([0.5,0,0.5])*a, np.array([0,0.5,0.5])*a]:
                    p = base + offset - center
                    if np.linalg.norm(p) <= R:
                        points.append(p)
    
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-R*1.3, R*1.3, 3)
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
            elif rng.random() < 0.05:
                points.append(p)
    else:
        idx = rng.choice(len(points), n, replace=False)
        points = [points[i] for i in idx]
    
    points = np.array(points)
    points = points + rng.normal(scale=0.015, size=points.shape)
    return points

def _generate_random(n, rng):
    """Generate a random compact cluster."""
    R = (3.0 * n / (4.0 * np.pi) * 10.0) ** (1.0/3.0) * 1.1
    points = []
    min_dist = 0.55
    max_attempts = n * 2000
    
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        r = R * (rng.random() ** (1.0/3.0))
        theta = rng.random() * 2 * np.pi
        phi_ang = np.arccos(2 * rng.random() - 1)
        p = np.array([
            r * np.sin(phi_ang) * np.cos(theta),
            r * np.sin(phi_ang) * np.sin(theta),
            r * np.cos(phi_ang)
        ])
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-R*1.5, R*1.5, 3)
        points.append(p)
    
    points = np.array(points[:n])
    points = points + rng.normal(scale=0.015, size=points.shape)
    return points

def _generate_markov(n, rng, n_steps=5000):
    """
    Generate diverse structures via a Markov chain of local moves.
    Start from random, then do random displacements of single atoms.
    """
    R = (3.0 * n / (4.0 * np.pi) * 10.0) ** (1.0/3.0) * 1.1
    points = []
    min_dist = 0.55
    max_attempts = n * 2000
    
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        r = R * (rng.random() ** (1.0/3.0))
        theta = rng.random() * 2 * np.pi
        phi_ang = np.arccos(2 * rng.random() - 1)
        p = np.array([
            r * np.sin(phi_ang) * np.cos(theta),
            r * np.sin(phi_ang) * np.sin(theta),
            r * np.cos(phi_ang)
        ])
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-R*1.5, R*1.5, 3)
        points.append(p)
    
    points = np.array(points[:n])
    
    # Markov chain: random single-atom displacements
    for _ in range(n_steps):
        i = rng.integers(n)
        disp = rng.normal(scale=0.3, size=3)
        new_p = points[i] + disp
        if all(np.linalg.norm(new_p - points[j]) >= 0.5 for j in range(n) if j != i):
            points[i] = new_p
    
    points = points + rng.normal(scale=0.01, size=points.shape)
    return points

def _generate_bimodal(n, rng):
    """
    Generate a structure with two icosahedral cores (bimodal).
    Useful for larger clusters where the global minimum may have
    two intertwined icosahedral sub-clusters.
    """
    if n <= 26:
        return _generate_icosahedral(n, rng)
    
    n1 = n // 2
    n2 = n - n1
    
    ico1 = _generate_icosahedral(n1, rng, shell_radius=2.0)
    ico2 = _generate_icosahedral(n2, rng, shell_radius=2.0)
    
    # Offset the two cores
    sep = 4.0
    ico1[:, 0] += sep / 2
    ico2[:, 0] -= sep / 2
    
    points = np.vstack([ico1, ico2])
    points = points + rng.normal(scale=0.02, size=points.shape)
    return points

def _worker(args):
    """Worker function to relax a single starting structure."""
    positions, potential, parameter, deadline = args
    try:
        if time.time() > deadline - 5:
            return (np.inf, positions)
        energy, relaxed = relax(positions, potential, parameter)
        return (energy, relaxed)
    except Exception:
        return (np.inf, positions)

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    deadline = _get_deadline()
    
    rng_seed = size
    rng = np.random.default_rng(rng_seed)
    
    # Generate diverse starting structures
    candidates = []
    
    # Multiple icosahedral variants with different shell radii
    for sr in [1.95, 2.0, 2.05, 2.1]:
        rng_ico = np.random.default_rng(rng_seed + int(sr*100))
        candidates.append(_generate_icosahedral(size, rng_ico, shell_radius=sr))
    
    # FCC
    rng_fcc = np.random.default_rng(rng_seed + 111)
    candidates.append(_generate_fcc(size, rng_fcc))
    
    # Random
    for seed_off in [222, 333, 444]:
        rng_r = np.random.default_rng(rng_seed + seed_off)
        candidates.append(_generate_random(size, rng_r))
    
    # Markov chain
    for seed_off in [555, 666]:
        rng_m = np.random.default_rng(rng_seed + seed_off)
        candidates.append(_generate_markov(size, rng_m, n_steps=3000))
    
    # Bimodal
    rng_b = np.random.default_rng(rng_seed + 777)
    candidates.append(_generate_bimodal(size, rng_b))
    
    # Additional icosahedral with different seed
    rng_ico2 = np.random.default_rng(rng_seed + 888)
    candidates.append(_generate_icosahedral(size, rng_ico2, shell_radius=2.05))
    
    # Limit candidates to avoid too many parallel tasks
    max_candidates = 12
    if len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]
    
    # Parallel relaxation
    tasks = [(pos, potential, parameter, deadline) for pos in candidates]
    
    best_energy = np.inf
    best_positions = candidates[0]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_worker, t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            if time.time() > deadline - 3:
                break
            try:
                energy, pos = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = pos
            except Exception:
                pass
    
    # If we have time, do a second relaxation on the best result
    remaining = deadline - time.time()
    if remaining > 20 and best_energy < np.inf:
        try:
            energy2, relaxed2 = relax(best_positions, potential, parameter)
            if energy2 < best_energy:
                best_energy = energy2
                best_positions = relaxed2
        except Exception:
            pass
    
    # If still time, try a local perturbation + relax on the best
    remaining = deadline - time.time()
    if remaining > 15 and best_energy < np.inf:
        rng_pert = np.random.default_rng(rng_seed + 999)
        perturbed = best_positions + rng_pert.normal(scale=0.1, size=best_positions.shape)
        try:
            energy3, relaxed3 = relax(perturbed, potential, parameter)
            if energy3 < best_energy:
                best_energy = energy3
                best_positions = relaxed3
        except Exception:
            pass
    
    return best_positions
