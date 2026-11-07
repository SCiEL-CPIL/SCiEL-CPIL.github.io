"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

def _generate_fcc(n, spacing, rng):
    """Generate FCC lattice points within a sphere."""
    R = spacing * (3 * n / (4 * np.pi)) ** (1.0 / 3.0) * 1.2
    max_idx = int(R / spacing) + 2
    
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                base = np.array([i, j, k]) * spacing
                for b in [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]:
                    p = base + np.array(b) * spacing
                    if np.linalg.norm(p) <= R:
                        points.append(p)
    
    if not points:
        return None
        
    points = np.array(points)
    if len(points) < n:
        return None
        
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)[:n]
    return points[indices]

def _generate_icosahedral(n, spacing, rng):
    """Generate icosahedral shell points."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ])
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    
    R1 = spacing / 2.0
    shell1 = v * R1
    
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if np.isclose(np.linalg.norm(v[i] - v[j]), 2.0):
                edges.append((i, j))
                
    shell2 = []
    for i, j in edges:
        p = (v[i] + v[j]) / 2.0
        p = p / np.linalg.norm(p)
        R2 = R1 * np.sqrt(1.0 + t)
        shell2.append(p * R2)
    shell2 = np.array(shell2)
    
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                if (np.isclose(np.linalg.norm(v[i] - v[j]), 2.0) and
                    np.isclose(np.linalg.norm(v[j] - v[k]), 2.0) and
                    np.isclose(np.linalg.norm(v[i] - v[k]), 2.0)):
                    faces.append((i, j, k))
                    
    shell3 = []
    for i, j, k in faces:
        p = (v[i] + v[j] + v[k]) / 3.0
        p = p / np.linalg.norm(p)
        R3 = R1 * (1.0 + t) / np.sqrt(3.0)
        shell3.append(p * R3)
    shell3 = np.array(shell3)
    
    all_points = np.vstack([np.array([[0, 0, 0]]), shell1, shell2, shell3])
    
    if n > len(all_points):
        R4 = R3 * 1.2
        num_extra = n - len(all_points)
        z = rng.uniform(-1, 1, num_extra)
        phi = rng.uniform(0, 2 * np.pi, num_extra)
        x = np.sqrt(1 - z**2) * np.cos(phi)
        y = np.sqrt(1 - z**2) * np.sin(phi)
        extra = np.column_stack((x, y, z)) * R4
        all_points = np.vstack([all_points, extra])
        
    if len(all_points) > n:
        dists = np.linalg.norm(all_points, axis=1)
        indices = np.argsort(dists)[:n]
        all_points = all_points[indices]
        
    return all_points

def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "sutton_chen")
    
    spacing = 0.72
    rng = np.random.default_rng(n)
    
    best_energy = np.inf
    best_pos = None
    
    start_time = time.time()
    deadline = 45.0 # seconds per instance
    
    starts = []
    
    ico_pos = _generate_icosahedral(n, spacing, rng)
    if ico_pos is not None:
        starts.append(ico_pos)
        
    fcc_pos = _generate_fcc(n, spacing, rng)
    if fcc_pos is not None:
        starts.append(fcc_pos)
        
    R = spacing * n ** (1.0 / 3.0) * 0.8
    for _ in range(10):
        pts = []
        while len(pts) < n:
            p = rng.uniform(-R, R, 3)
            if np.linalg.norm(p) <= R:
                if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in pts):
                    pts.append(p)
        starts.append(np.array(pts))
        
    for _ in range(10):
        fcc_pos = _generate_fcc(n, spacing, rng)
        if fcc_pos is not None:
            fcc_pos = fcc_pos + rng.normal(0, 0.05 * spacing, fcc_pos.shape)
            starts.append(fcc_pos)

    for pos in starts:
        if time.time() - start_time > deadline:
            break
            
        if np.any(~np.isfinite(pos)):
            continue
            
        min_d = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_d:
                    min_d = d
                if min_d < 0.35:
                    break
            if min_d < 0.35:
                break
                
        if min_d < 0.35:
            continue
            
        try:
            energy, rel_pos = relax(pos, potential, kind)
            if energy < best_energy:
                best_energy = energy
                best_pos = rel_pos
        except Exception:
            pass

    if best_pos is None:
        R = spacing * n ** (1.0 / 3.0)
        best_pos = rng.uniform(-R, R, (n, 3))
        
    return best_pos
# EVOLVE-BLOCK-END
