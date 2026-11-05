"""Dzugutov cluster global minima search using polytetrahedral/icosahedral initial guesses."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax


# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16.0


def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-12:
                continue
            if r < a:
                energy += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                energy += B * np.exp(d / (r - b))
    return energy


def _icosahedron_vertices(radius=1.0):
    """Generate 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis] * radius
    return verts


def _get_icosahedron_faces():
    """Get the 20 triangular faces of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    verts_unit = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts_unit, axis=1)
    verts_unit = verts_unit / norms[:, np.newaxis]
    
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                d1 = np.linalg.norm(verts_unit[i] - verts_unit[j])
                d2 = np.linalg.norm(verts_unit[j] - verts_unit[k])
                d3 = np.linalg.norm(verts_unit[k] - verts_unit[i])
                edge_len = 2.0 / np.sqrt(phi * phi + 1)
                if abs(d1 - edge_len) < 1e-6 and abs(d2 - edge_len) < 1e-6 and abs(d3 - edge_len) < 1e-6:
                    faces.append((i, j, k))
    return faces, verts_unit


def _build_mackay_icosahedron(n, rng):
    """Build a Mackay icosahedral cluster with n atoms."""
    # Mackay shells: 1 (center), 12 (first shell), 42 (second shell), 92 (third shell)
    r1 = 1.0  # first shell radius
    r2 = 1.732  # second shell radius (sqrt(3) for Mackay)
    r3 = 2.449  # third shell radius (sqrt(6) for Mackay)
    
    points = [np.array([0.0, 0.0, 0.0])]  # center
    
    # First shell: icosahedron vertices
    ico = _icosahedron_vertices(r1)
    points.extend(list(ico))
    
    if n <= 13:
        return np.array(points[:n])
    
    # Second shell: 42 positions
    faces, verts_unit = _get_icosahedron_faces()
    
    # Vertex positions at r2
    vertex_pos = [v * r2 for v in verts_unit]
    
    # Face center positions
    face_center_pos = []
    for face in faces:
        fc = np.mean(verts_unit[list(face)], axis=0)
        fc = fc / np.linalg.norm(fc) * r2
        face_center_pos.append(fc)
    
    # Edge midpoint positions
    edge_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            edge_len = 2.0 / np.sqrt((1 + np.sqrt(5)) / 2 * ((1 + np.sqrt(5)) / 2) + 1)
            if abs(d - edge_len) < 1e-6:
                em = (verts_unit[i] + verts_unit[j]) / 2
                em = em / np.linalg.norm(em) * r2
                edge_mids.append(em)
    
    second_shell = vertex_pos + face_center_pos + edge_mids
    
    # Add second shell atoms
    needed_second = min(n - 13, len(second_shell))
    for i in range(needed_second):
        points.append(second_shell[i])
    
    if n <= 13 + needed_second:
        return np.array(points[:n])
    
    # Third shell if needed
    needed_third = n - len(points)
    if needed_third > 0:
        for i in range(needed_third):
            theta = rng.uniform(0, 2 * np.pi)
            phi_angle = rng.uniform(0, np.pi)
            p = np.array([
                r3 * np.sin(phi_angle) * np.cos(theta),
                r3 * np.sin(phi_angle) * np.sin(theta),
                r3 * np.cos(phi_angle)
            ])
            points.append(p)
    
    return np.array(points[:n])


def _build_compact_random(n, rng):
    """Build a random compact cluster with better packing."""
    spacing = 1.0
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    points = []
    attempts = 0
    min_dist = 0.55
    
    while len(points) < n and attempts < 20000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        base = points[len(points) % max(1, len(points))] if points else np.zeros(3)
        p = base + 0.25 * rng.standard_normal(3)
        points.append(p)
    
    return np.array(points[:n])


def _build_polytetrahedral(n, rng):
    """Build a cluster based on polytetrahedral packing."""
    r = 1.0
    points = []
    
    tet = _tetrahedron_vertices(r)
    points.extend(list(tet))
    
    layer_radius = r * 1.5
    while len(points) < n:
        p = rng.uniform(-layer_radius, layer_radius, 3)
        if np.linalg.norm(p) <= layer_radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5:
                    valid = False
                    break
            if valid:
                points.append(p)
            else:
                p2 = p + 0.15 * rng.standard_normal(3)
                points.append(p2)
        else:
            points.append(p + 0.1 * rng.standard_normal(3))
    
    return np.array(points[:n])


def _tetrahedron_vertices(radius=1.0):
    """Generate 4 vertices of a regular tetrahedron."""
    verts = np.array([
        [1, 1, 1], [1, -1, -1],
        [-1, 1, -1], [-1, -1, 1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis] * radius
    return verts


def _octahedron_vertices(radius=1.0):
    """Generate 6 vertices of a regular octahedron."""
    verts = np.array([
        [radius, 0, 0], [-radius, 0, 0],
        [0, radius, 0], [0, -radius, 0],
        [0, 0, radius], [0, 0, -radius]
    ], dtype=float)
    return verts


def _build_icosahedral_cluster(n, rng):
    """Build a cluster based on icosahedral symmetry for n atoms."""
    r1 = 1.0
    r2 = 1.7
    
    points = []
    points.append(np.array([0.0, 0.0, 0.0]))
    
    ico = _icosahedron_vertices(r1)
    points.extend(list(ico))
    
    if n <= 13:
        if n < len(points):
            idx = rng.choice(len(points), n, replace=False)
            points = [points[i] for i in idx]
        return np.array(points)
    
    faces, verts_unit = _get_icosahedron_faces()
    
    face_centers = []
    for face in faces:
        fc = np.mean(verts_unit[list(face)], axis=0)
        fc = fc / np.linalg.norm(fc) * r2
        face_centers.append(fc)
    
    edge_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            edge_len = 2.0 / np.sqrt((1 + np.sqrt(5)) / 2 * ((1 + np.sqrt(5)) / 2) + 1)
            if abs(d - edge_len) < 1e-6:
                em = (verts_unit[i] + verts_unit[j]) / 2
                em = em / np.linalg.norm(em) * r2
                edge_mids.append(em)
    
    second_shell_candidates = face_centers + edge_mids + [v * r2 for v in verts_unit]
    
    needed = n - len(points)
    if needed <= 0:
        return np.array(points[:n])
    
    idx = rng.permutation(len(second_shell_candidates))
    for i in range(min(needed, len(second_shell_candidates))):
        points.append(second_shell_candidates[idx[i]])
    
    while len(points) < n:
        p = rng.uniform(-r2, r2, 3)
        if np.linalg.norm(p) <= r2:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5:
                    valid = False
                    break
            if valid:
                points.append(p)
            else:
                points.append(p + 0.1 * rng.standard_normal(3))
    
    return np.array(points[:n])


def _build_random_compact(n, rng):
    """Build a random compact cluster."""
    return _build_compact_random(n, rng)


def _build_tetrahedral_cluster(n, rng):
    """Build a cluster based on tetrahedral packing."""
    return _build_polytetrahedral(n, rng)


def construct(instance):
    """Find low-energy Dzugutov clusters using polytetrahedral initial guesses."""
    start_time = time.time()
    time_limit = 300.0
    safety_margin = 15.0
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(size)
    
    best_energy = float('inf')
    best_positions = None
    
    candidates = []
    
    # Strategy 1: Mackay icosahedral with different shell radii
    for r1_scale in [0.95, 1.0, 1.05]:
        try:
            pos = _build_mackay_icosahedron(size, rng)
            pos = pos * r1_scale
            pos = pos + 0.03 * rng.standard_normal((size, 3))
            candidates.append(pos)
        except Exception:
            pass
    
    # Strategy 2: Polyteahedral packing
    for i in range(3):
        try:
            pos = _build_polytetrahedral(size, rng)
            pos = pos + 0.05 * rng.standard_normal((size, 3))
            candidates.append(pos)
        except Exception:
            pass
    
    # Strategy 3: Random compact with different densities
    for density in [0.8, 1.0, 1.2]:
        try:
            pos = _build_compact_random(size, rng)
            pos = pos * density
            pos = pos + 0.02 * rng.standard_normal((size, 3))
            candidates.append(pos)
        except Exception:
            pass
    
    # Strategy 4: Icosahedral with random second shell selection
    for i in range(2):
        try:
            pos = _build_icosahedral_cluster(size, rng)
            pos = pos + 0.04 * rng.standard_normal((size, 3))
            candidates.append(pos)
        except Exception:
            pass
    
    # Relax each candidate and keep the best
    for pos in candidates:
        elapsed = time.time() - start_time
        if elapsed > time_limit - safety_margin:
            break
        
        try:
            energy, positions = relax(pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue
    
    # If we still have time, try more diverse restarts
    elapsed = time.time() - start_time
    if best_positions is not None and elapsed < time_limit - safety_margin - 30:
        for i in range(6):
            elapsed = time.time() - start_time
            if elapsed > time_limit - safety_margin:
                break
            
            try:
                if i < 2:
                    pos = best_positions + 0.1 * rng.standard_normal((size, 3))
                elif i < 4:
                    pos = best_positions + 0.2 * rng.standard_normal((size, 3))
                else:
                    pos = best_positions.copy()
                    swap_idx = rng.choice(size, 2, replace=False)
                    pos[swap_idx[0]], pos[swap_idx[1]] = pos[swap_idx[1]].copy(), pos[swap_idx[0]].copy()
                    pos = pos + 0.15 * rng.standard_normal((size, 3))
                
                energy, positions = relax(pos, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            except Exception:
                continue
    
    # If we still have time, try completely new structures
    elapsed = time.time() - start_time
    if elapsed < time_limit - safety_margin - 20:
        for i in range(4):
            elapsed = time.time() - start_time
            if elapsed > time_limit - safety_margin:
                break
            
            try:
                if i % 3 == 0:
                    pos = _build_mackay_icosahedron(size, rng)
                elif i % 3 == 1:
                    pos = _build_compact_random(size, rng)
                else:
                    pos = _build_polytetrahedral(size, rng)
                
                pos = pos + 0.08 * rng.standard_normal((size, 3))
                
                energy, positions = relax(pos, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            except Exception:
                continue
    
    # Final refinement
    elapsed = time.time() - start_time
    if best_positions is not None and elapsed < time_limit - safety_margin - 10:
        for i in range(3):
            elapsed = time.time() - start_time
            if elapsed > time_limit - safety_margin:
                break
            
            try:
                pos = best_positions + 0.05 * rng.standard_normal((size, 3))
                
                energy, positions = relax(pos, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            except Exception:
                continue
    
    if best_positions is None:
        best_positions = _build_compact_random(size, rng)
    
    return best_positions
# EVOLVE-BLOCK-END
