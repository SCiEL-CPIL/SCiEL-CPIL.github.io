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


def _octahedron_vertices(radius=1.0):
    """Generate 6 vertices of a regular octahedron."""
    verts = np.array([
        [radius, 0, 0], [-radius, 0, 0],
        [0, radius, 0], [0, -radius, 0],
        [0, 0, radius], [0, 0, -radius]
    ], dtype=float)
    return verts


def _tetrahedron_vertices(radius=1.0):
    """Generate 4 vertices of a regular tetrahedron."""
    verts = np.array([
        [1, 1, 1], [1, -1, -1],
        [-1, 1, -1], [-1, -1, 1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis] * radius
    return verts


def _build_icosahedral_cluster(n, rng):
    """Build a cluster based on icosahedral symmetry for n atoms."""
    # Mackay icosahedron shells: 1, 12, 42, 92, ...
    # For n=38: 1 + 12 + 25 (partial second shell)
    # For n=45: 1 + 12 + 32 (more complete second shell)
    # For n=50: 1 + 12 + 37
    # For n=55: 1 + 12 + 42 (full Mackay second shell)
    # For n=57: 1 + 12 + 44
    # For n=60: 1 + 12 + 47
    
    # Start with icosahedron core (12 atoms) + center (1 atom) = 13 atoms
    # Then add atoms on second shell positions
    
    # First shell: icosahedron vertices at radius r1
    # Second shell: positions on faces and edges of icosahedron
    
    r1 = 1.0  # first shell radius
    r2 = 1.7  # second shell radius (approximate for Dzugutov)
    
    points = []
    
    # Center atom
    points.append(np.array([0.0, 0.0, 0.0]))
    
    # First shell: icosahedron vertices
    ico = _icosahedron_vertices(r1)
    points.extend(list(ico))
    
    if n <= 13:
        # Randomly select n points
        if n < len(points):
            idx = rng.choice(len(points), n, replace=False)
            points = [points[i] for i in idx]
        return np.array(points)
    
    # Second shell: generate candidate positions
    # Face centers of icosahedron (20 faces)
    # Edge midpoints (30 edges)
    # Vertex positions at larger radius
    
    # Get icosahedron faces and edges
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
    
    # Find faces (triangles) of icosahedron
    # Each face is a set of 3 vertices that form an equilateral triangle
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                d1 = np.linalg.norm(verts_unit[i] - verts_unit[j])
                d2 = np.linalg.norm(verts_unit[j] - verts_unit[k])
                d3 = np.linalg.norm(verts_unit[k] - verts_unit[i])
                # Edge length of unit icosahedron is 2/sqrt(phi^2+1)
                edge_len = 2.0 / np.sqrt(phi * phi + 1)
                if abs(d1 - edge_len) < 1e-6 and abs(d2 - edge_len) < 1e-6 and abs(d3 - edge_len) < 1e-6:
                    faces.append((i, j, k))
    
    # Face centers (20 faces)
    face_centers = []
    for face in faces:
        fc = np.mean(verts_unit[list(face)], axis=0)
        fc = fc / np.linalg.norm(fc)
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    # Edge midpoints (30 edges)
    edge_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            edge_len = 2.0 / np.sqrt(phi * phi + 1)
            if abs(d - edge_len) < 1e-6:
                em = (verts_unit[i] + verts_unit[j]) / 2
                em = em / np.linalg.norm(em)
                edge_mids.append(em)
    edge_mids = np.array(edge_mids)
    
    # Generate second shell positions
    # Scale face centers and edge midpoints to r2
    second_shell_candidates = []
    for fc in face_centers:
        second_shell_candidates.append(fc * r2)
    for em in edge_mids:
        second_shell_candidates.append(em * r2)
    # Also add vertices at larger radius
    for v in verts_unit:
        second_shell_candidates.append(v * r2)
    
    second_shell_candidates = np.array(second_shell_candidates)
    
    # Select needed number of second shell atoms
    needed = n - len(points)
    if needed <= 0:
        return np.array(points[:n])
    
    # Shuffle and select
    idx = rng.permutation(len(second_shell_candidates))
    for i in range(min(needed, len(second_shell_candidates))):
        points.append(second_shell_candidates[idx[i]])
    
    # If we still need more, add random points
    while len(points) < n:
        p = rng.uniform(-r2, r2, 3)
        if np.linalg.norm(p) <= r2:
            # Check minimum distance
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
    spacing = 1.1
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.6 * spacing:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    # If we couldn't place all atoms, add them with small offsets
    while len(points) < n:
        base = points[len(points) % max(1, len(points))] if points else np.zeros(3)
        p = base + 0.3 * rng.standard_normal(3)
        points.append(p)
    
    return np.array(points[:n])


def _build_tetrahedral_cluster(n, rng):
    """Build a cluster based on tetrahedral packing."""
    # Start with a tetrahedron and add layers
    r = 1.0
    points = []
    
    # Central tetrahedron
    tet = _tetrahedron_vertices(r)
    points.extend(list(tet))
    
    # Add octahedron
    octa = _octahedron_vertices(r * 1.2)
    for p in octa:
        if len(points) >= n:
            break
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.5:
                valid = False
                break
        if valid:
            points.append(p)
    
    # Add more atoms in shells
    shell_radius = r * 1.5
    while len(points) < n:
        p = rng.uniform(-shell_radius, shell_radius, 3)
        if np.linalg.norm(p) <= shell_radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5:
                    valid = False
                    break
            if valid:
                points.append(p)
            else:
                points.append(p + 0.2 * rng.standard_normal(3))
        else:
            points.append(p + 0.2 * rng.standard_normal(3))
    
    return np.array(points[:n])


def construct(instance):
    """Find low-energy Dzugutov clusters using polytetrahedral initial guesses."""
    start_time = time.time()
    time_limit = 300.0
    safety_margin = 10.0
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(size)
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate diverse initial candidates
    n_ico = 3  # icosahedral-based
    n_tet = 2  # tetrahedral-based
    n_rand = 3  # random compact
    
    candidates = []
    
    # Icosahedral candidates with varying parameters
    for i in range(n_ico):
        try:
            pos = _build_icosahedral_cluster(size, rng)
            # Add small random perturbation
            pos = pos + 0.05 * rng.standard_normal((size, 3))
            candidates.append(pos)
        except Exception:
            pass
    
    # Tetrahedral candidates
    for i in range(n_tet):
        try:
            pos = _build_tetrahedral_cluster(size, rng)
            pos = pos + 0.05 * rng.standard_normal((size, 3))
            candidates.append(pos)
        except Exception:
            pass
    
    # Random compact candidates
    for i in range(n_rand):
        try:
            pos = _build_random_compact(size, rng)
            pos = pos + 0.03 * rng.standard_normal((size, 3))
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
                best_positions = positions
        except Exception:
            continue
    
    # If we still have time, try more random restarts
    elapsed = time.time() - start_time
    if best_positions is None or elapsed < time_limit - safety_margin - 30:
        n_extra = 5
        for i in range(n_extra):
            elapsed = time.time() - start_time
            if elapsed > time_limit - safety_margin:
                break
            
            try:
                # Mix of icosahedral and random
                if i % 2 == 0:
                    pos = _build_icosahedral_cluster(size, rng)
                else:
                    pos = _build_random_compact(size, rng)
                pos = pos + 0.1 * rng.standard_normal((size, 3))
                
                energy, positions = relax(pos, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                continue
    
    # If we still have time, do additional local optimization with different perturbations
    elapsed = time.time() - start_time
    if best_positions is not None and elapsed < time_limit - safety_margin - 20:
        n_perturb = 4
        for i in range(n_perturb):
            elapsed = time.time() - start_time
            if elapsed > time_limit - safety_margin:
                break
            
            try:
                # Perturb the best solution
                perturbation_scale = 0.2 + 0.1 * i
                pos = best_positions + perturbation_scale * rng.standard_normal((size, 3))
                
                energy, positions = relax(pos, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                continue
    
    if best_positions is None:
        # Fallback: return a simple structure
        best_positions = _build_random_compact(size, rng)
    
    return best_positions
# EVOLVE-BLOCK-END
