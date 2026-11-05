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


def _get_icosahedron_topology():
    """Get faces and edges of unit icosahedron."""
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
    
    edge_len = 2.0 / np.sqrt(phi * phi + 1)
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if abs(d - edge_len) < 1e-6:
                edges.append((i, j))
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                if (tuple(sorted((i, j))) in edge_set and 
                    tuple(sorted((j, k))) in edge_set and 
                    tuple(sorted((i, k))) in edge_set):
                    faces.append((i, j, k))
    
    return verts_unit, faces, edges


def _build_mackay_icosahedron(n, rng):
    """Build a Mackay icosahedral cluster with proper shell structure."""
    points = [np.zeros(3)]  # center
    r1 = 1.0
    ico = _icosahedron_vertices(r1)
    points.extend(list(ico))
    
    if n <= 13:
        if n < len(points):
            idx = rng.choice(len(points), n, replace=False)
            points = [points[i] for i in idx]
        return np.array(points)
    
    verts_unit, faces, edges = _get_icosahedron_topology()
    
    # Second shell: 20 face centers + 30 edge centers = 50 positions
    face_centers = []
    for face in faces:
        fc = np.mean(verts_unit[list(face)], axis=0)
        fc = fc / np.linalg.norm(fc)
        face_centers.append(fc)
    
    edge_centers = []
    for i, j in edges:
        ec = (verts_unit[i] + verts_unit[j]) / 2
        ec = ec / np.linalg.norm(ec)
        edge_centers.append(ec)
    
    # Build second shell candidates
    second_shell = []
    r2 = 1.65
    for fc in face_centers:
        second_shell.append(fc * r2)
    for ec in edge_centers:
        second_shell.append(ec * r2)
    
    second_shell = np.array(second_shell)
    needed = n - len(points)
    
    if needed <= len(second_shell):
        # Use first 'needed' candidates (face centers prioritized)
        idx = np.arange(needed)
        for i in idx:
            points.append(second_shell[i])
    else:
        # Use all second shell, then add third shell
        for p in second_shell:
            points.append(p)
        
        # Third shell: 12 vertex directions at larger radius
        r3 = 2.4
        while len(points) < n:
            for v in verts_unit:
                if len(points) >= n:
                    break
                p = v * r3
                valid = True
                for q in points:
                    if np.linalg.norm(p - q) < 0.5:
                        valid = False
                        break
                if valid:
                    points.append(p)
            
            if len(points) < n:
                # Add random positions if still needed
                p = rng.uniform(-r3, r3, 3)
                points.append(p)
    
    return np.array(points[:n])


def _build_compact_random(n, rng, min_dist=0.8):
    """Build a random compact cluster with minimum distance constraint."""
    radius = 1.2 * n ** (1.0 / 3.0)
    points = []
    attempts = 0
    max_attempts = 5000
    while len(points) < n and attempts < max_attempts:
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
        if points:
            base = points[rng.integers(0, len(points))]
            p = base + 0.4 * rng.standard_normal(3)
        else:
            p = np.zeros(3)
        points.append(p)
    
    return np.array(points[:n])


def _build_polytetrahedral(n, rng):
    """Build a polytetrahedral cluster (favored by Dzugutov potential)."""
    r = 1.0
    points = [np.zeros(3)]
    
    # Start with a tetrahedron
    phi = (1 + np.sqrt(5)) / 2
    tet = np.array([
        [1, 1, 1], [1, -1, -1],
        [-1, 1, -1], [-1, -1, 1]
    ], dtype=float)
    tet = tet / np.linalg.norm(tet, axis=1)[:, np.newaxis] * r
    points.extend(list(tet))
    
    # Add more atoms in tetrahedral positions
    for _ in range(n - len(points)):
        # Pick a random existing atom and add neighbor in tetrahedral direction
        if points:
            base_idx = rng.integers(0, len(points))
            base = points[base_idx]
            # Random direction with minimum distance
            for attempt in range(20):
                p = base + r * rng.standard_normal(3)
                valid = True
                for q in points:
                    if np.linalg.norm(p - q) < 0.5:
                        valid = False
                        break
                if valid:
                    points.append(p)
                    break
            else:
                points.append(base + 0.3 * rng.standard_normal(3))
        else:
            points.append(np.zeros(3))
    
    return np.array(points[:n])


def construct(instance):
    """Find low-energy Dzugutov clusters using polytetrahedral/icosahedral initial guesses."""
    start_time = time.time()
    time_limit = 300.0
    safety_margin = 20.0
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(size)
    
    best_energy = float('inf')
    best_positions = None
    
    def check_time():
        return (time.time() - start_time) < (time_limit - safety_margin)
    
    def try_relax(pos, label=""):
        nonlocal best_energy, best_positions
        try:
            energy, positions = relax(pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
            return energy
        except Exception:
            return float('inf')
    
    # Phase 1: Icosahedral structures (primary strategy for Dzugutov)
    for i in range(8):
        if not check_time():
            break
        # Vary shell radii slightly
        r1 = 0.95 + 0.02 * rng.integers(0, 5)
        r2 = 1.55 + 0.03 * rng.integers(0, 6)
        pos = _build_mackay_icosahedron(size, rng)
        # Small perturbation
        scale = 0.03 + 0.02 * i
        pos = pos + scale * rng.standard_normal((size, 3))
        try_relax(pos, f"ico_{i}")
    
    # Phase 2: Polytetrahedral structures
    for i in range(5):
        if not check_time():
            break
        pos = _build_polytetrahedral(size, rng)
        scale = 0.05 + 0.03 * i
        pos = pos + scale * rng.standard_normal((size, 3))
        try_relax(pos, f"poly_{i}")
    
    # Phase 3: Random compact structures
    for i in range(5):
        if not check_time():
            break
        pos = _build_compact_random(size, rng)
        scale = 0.04 + 0.02 * i
        pos = pos + scale * rng.standard_normal((size, 3))
        try_relax(pos, f"rand_{i}")
    
    # Phase 4: Perturb best solution to escape local minima
    if best_positions is not None:
        for i in range(8):
            if not check_time():
                break
            # Geometrically motivated perturbations
            if i < 4:
                # Small random perturbations
                scale = 0.1 + 0.05 * i
                pos = best_positions + scale * rng.standard_normal((size, 3))
            else:
                # Swap two atoms or move one atom significantly
                pos = best_positions.copy()
                idx1 = rng.integers(0, size)
                if i == 4:
                    # Move one atom
                    pos[idx1] += 0.5 * rng.standard_normal(3)
                elif i == 5:
                    # Swap two atoms
                    idx2 = rng.integers(0, size)
                    if idx2 != idx1:
                        pos[idx1], pos[idx2] = pos[idx2].copy(), pos[idx1].copy()
                else:
                    # Move multiple atoms
                    n_move = 2 + i - 5
                    idxs = rng.choice(size, n_move, replace=False)
                    pos[idxs] += 0.3 * rng.standard_normal((n_move, 3))
            
            try_relax(pos, f"perturb_{i}")
    
    # Phase 5: If time remains, do more diverse restarts
    remaining_checks = 0
    while check_time() and remaining_checks < 10:
        remaining_checks += 1
        strategy = rng.integers(0, 3)
        if strategy == 0:
            pos = _build_mackay_icosahedron(size, rng)
        elif strategy == 1:
            pos = _build_polytetrahedral(size, rng)
        else:
            pos = _build_compact_random(size, rng)
        
        scale = 0.05 + 0.1 * rng.random()
        pos = pos + scale * rng.standard_normal((size, 3))
        try_relax(pos, f"extra_{remaining_checks}")
    
    if best_positions is None:
        best_positions = _build_compact_random(size, rng)
    
    return best_positions
# EVOLVE-BLOCK-END
