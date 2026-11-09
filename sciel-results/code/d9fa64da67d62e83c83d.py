"""Dzugutov cluster global minima via icosahedral superstructure generation with basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global deadline: 300s total, leave 20s safety margin
_TOTAL_DEADLINE = time.time() + 280.0


def _icosahedron_vertices(scale=1.0):
    """Return 12 vertices of a regular icosahedron scaled so edge length = scale."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Edge length: distance between vertex 0 and vertex 4
    d = np.linalg.norm(verts[0] - verts[4])
    verts = verts * (scale / d)
    return verts


def _icosahedron_face_centers(scale=1.0):
    """Return 20 face centers of a regular icosahedron (second shell of Mackay)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit icosahedron (circumradius = 1)
    circumradius = np.linalg.norm(verts[0])
    verts = verts / circumradius

    # Find all faces (triangles) of the icosahedron
    # The icosahedron has 20 faces. Each vertex connects to 5 others.
    # We'll find faces by checking all triplets that form equilateral triangles
    # on the unit sphere (all pairwise distances equal).
    n = 12
    edge_len = np.linalg.norm(verts[0] - verts[4])
    faces = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[k] - verts[i])
                if abs(d1 - d2) < 1e-9 and abs(d2 - d3) < 1e-9:
                    faces.append((i, j, k))

    # Face centers
    face_centers = []
    for f in faces:
        fc = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        face_centers.append(fc)
    face_centers = np.array(face_centers)

    # Scale so that the distance from center to face center matches the
    # second-shell radius relative to the first-shell radius
    # For a regular icosahedron with edge length L:
    #   circumradius R = L * sqrt(10 + 2*sqrt(5)) / 4
    #   inradius (face center distance) r = L * sqrt(3(5+sqrt(5))) / 12
    #   ratio r/R = sqrt(3(5+sqrt(5))) / (sqrt(10+2*sqrt(5))) * (4/12) ... let's just compute
    # Actually, let's just scale the face centers to have the correct ratio.
    # The Mackay icosahedron second shell is at the face centers of the first shell.
    # We need to scale so that the overall structure has the right proportions.
    # For now, return them at their natural position (circumradius * ratio)
    # and let the caller scale appropriately.
    return face_centers


def _generate_mackay_icosahedron(n_atoms, spacing=1.0):
    """Generate a Mackay icosahedral superstructure with n_atoms.
    
    The Mackay icosahedron has shells:
    - 1 (center)
    - 12 (vertices)
    - 20 (face centers)
    - 30 (edge midpoints)
    - 20 (next face centers)
    - 12 (next vertices)
    - 60 (next edge midpoints)
    ...
    
    Cumulative: 1, 13, 33, 63, 83, 95, 155, ...
    
    For n atoms, we fill shells in order and take the first n atoms.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # Unit icosahedron vertices (circumradius = 1)
    base_verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    circumradius = np.linalg.norm(base_verts[0])
    base_verts = base_verts / circumradius  # unit circumradius

    # All shell points (unit circumradius icosahedron, then scaled)
    # Shell 1: 12 vertices at radius 1
    shell1 = base_verts.copy()

    # Shell 2: 20 face centers
    # Find faces
    n_v = 12
    faces = []
    for i in range(n_v):
        for j in range(i + 1, n_v):
            for k in range(j + 1, n_v):
                d1 = np.linalg.norm(base_verts[i] - base_verts[j])
                d2 = np.linalg.norm(base_verts[j] - base_verts[k])
                d3 = np.linalg.norm(base_verts[k] - base_verts[i])
                if abs(d1 - d2) < 1e-9 and abs(d2 - d3) < 1e-9:
                    faces.append((i, j, k))
    face_centers = np.array([(base_verts[f[0]] + base_verts[f[1]] + base_verts[f[2]]) / 3.0 for f in faces])
    # Normalize to unit radius
    fc_radii = np.linalg.norm(face_centers, axis=1)
    face_centers = face_centers / fc_radii[:, None]
    shell2 = face_centers.copy()

    # Shell 3: 30 edge midpoints
    edges = set()
    for f in faces:
        edges.add((f[0], f[1]))
        edges.add((f[1], f[2]))
        edges.add((f[2], f[0]))
    edge_mids = []
    for e in edges:
        mid = (base_verts[e[0]] + base_verts[e[1]]) / 2.0
        mid = mid / np.linalg.norm(mid)
        edge_mids.append(mid)
    shell3 = np.array(edge_mids)

    # Shell 4: 20 next face centers (at radius ~1.618)
    # These are the face centers of the icosahedron scaled by phi
    shell4 = face_centers * phi

    # Shell 5: 12 next vertices (at radius phi^2)
    shell5 = base_verts * (phi * phi)

    # Shell 6: 60 edge midpoints of the larger icosahedron
    # Edge midpoints at radius phi^2 * (1/phi) = phi
    shell6 = edge_mids * (phi * phi)

    # Shell 7: 20 face centers at radius phi^3
    shell7 = face_centers * (phi ** 3)

    # Shell 8: 12 vertices at radius phi^4
    shell8 = base_verts * (phi ** 4)

    # Shell 9: 30 edge midpoints at radius phi^4
    shell9 = edge_mids * (phi ** 4)

    # Shell 10: 60 edge midpoints at radius phi^5
    shell10 = edge_mids * (phi ** 5)

    # Shell 11: 20 face centers at radius phi^6
    shell11 = face_centers * (phi ** 6)

    # Shell 12: 12 vertices at radius phi^7
    shell12 = base_verts * (phi ** 7)

    # Combine shells in order
    all_shells = [
        np.array([[0.0, 0.0, 0.0]]),  # Shell 0: center
        shell1,  # 12
        shell2,  # 20
        shell3,  # 30
        shell4,  # 20
        shell5,  # 12
        shell6,  # 60
        shell7,  # 20
        shell8,  # 12
        shell9,  # 30
        shell10,  # 60
        shell11,  # 20
        shell12,  # 12
    ]

    # For n_atoms, fill shells in order
    atoms = []
    remaining = n_atoms
    for shell in all_shells:
        if remaining <= 0:
            break
        take = min(remaining, len(shell))
        atoms.append(shell[:take])
        remaining -= take

    if remaining > 0:
        # Not enough atoms in standard shells; add random points
        rng = np.random.default_rng(42)
        extra = rng.uniform(-1, 1, size=(remaining, 3))
        atoms.append(extra)

    positions = np.vstack(atoms)

    # Scale so that the nearest-neighbor distance is approximately 'spacing'
    # The nearest neighbors in the icosahedron are at distance = edge length
    # For unit circumradius icosahedron, edge length = 2*sin(pi/5) = 2*sin(36°)
    edge_len_unit = 2 * np.sin(np.pi / 5)
    scale_factor = spacing / edge_len_unit
    positions = positions * scale_factor

    return positions


def _generate_partial_icosahedron(n_atoms, spacing=1.0):
    """Generate a partial icosahedral structure for arbitrary n.
    
    Uses the Mackay icosahedron shell structure but allows for
    non-standard sizes by trimming or adding atoms.
    """
    # Generate a slightly larger icosahedron and trim
    # Find the smallest Mackay size >= n_atoms
    mackay_sizes = [1, 13, 33, 63, 83, 95, 155]
    if n_atoms <= 1:
        return np.array([[0.0, 0.0, 0.0]])

    # Generate the next larger Mackay icosahedron
    target = mackay_sizes[0]
    for ms in mackay_sizes:
        if ms >= n_atoms:
            target = ms
            break
    else:
        target = n_atoms  # If n is larger than our table, just use n

    positions = _generate_mackay_icosahedron(target, spacing)

    if len(positions) > n_atoms:
        # Trim: keep atoms closest to center (or random subset)
        # For a better structure, keep a random subset to break symmetry
        rng = np.random.default_rng(n_atoms)
        idx = rng.choice(len(positions), size=n_atoms, replace=False)
        positions = positions[idx]
    elif len(positions) < n_atoms:
        # Add random atoms
        rng = np.random.default_rng(n_atoms)
        radius = spacing * n_atoms ** (1.0 / 3.0) * 0.8
        extra = rng.uniform(-radius, radius, size=(n_atoms - len(positions), 3))
        positions = np.vstack([positions, extra])

    return positions


def _basin_hopping(positions, potential, parameter, max_iter=30, time_budget=10.0):
    """Simple basin hopping: perturb, relax, accept if lower energy."""
    best_energy, best_positions = relax(positions.copy(), potential, parameter)
    rng = np.random.default_rng(int(np.sum(np.abs(positions)) * 1000) % 100000)
    start_time = time.time()

    for i in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        if time.time() > _TOTAL_DEADLINE - 30:
            break

        # Perturbation: random displacement with Gaussian noise
        # Annealed: noise decreases over iterations
        noise_scale = 0.3 * (1.0 - i / max_iter) + 0.1
        perturbed = positions + rng.normal(0, noise_scale, size=positions.shape)

        # Ensure minimum distance constraint
        # If any pair is too close, push them apart
        for _ in range(3):
            d = np.linalg.norm(perturbed[0] - perturbed[1])
            if d < 0.5:
                perturbed[1] = perturbed[0] + 0.5 * (perturbed[1] - perturbed[0]) / max(d, 1e-8)
                break

        energy, relaxed = relax(perturbed, potential, parameter)

        if energy < best_energy:
            best_energy = energy
            best_positions = relaxed.copy()
            positions = relaxed.copy()
        else:
            # Metropolis acceptance (simplified: always reject for global min search)
            pass

    return best_energy, best_positions


def construct(instance):
    """Construct a low-energy Dzugutov cluster using icosahedral initialization + basin hopping."""
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))

    # Dzugutov potential parameters
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16

    # Optimal spacing for Dzugutov potential
    # The potential minimum is around r ≈ 1.0-1.2 in reduced units
    # Let's use spacing = 1.0 as a starting point
    spacing = 1.0

    start_time = time.time()
    best_energy = np.inf
    best_positions = None

    # Strategy 1: Mackay icosahedron (or partial) + basin hopping
    # This is the known global minimum family for Dzugutov potential
    if n in [1, 13, 33, 63, 83, 95]:
        init_positions = _generate_mackay_icosahedron(n, spacing)
    else:
        init_positions = _generate_partial_icosahedron(n, spacing)

    # Try multiple spacings
    for sp in [0.9, 1.0, 1.1]:
        if time.time() > _TOTAL_DEADLINE - 60:
            break
        init = _generate_partial_icosahedron(n, sp) if n not in [1, 13, 33, 63, 83, 95] else _generate_mackay_icosahedron(n, sp)

        # Basin hopping with time budget
        time_per_bh = min(15.0, max(3.0, (_TOTAL_DEADLINE - 60 - time.time()) / 3))
        energy, positions = _basin_hopping(init, potential, parameter, max_iter=20, time_budget=time_per_bh)

        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()

    # Strategy 2: Random compact clusters (backup)
    if time.time() < _TOTAL_DEADLINE - 30:
        rng = np.random.default_rng(n)
        radius = spacing * n ** (1.0 / 3.0) * 0.75
        for trial in range(3):
            if time.time() > _TOTAL_DEADLINE - 30:
                break
            points = []
            while len(points) < n:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                    points.append(p)
            energy, positions = relax(np.array(points), potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()

    # Final relaxation of the best structure
    if best_positions is not None and time.time() < _TOTAL_DEADLINE - 10:
        final_energy, final_positions = relax(best_positions, potential, parameter)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_positions

    if best_positions is None:
        # Fallback: random
        rng = np.random.default_rng(n)
        best_positions = rng.uniform(-2, 2, size=(n, 3))

    return best_positions
# EVOLVE-BLOCK-END
