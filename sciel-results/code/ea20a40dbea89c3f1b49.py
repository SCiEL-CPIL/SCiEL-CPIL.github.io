"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax
import time

def _dzugutov_energy(pos, A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16):
    """Compute Dzugutov potential energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(r, np.inf)
    
    E = 0.0
    # V(r) = A*(r^-m - B)*exp(c/(r-a)) for r < a
    #      + B*exp(d/(r-b)) for r < b
    mask_a = r < a
    mask_b = r < b
    
    if np.any(mask_a):
        r_a = r[mask_a]
        E += np.sum(A * (r_a**(-m) - B) * np.exp(c / (r_a - a)))
    if np.any(mask_b):
        r_b = r[mask_b]
        E += np.sum(B * np.exp(d / (r_b - b)))
    
    return E

def _generate_icosahedral_cluster(n, spacing=1.05):
    """Generate an icosahedral cluster with n atoms using polytetrahedral packing."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    phi = (1 + np.sqrt(5)) / 2
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico_vertices = ico_vertices / np.linalg.norm(ico_vertices[0]) * spacing
    
    # Second shell: 30 positions (icosahedral)
    # These are at the midpoints of edges and other positions
    second_shell = []
    # Edge midpoints scaled
    for i in range(12):
        for j in range(i+1, 12):
            mid = (ico_vertices[i] + ico_vertices[j]) / 2
            second_shell.append(mid)
    # Additional positions at octahedral directions
    oct_dir = np.array([
        [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
    ], dtype=float)
    for d in oct_dir:
        second_shell.append(d * spacing * 1.5)
    
    # Third shell positions
    third_shell = []
    for v in ico_vertices:
        third_shell.append(v * 2.0)
        third_shell.append(v * 2.2)
    
    points = []
    # Center
    points.append([0.0, 0.0, 0.0])
    # First shell
    for v in ico_vertices:
        if len(points) < n:
            points.append(v)
    # Second shell
    for v in second_shell:
        if len(points) < n:
            points.append(v)
    # Third shell
    for v in third_shell:
        if len(points) < n:
            points.append(v)
    # Fill remaining with scaled positions
    scale = 2.5
    while len(points) < n:
        for v in ico_vertices:
            if len(points) < n:
                points.append(v * scale)
        scale += 0.3
    
    return np.array(points[:n])

def _generate_polytetrahedral(n, spacing=1.05):
    """Generate polytetrahedral packing - multiple icosahedra sharing faces."""
    if n <= 13:
        return _generate_icosahedral_cluster(n, spacing)
    
    phi = (1 + np.sqrt(5)) / 2
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico_vertices = ico_vertices / np.linalg.norm(ico_vertices[0]) * spacing
    
    points = []
    # First icosahedron center at origin
    points.append([0.0, 0.0, 0.0])
    for v in ico_vertices:
        points.append(v)
    
    # Add second icosahedron sharing a face
    # The face is defined by 3 vertices of the first icosahedron
    # Find a face and place second center opposite
    v0, v1, v2 = ico_vertices[0], ico_vertices[1], ico_vertices[2]
    face_center = (v0 + v1 + v2) / 3
    face_normal = np.cross(v1 - v0, v2 - v0)
    face_normal = face_normal / np.linalg.norm(face_normal)
    
    # Second center is on the other side of the face
    second_center = face_center + face_normal * spacing * 0.9
    
    # Add second icosahedron (excluding shared atoms)
    for v in ico_vertices:
        new_pos = second_center + v
        # Check if this position is close to any existing point
        is_new = True
        for p in points:
            if np.linalg.norm(new_pos - p) < 0.3:
                is_new = False
                break
        if is_new and len(points) < n:
            points.append(new_pos)
    
    # Add third icosahedron if needed
    if len(points) < n:
        v3, v4, v5 = ico_vertices[3], ico_vertices[4], ico_vertices[5]
        face_center2 = (v3 + v4 + v5) / 3
        face_normal2 = np.cross(v4 - v3, v5 - v3)
        face_normal2 = face_normal2 / np.linalg.norm(face_normal2)
        third_center = face_center2 + face_normal2 * spacing * 0.9
        
        for v in ico_vertices:
            new_pos = third_center + v
            is_new = True
            for p in points:
                if np.linalg.norm(new_pos - p) < 0.3:
                    is_new = False
                    break
            if is_new and len(points) < n:
                points.append(new_pos)
    
    # Fill remaining
    while len(points) < n:
        rng_local = np.random.default_rng(len(points) + n)
        p = rng_local.uniform(-spacing*3, spacing*3, 3)
        is_valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.5:
                is_valid = False
                break
        if is_valid:
            points.append(p)
        else:
            points.append(p)  # Just add it, relax will fix
    
    return np.array(points[:n])

def _generate_random_compact(n, seed, spacing=1.05):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.85
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            min_dist = min((np.linalg.norm(p - q) for q in points), default=np.inf)
            if min_dist >= 0.5:
                points.append(p)
        attempts += 1
    
    # If we couldn't place all, just add random positions
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])

def construct(instance):
    """Find low-energy Dzugutov cluster structures using multiple starting geometries."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 285  # Leave 15s safety margin before 300s limit
    
    # Generate diverse starting geometries
    starting_points_list = []
    
    # 1. Icosahedral cluster
    ico_points = _generate_icosahedral_cluster(size, 1.05)
    starting_points_list.append(("ico", ico_points))
    
    # 2. Polytetrahedral cluster
    poly_points = _generate_polytetrahedral(size, 1.05)
    starting_points_list.append(("poly", poly_points))
    
    # 3-6. Random compact clusters with different seeds
    for seed in [size, size + 1000, size + 2000, size + 3000]:
        rand_points = _generate_random_compact(size, seed, 1.05)
        starting_points_list.append((f"rand{seed}", rand_points))
    
    # 7. Slightly perturbed icosahedral
    rng_pert = np.random.default_rng(size + 5000)
    pert_points = ico_points + rng_pert.normal(0, 0.1, ico_points.shape)
    starting_points_list.append(("ico_pert", pert_points))
    
    best_energy = float('inf')
    best_positions = None
    
    # Relax each starting geometry
    for name, init_points in starting_points_list:
        if time.time() > deadline:
            break
        
        try:
            energy, positions = relax(init_points, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception as e:
            continue
    
    # If we have time, try more random starts
    remaining_time = deadline - time.time()
    if remaining_time > 20 and best_positions is not None:
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            seed = size + 5000 + trial * 100
            rand_points = _generate_random_compact(size, seed, 1.05)
            try:
                energy, positions = relax(rand_points, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception as e:
                continue
    
    # If we still have time, try perturbations of the best solution
    remaining_time = deadline - time.time()
    if remaining_time > 15 and best_positions is not None:
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            rng_pert = np.random.default_rng(size + 8000 + trial)
            perturbed = best_positions + rng_pert.normal(0, 0.15, best_positions.shape)
            try:
                energy, positions = relax(perturbed, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception as e:
                continue
    
    if best_positions is None:
        # Fallback: return the first starting geometry
        return starting_points_list[0][1]
    
    return best_positions
# EVOLVE-BLOCK-END
