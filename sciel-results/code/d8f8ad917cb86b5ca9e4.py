import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Constants
POTENTIAL = 'dzugutov'
# Global time tracking
_GLOBAL_START_TIME = None
_GLOBAL_DEADLINE = None

def _get_icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron centered at origin, normalized to unit radius."""
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _get_dodecahedron_vertices():
    """Return the 20 vertices of a regular dodecahedron (dual of icosahedron), normalized to unit radius."""
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
        [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
        [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
    ]
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _generate_mackay_icosahedron_55():
    """
    Generate the 55-atom Mackay icosahedron.
    Structure:
    - 1 atom at center
    - 12 atoms at icosahedron vertices (radius r2)
    - 42 atoms in outer shell: 30 edge midpoints + 12 face centers (dodecahedron vertices)
    """
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = _get_icosahedron_vertices()
    
    # Scale factor to get reasonable bond lengths
    # For Dzugutov, equilibrium distance is around 1.1-1.2
    # Let's use a scale factor that makes nearest neighbor distance ~1.1
    # In unit icosahedron, nearest neighbor distance is 2*sin(pi/5) ~ 1.1756
    # So if we scale by 1.0, NN distance is ~1.1756 which is reasonable
    scale = 1.0
    
    center = np.zeros(3)
    shell2 = ico_verts * scale
    
    # Find edges of the icosahedron
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            dists[i, j] = d
            dists[j, i] = d
    
    min_dist = np.min(dists[dists > 0])
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
    
    # 30 edge midpoints
    edge_mids = []
    for i, j in edges:
        mid = (ico_verts[i] + ico_verts[j]) / 2
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    
    # Scale edge midpoints to same radius as shell3
    # The radius of edge midpoints in unit icosahedron is cos(pi/5) ~ 0.809
    # We want them at the same radius as the dodecahedron vertices
    edge_mid_norms = np.linalg.norm(edge_mids, axis=1, keepdims=True)
    # The dodecahedron vertices (face centers) are at radius 1/3 * sum of 3 unit vectors
    # For a regular icosahedron, face center distance from center is r_face = R * sqrt(3) / (3 * sin(pi/5))?
    # Actually, let's just compute the face centers and use their radius
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(ico_verts[i] - ico_verts[j])
                d2 = np.linalg.norm(ico_verts[j] - ico_verts[k])
                d3 = np.linalg.norm(ico_verts[i] - ico_verts[k])
                if abs(d1 - min_dist) < 1e-6 and abs(d2 - min_dist) < 1e-6 and abs(d3 - min_dist) < 1e-6:
                    faces.append((i, j, k))
    
    face_centers = []
    for i, j, k in faces:
        fc = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    face_center_norms = np.linalg.norm(face_centers, axis=1, keepdims=True)
    # All face centers should have the same norm
    r_face = face_center_norms[0, 0]
    
    # Scale edge midpoints to r_face
    edge_mids_scaled = edge_mids / edge_mid_norms * r_face
    
    # We have 20 face centers, but we need only 12 for the Mackay icosahedron
    # The 12 atoms in the outer shell that are not edge midpoints are at the vertices of the dual dodecahedron
    # But there are 20 face centers. Let's check: 30 + 20 = 50, not 42.
    # Actually, the Mackay icosahedron M(3) has 42 atoms in the outer shell.
    # Let me re-check the structure.
    # M(3) = 1 + 12 + 42 = 55
    # The 42 atoms are:
    # - 30 at edge midpoints
    # - 12 at face centers? No, there are 20 faces.
    # 
    # Let me look up the correct structure.
    # The Mackay icosahedron M(n) has shells:
    # n=1: 1
    # n=2: 12 (icosahedron vertices)
    # n=3: 42
    # 
    # The 42 atoms in shell 3 are located at:
    # - 30 edge midpoints of the icosahedron
    # - 12 vertices of a dodecahedron (dual of icosahedron)
    # But the dual dodecahedron has 20 vertices, not 12.
    # 
    # Wait, maybe I'm confusing this. Let me just use a simpler approach:
    # Generate 55 atoms in a compact icosahedral-like arrangement.
    # Or, use the known fact that the 55-atom Mackay icosahedron is a local minimum.
    # 
    # For now, let's create a seed with 1 center, 12 at icosahedron vertices,
    # and 42 atoms distributed on a sphere of slightly larger radius.
    
    # Let's use a Fibonacci sphere for the 42 outer atoms
    n_outer = 42
    golden_angle = np.pi * (3 - np.sqrt(5))
    outer_atoms = []
    r_outer = r_face  # Use the face center radius
    for i in range(n_outer):
        y = 1 - 2 * i / (n_outer - 1)
        radius_at_y = np.sqrt(1 - y**2)
        theta = golden_angle * i
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        outer_atoms.append([x, y, z] * r_outer)
    outer_atoms = np.array(outer_atoms)
    
    return np.vstack([center, shell2, outer_atoms]) * scale

def _generate_seed(size, rng):
    """Generate a diverse set of seeds for the given size using defect-seeded approach."""
    seeds = []
    
    # Generate the 55-atom Mackay icosahedron
    mackay55 = _generate_mackay_icosahedron_55()
    
    if size == 55:
        seeds.append(mackay55)
    elif size < 55:
        # Remove atoms from the 55-atom structure
        # Remove from the outer shell first, then inner
        # The 55-atom structure has:
        # Index 0: center
        # Indices 1-12: shell 2 (12 atoms)
        # Indices 13-54: shell 3 (42 atoms)
        
        # To get size atoms, we keep the center and as many shell atoms as possible
        # Priority: center > shell 2 > shell 3
        
        if size == 1:
            seeds.append(mackay55[:1])
        elif size <= 13:
            # Keep center + some shell 2 atoms
            n_shell2 = size - 1
            seeds.append(mackay55[:1 + n_shell2])
        else:
            # Keep center + all 12 shell 2 + some shell 3
            n_shell3 = size - 13
            n_shell3 = min(n_shell3, 42)
            seeds.append(mackay55[:1 + 12 + n_shell3])
        
        # Also add a version with some random removals for diversity
        if size >= 13 and size <= 54:
            # Randomly remove (55 - size) atoms from the outer shells
            indices = list(range(55))
            # Keep center always
            to_remove = 55 - size
            # Remove from outer shell first (indices 13-54)
            outer_indices = list(range(13, 55))
            rng.shuffle(outer_indices)
            remove_set = set(outer_indices[:to_remove])
            keep_indices = [i for i in range(55) if i not in remove_set]
            seeds.append(mackay55[keep_indices])
    else:
        # size > 55: Add atoms to the 55-atom structure
        current = mackay55.copy()
        r_surface = 1.2  # Slightly outside the outer shell
        for _ in range(size - 55):
            # Add atom at random position on the surface
            u1 = rng.uniform(0, 2 * np.pi)
            u2 = rng.uniform(-1, 1)
            x = r_surface * np.sqrt(1 - u2**2) * np.cos(u1)
            y = r_surface * np.sqrt(1 - u2**2) * np.sin(u1)
            z = r_surface * u2
            current = np.vstack([current, [x, y, z]])
        seeds.append(current)
    
    # Add a 13-atom icosahedron-based seed for small sizes
    if size >= 13:
        verts = _get_icosahedron_vertices()
        center = np.zeros(3)
        atoms = np.vstack([center, verts])
        
        if size == 13:
            seeds.append(atoms)
        elif size > 13:
            # Add random atoms around the surface
            current = atoms.copy()
            for _ in range(size - 13):
                u1 = rng.uniform(0, 2 * np.pi)
                u2 = rng.uniform(-1, 1)
                r = 1.0 + rng.uniform(0.2, 0.5)
                x = r * np.sqrt(1 - u2**2) * np.cos(u1)
                y = r * np.sqrt(1 - u2**2) * np.sin(u1)
                z = r * u2
                current = np.vstack([current, [x, y, z]])
            seeds.append(current)
    
    # Add a random compact seed
    radius = 0.9 * size ** (1/3) + 1.0
    points = []
    for _ in range(size):
        found = False
        for _ in range(500):
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) > radius:
                continue
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
                found = True
                break
        if not found:
            p = rng.uniform(-radius*1.5, radius*1.5, 3)
            points.append(p)
    seeds.append(np.array(points))
    
    return seeds

def construct(instance):
    global _GLOBAL_START_TIME, _GLOBAL_DEADLINE
    
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        _GLOBAL_DEADLINE = _GLOBAL_START_TIME + 325.0  # Safety margin before 330s kill
    
    size = instance["n"]
    start_time = time.time()
    
    # Estimate time budget per instance
    # 6 instances, total 325s, so ~54s per instance
    # But we need to account for variable difficulty
    instance_deadline = start_time + 50.0
    if instance_deadline > _GLOBAL_DEADLINE:
        instance_deadline = _GLOBAL_DEADLINE - 5.0

    rng = np.random.default_rng(42 + size)
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate seeds
    seeds = _generate_seed(size, rng)
    
    # Relax each seed
    for seed in seeds:
        if time.time() > instance_deadline - 10.0:  # Leave time for basin hopping
            break
        try:
            e, pos = relax(seed, POTENTIAL)
            if e < best_energy:
                best_energy = e
                best_positions = pos
        except Exception:
            continue
    
    if best_positions is None:
        # Last resort fallback
        best_positions = np.random.randn(size, 3) * 0.1
        try:
            best_energy, best_positions = relax(best_positions, POTENTIAL)
        except:
            pass

    # Basin Hopping
    current_positions = best_positions.copy()
    temperature = 0.1
    step_size = 0.2
    max_hops = 200
    
    for hop in range(max_hops):
        if time.time() > instance_deadline:
            break
            
        perturbation = rng.normal(0, step_size, size=(size, 3))
        new_positions = current_positions + perturbation
        
        try:
            new_energy, new_relaxed_positions = relax(new_positions, POTENTIAL)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_positions = new_relaxed_positions.copy()
                current_positions = new_relaxed_positions.copy()
            else:
                delta_e = new_energy - best_energy
                acceptance_prob = np.exp(-delta_e / temperature)
                if rng.random() < acceptance_prob:
                    current_positions = new_relaxed_positions.copy()
                    
        except Exception:
            continue
        
        temperature *= 0.95
        if temperature < 1e-4:
            temperature = 1e-4
        step_size *= 0.995  # Slightly reduce step size

    return best_positions
