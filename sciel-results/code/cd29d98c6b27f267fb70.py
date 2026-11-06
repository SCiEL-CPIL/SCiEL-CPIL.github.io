import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Constants
POTENTIAL = 'dzugutov'
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
    """Return the 12 vertices of a regular dodecahedron (dual of icosahedron), normalized to unit radius."""
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1/phi, phi], [0, -1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, -phi],
        [1/phi, phi, 0], [-1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, -phi, 0],
        [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
    ]
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _generate_mackay_icosahedron(n_shells=3):
    """
    Generate Mackay icosahedron with n_shells.
    n_shells=1: 1 atom
    n_shells=2: 13 atoms (1 + 12)
    n_shells=3: 55 atoms (1 + 12 + 42)
    """
    if n_shells <= 0:
        return np.zeros((0, 3))
    
    atoms = [np.zeros(3)]
    
    if n_shells >= 2:
        ico_verts = _get_icosahedron_vertices()
        atoms.extend(ico_verts.tolist())
    
    if n_shells >= 3:
        # Add 42 atoms: 30 edge midpoints + 12 face centers (dodecahedron vertices)
        ico_verts = _get_icosahedron_vertices()
        
        # Find edges: pairs of vertices at minimum distance
        dists = np.zeros((12, 12))
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                dists[i, j] = d
                dists[j, i] = d
        
        # Minimum distance is the edge length
        min_dist = np.min(dists[dists > 0])
        
        # Edge midpoints
        for i in range(12):
            for j in range(i+1, 12):
                if abs(dists[i, j] - min_dist) < 1e-10:
                    midpoint = (ico_verts[i] + ico_verts[j]) / 2
                    atoms.append(midpoint)
        
        # Face centers: use dodecahedron vertices scaled to match
        dode_verts = _get_dodecahedron_vertices()
        # Scale dodecahedron to same radius as icosahedron face centers
        # The face center distance from origin for unit icosahedron is ~0.7558
        ico_face_radius = np.linalg.norm(np.mean(ico_verts[0:3], axis=0))
        dode_scale = ico_face_radius / np.linalg.norm(dode_verts[0])
        dode_verts_scaled = dode_verts * dode_scale
        atoms.extend(dode_verts_scaled.tolist())
    
    return np.array(atoms)

def _generate_seed(size, rng):
    """Generate a diverse set of seeds for the given size."""
    seeds = []
    
    # 1. Mackay icosahedron based seeds
    if size >= 13:
        # Use 55-atom Mackay as base for larger clusters
        mackay_55 = _generate_mackay_icosahedron(3)
        if size == 55:
            seeds.append(mackay_55.copy())
        elif size > 55:
            # Add random atoms around surface
            current = mackay_55.copy()
            for _ in range(size - 55):
                u1 = rng.uniform(0, 2 * np.pi)
                u2 = rng.uniform(-1, 1)
                r = 1.0 + rng.uniform(0.3, 0.6)
                x = r * np.sqrt(1 - u2**2) * np.cos(u1)
                y = r * np.sqrt(1 - u2**2) * np.sin(u1)
                z = r * u2
                current = np.vstack([current, [x, y, z]])
            seeds.append(current)
        elif size == 13:
            mackay_13 = _generate_mackay_icosahedron(2)
            seeds.append(mackay_13.copy())
    
    # 2. Random compact seed
    radius = 0.8 * size ** (1/3) + 1.0
    points = []
    for _ in range(size):
        found = False
        for _ in range(300):
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
    
    # 3. FCC-like seed
    fcc_points = []
    a = 1.0  # lattice constant
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                # FCC positions
                fcc_points.append([i*a, j*a, k*a])
                fcc_points.append([(i+0.5)*a, (j+0.5)*a, k*a])
                fcc_points.append([(i+0.5)*a, j*a, (k+0.5)*a])
                fcc_points.append([i*a, (j+0.5)*a, (k+0.5)*a])
    
    fcc_points = np.array(fcc_points)
    # Keep only points within radius
    mask = np.linalg.norm(fcc_points, axis=1) < radius
    fcc_points = fcc_points[mask]
    
    if len(fcc_points) >= size:
        idx = rng.choice(len(fcc_points), size=size, replace=False)
        seeds.append(fcc_points[idx].copy())
    elif len(fcc_points) > 0:
        # Add random atoms to fill
        current = fcc_points.copy()
        for _ in range(size - len(fcc_points)):
            p = rng.uniform(-radius, radius, 3)
            current = np.vstack([current, [p]])
        seeds.append(current)
    
    return seeds

def _repulsive_annealing(positions, rng, n_steps=50, max_time=None):
    """
    Repulsive Annealing: use a repulsive potential to push atoms apart,
    breaking local minima. Then relax back.
    """
    current = positions.copy()
    n_atoms = len(current)
    best_energy = float('inf')
    best_positions = current.copy()
    
    # Start with strong repulsion, decrease over time
    repulsion_strength = 5.0
    decay = 0.95
    
    for step in range(n_steps):
        if max_time is not None and time.time() > max_time:
            break
        
        # Compute repulsive forces: push all atoms apart
        forces = np.zeros_like(current)
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                r_vec = current[i] - current[j]
                r = np.linalg.norm(r_vec)
                if r < 1e-6:
                    r_vec = rng.normal(0, 0.01, 3)
                    r = np.linalg.norm(r_vec)
                # Repulsive force ~ 1/r^2
                f_mag = repulsion_strength / (r * r + 0.1)
                f_vec = f_mag * r_vec / r
                forces[i] += f_vec
                forces[j] -= f_vec
        
        # Move atoms
        current = current + 0.05 * forces
        
        # Relax
        try:
            e, relaxed = relax(current, POTENTIAL)
            if e < best_energy:
                best_energy = e
                best_positions = relaxed.copy()
                current = relaxed.copy()
        except Exception:
            pass
        
        repulsion_strength *= decay
    
    return best_positions, best_energy

def construct(instance):
    global _GLOBAL_START_TIME, _GLOBAL_DEADLINE
    
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        _GLOBAL_DEADLINE = _GLOBAL_START_TIME + 295.0  # Safety margin

    size = instance["n"]
    start_time = time.time()
    
    # Estimate time budget per instance
    instance_deadline = start_time + 40.0
    if instance_deadline > _GLOBAL_DEADLINE:
        instance_deadline = _GLOBAL_DEADLINE - 5.0

    rng = np.random.default_rng(42 + size)
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate seeds
    seeds = _generate_seed(size, rng)
    
    # Relax each seed
    for seed in seeds:
        if time.time() > instance_deadline - 5.0:
            break
        try:
            e, pos = relax(seed, POTENTIAL)
            if e < best_energy:
                best_energy = e
                best_positions = pos.copy()
        except Exception:
            continue
    
    if best_positions is None:
        # Last resort fallback
        best_positions = np.random.randn(size, 3) * 0.1
        try:
            best_energy, best_positions = relax(best_positions, POTENTIAL)
        except:
            pass
    
    # Repulsive Annealing phase
    if time.time() < instance_deadline - 10.0:
        ra_deadline = instance_deadline - 5.0
        try:
            ra_positions, ra_energy = _repulsive_annealing(
                best_positions, rng, n_steps=30, max_time=ra_deadline
            )
            if ra_energy < best_energy:
                best_energy = ra_energy
                best_positions = ra_positions.copy()
        except Exception:
            pass
    
    # Basin Hopping with adaptive step size
    current_positions = best_positions.copy()
    temperature = 0.1
    step_size = 0.15
    max_hops = 80
    
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
                # Reduce step size when finding better solutions
                step_size *= 0.95
            else:
                delta_e = new_energy - best_energy
                acceptance_prob = np.exp(-delta_e / temperature)
                if rng.random() < acceptance_prob:
                    current_positions = new_relaxed_positions.copy()
                # Increase step size when stuck
                step_size *= 1.02
                step_size = min(step_size, 0.3)
                
        except Exception:
            continue
        
        temperature *= 0.97
        if temperature < 1e-4:
            temperature = 1e-4
    
    return best_positions
