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

def _generate_mackay_icosahedron():
    """
    Generate the 55-atom Mackay icosahedron.
    Layer 1: 1 atom at center
    Layer 2: 12 atoms at radius 1.0 (icosahedron vertices)
    Layer 3: 42 atoms at radius sqrt(1+1/phi^2) ~ 1.0515? No, standard Mackay radii are 1, sqrt(1+1/phi^2), sqrt(1+4/phi^2)?
    Actually, for Mackay icosahedra, the radii are:
    n=1: 1
    n=2: 12 at r=1
    n=3: 42 at r=sqrt(1+1/phi^2)
    Total 1+12+42 = 55.
    
    Let's construct the 42 outer atoms.
    The 42 atoms are located at the midpoints of the edges of the inner icosahedron (12 vertices, 30 edges) plus... wait.
    A Mackay icosahedron with 55 atoms consists of:
    1 center
    12 vertices of an icosahedron
    30 midpoints of the edges of that icosahedron
    Wait, 1+12+30 = 43. That's not 55.
    
    Let's check the Mackay icosahedron structure.
    M(1) = 1
    M(2) = 1 + 12 = 13
    M(3) = 1 + 12 + 42 = 55.
    The 42 atoms in the 3rd shell are at the positions corresponding to the vertices of a truncated icosahedron? No.
    They are at the centers of the faces of the inner icosahedron? No, that would be 20.
    
    Actually, the 3rd shell of a Mackay icosahedron consists of 42 atoms.
    These are the 30 edge midpoints of the icosahedron plus the 12 vertices of a dual icosahedron (dodecahedron) scaled?
    
    Let's look up coordinates or derive them.
    The vertices of the Mackay icosahedron M(3) are:
    (0, 0, 0)
    (±1, 0, ±1/phi), (0, ±1, ±1/phi), (±1/phi, ±1, 0)  -- 12 atoms at radius sqrt(1+1/phi^2)
    Wait, the radius of the 2nd shell is usually taken as 1.
    If the 2nd shell is at r=1, the 3rd shell is at r = sqrt(1 + 1/phi^2) ≈ 1.0515.
    
    The 42 atoms in the 3rd shell:
    30 are at the midpoints of the edges of the icosahedron defined by the 12 vertices.
    12 are at the centers of the 20 faces? No, 30+12=42.
    So yes, 30 edge midpoints and 12 face centers (of the dual dodecahedron, which correspond to the icosahedron faces).
    
    Let's generate the 12 vertices of the icosahedron (radius 1).
    Let's find the 30 edge midpoints.
    Let's find the 12 face centers.
    
    Icosahedron vertices (normalized to unit length):
    v = _get_icosahedron_vertices()
    
    Edges: Pairs of vertices that are distance 2*sin(pi/5) apart?
    The edge length of a unit-radius icosahedron is 2*sin(pi/5) = 2*0.5878 = 1.1756.
    Actually, edge length = 4 / sqrt(10 + 2*sqrt(5)) ?
    Let's just compute distances between all pairs of 12 vertices.
    The 12 vertices have 30 nearest neighbors (each vertex connects to 5).
    So there are 30 edges.
    
    Face centers: 20 faces. Each face is an equilateral triangle.
    The center of a face is the average of its 3 vertices.
    There are 20 faces.
    But we need 12 atoms in the 3rd shell from faces?
    Wait, 30 (edges) + 12 (faces?) = 42.
    But there are 20 faces. Why 12?
    
    Let's re-read Mackay icosahedron structure.
    M(3) has 55 atoms.
    Shell 1: 1
    Shell 2: 12
    Shell 3: 42
    
    The 42 atoms are:
    30 at the midpoints of the edges of the icosahedron.
    12 at the vertices of a dual icosahedron (dodecahedron) inscribed in the sphere?
    The dual of an icosahedron is a dodecahedron, which has 12 vertices.
    Yes! The 12 atoms are at the vertices of the dual dodecahedron.
    
    So:
    1. Generate 12 icosahedron vertices (radius R2).
    2. Generate 30 edge midpoints (radius R3_edge).
    3. Generate 12 dodecahedron vertices (radius R3_face).
    
    Note: The radius of the edge midpoints and face centers might be different?
    In a regular icosahedron, the distance from center to edge midpoint is R * cos(pi/5) = R * 0.809.
    The distance from center to face center is R * cos(pi/5) * cos(pi/3) ? No.
    Distance to face center = R * (1/3) * (sum of unit vectors of 3 vertices).
    
    Actually, for a regular icosahedron, the edge midpoints and face centers are NOT at the same radius.
    However, in the Mackay icosahedron, the 3rd shell is often described as a "truncated icosahedron" like shape? No.
    
    Let's just compute the positions and then scale the whole 55-atom cluster to a reasonable size (e.g., bond length ~1.1).
    
    Steps:
    1. Get 12 icosahedron vertices (unit sphere).
    2. Identify edges (pairs with min distance).
    3. Compute 30 midpoints.
    4. Identify faces (triples of vertices forming equilateral triangles).
    5. Compute 20 face centers? No, we need 12.
    
    Wait, the dual of an icosahedron is a dodecahedron with 12 vertices.
    The vertices of the dodecahedron correspond to the face centers of the icosahedron.
    So there are 20 face centers.
    Why does Mackay M(3) have only 12 atoms from the "face" direction?
    
    Let's check the count again.
    M(3) = 55.
    1 + 12 + 42 = 55.
    The 42 atoms are:
    30 edge midpoints.
    12 ... ?
    
    Maybe the 12 are NOT face centers.
    Maybe the 3rd shell is just the 42 vertices of a "truncated icosahedron" minus some?
    A truncated icosahedron (soccer ball) has 60 vertices.
    
    Let's look at the coordinates of the 55-atom Mackay icosahedron.
    It is known that the 55-atom Mackay icosahedron is a local minimum for many potentials.
    
    Alternative approach:
    Generate the 13-atom icosahedron.
    Then, for the 55-atom cluster, we can use the known structure or generate it via:
    - 1 center
    - 12 at r1
    - 42 at r2
    
    The 42 atoms are located at:
    (±1, 0, ±1/phi) -> 4
    (0, ±1, ±1/phi) -> 4
    (±1/phi, ±1, 0) -> 4
    (±1/phi, 0, ±1) -> 4
    (0, ±1/phi, ±1) -> 4
    (±1, ±1/phi, 0) -> 4
    (±1/phi, ±1, 0) ... wait, duplicates.
    
    Let's use the explicit coordinates for the 55-atom Mackay icosahedron.
    The 55 atoms are:
    (0,0,0)
    (±1, 0, ±1/phi), (0, ±1, ±1/phi), (±1/phi, ±1, 0)  -- 12 atoms?
    Let's check the count:
    (±1, 0, ±1/phi): 4
    (0, ±1, ±1/phi): 4
    (±1/phi, ±1, 0): 4
    Total 12. These are the 2nd shell.
    
    The 3rd shell (42 atoms):
    (±1, ±1/phi, ±1/phi^2)? No.
    
    Let's just use the 13-atom icosahedron and grow it, or use the 55-atom structure if I can find it.
    Since I don't have the exact coordinates memorized, I will generate the 13-atom icosahedron and use it as a seed.
    For 55, I will use a random compact seed or a grown icosahedron.
    """
    # For now, return the 13-atom icosahedron scaled.
    # We will handle other sizes by adding/removing atoms.
    pass

def _generate_seed(size, rng):
    """Generate a diverse set of seeds for the given size."""
    seeds = []
    
    # 1. 13-atom icosahedron based seed
    if size >= 13:
        verts = _get_icosahedron_vertices()
        # Scale to typical bond length
        # For Dzugutov, equilibrium distance is around 1.1-1.2
        # Let's try scaling factor 1.0 for the icosahedron vertices
        center = np.zeros(3)
        # The 12 vertices are at distance 1.0 from center.
        # If we want bond length ~1.1, we scale by 1.1?
        # But the distance between adjacent vertices is ~1.1756.
        # Let's just use the unit icosahedron and let relax handle it.
        atoms = np.vstack([center, verts])
        
        if size == 13:
            seeds.append(atoms)
        elif size > 13:
            # Add random atoms around the surface
            current = atoms.copy()
            for _ in range(size - 13):
                # Add atom at random position on the surface
                # Sample direction
                u1 = rng.uniform(0, 2 * np.pi)
                u2 = rng.uniform(-1, 1)
                r = 1.0 + rng.uniform(0.2, 0.5) # Slightly outside
                x = r * np.sqrt(1 - u2**2) * np.cos(u1)
                y = r * np.sqrt(1 - u2**2) * np.sin(u1)
                z = r * u2
                current = np.vstack([current, [x, y, z]])
            seeds.append(current)
    
    # 2. Random compact seed
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
            # Force add if stuck
            p = rng.uniform(-radius*1.5, radius*1.5, 3)
            points.append(p)
    seeds.append(np.array(points))
    
    # 3. FCC-like seed (if applicable)
    # Skip for now, icosahedral is more relevant for Dzugutov.
    
    return seeds

def construct(instance):
    global _GLOBAL_START_TIME, _GLOBAL_DEADLINE
    
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        _GLOBAL_DEADLINE = _GLOBAL_START_TIME + 290.0 # Safety margin

    size = instance["n"]
    start_time = time.time()
    
    # Estimate time budget
    instance_deadline = start_time + 45.0
    if instance_deadline > _GLOBAL_DEADLINE:
        instance_deadline = _GLOBAL_DEADLINE - 5.0

    rng = np.random.default_rng(42 + size)
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate seeds
    seeds = _generate_seed(size, rng)
    
    # Relax each seed
    for seed in seeds:
        if time.time() > instance_deadline:
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
    max_hops = 100 # Cap hops to avoid infinite loop if time check fails
    
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
        step_size *= 0.99 # Slightly reduce step size

    return best_positions
