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
        # Avoid division by zero or negative base for even m if r is very small
        # r^-m is huge for small r
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
            # Scale midpoint to be on the shell? Actually midpoints of edges of icosahedron 
            # are not on a sphere. We want positions that form a good shell.
            # Let's use the standard Mackay icosahedron shells.
            # Shell 1: 12 atoms (radius R1)
            # Shell 2: 42 atoms (radius R2). 
            # The 42 positions are: 12 at vertices of a larger icosahedron? No.
            # Mackay icosahedron:
            # 1 atom at center.
            # 12 atoms at radius R1 (vertices of icosahedron).
            # 42 atoms at radius R2.
            # The 42 atoms consist of:
            # - 12 atoms at the vertices of a larger icosahedron (radius R2)
            # - 30 atoms at the midpoints of the edges of the larger icosahedron? No.
            # Let's just generate points on a sphere of radius R2 with icosahedral symmetry.
            # R2 is typically sqrt(2) * R1 for close packing? 
            # In icosahedral packing, the second shell radius is such that atoms touch.
            # Let's use R2 = spacing * sqrt(2.0) approx or just generate a dense spherical shell.
            
            # Simpler approach for generation:
            # Use the 12 icosahedron vertices for shell 1.
            # For shell 2, use the 42 positions of a Mackay icosahedron.
            # The 42 positions are:
            # 12 at (0, 1, phi) scaled to R2
            # 12 at (1, phi, 0) scaled to R2
            # 12 at (phi, 0, 1) scaled to R2
            # 6 at (1, 0, 0) scaled to R2? No.
            
            # Let's just use a simpler heuristic:
            # Shell 1: 12 icosahedron vertices at radius R1.
            # Shell 2: 42 points.
            #   12 points: icosahedron vertices at radius R2.
            #   30 points: midpoints of edges of the icosahedron defined by R2 vertices, projected to sphere R2?
            #   Actually, the 30 points in Mackay are at the midpoints of the 30 edges of the icosahedron.
            #   If the icosahedron has vertices at R2, the edge midpoints are at distance R2 * cos(2pi/5) ?
            #   Let's just generate the 42 points explicitly.
            
            # Vertices of icosahedron (12)
            verts = ico_vertices / spacing # unit vectors
            # Scale to R2
            R2 = spacing * 1.9 # Approximate second shell radius
            shell2_verts = verts * R2
            
            # Edge midpoints (30)
            # Edges of icosahedron connect vertices with distance 2.0 (if unit edge length is 2)
            # In our unit vectors, edge length is 2.
            # Midpoint of edge (u, v) is (u+v)/2.
            # Distance from center: |(u+v)/2| = 0.5 * |u+v|.
            # |u+v|^2 = |u|^2 + |v|^2 + 2 u.v = 1 + 1 + 2 cos(theta).
            # For icosahedron, angle between adjacent vertices is such that cos(theta) = 1/phi^2 ?
            # Actually, let's just compute the midpoint and normalize to R2.
            
            midpoints = []
            for i in range(12):
                for j in range(i+1, 12):
                    dist = np.linalg.norm(verts[i] - verts[j])
                    if dist < 2.1: # Adjacent vertices
                        mid = (verts[i] + verts[j]) / 2
                        mid_norm = mid / np.linalg.norm(mid)
                        midpoints.append(mid_norm * R2)
            
            second_shell = list(shell2_verts) + midpoints
            
    # Third shell: 92 positions
    R3 = spacing * 2.8
    # 92 points:
    # 12 at icosahedron vertices scaled to R3
    # 30 at edge midpoints scaled to R3
    # 30 at face centers scaled to R3
    # 20 at ...?
    # Let's just add more points on the sphere if needed.
    
    points = []
    # Center
    points.append([0.0, 0.0, 0.0])
    
    # Shell 1
    for v in ico_vertices:
        if len(points) < n:
            points.append(v)
    
    # Shell 2
    for v in second_shell:
        if len(points) < n:
            points.append(v)
            
    # Fill remaining with scaled positions or random on sphere
    if len(points) < n:
        # Generate points on a sphere of radius R3
        R3 = spacing * 2.8
        # Use Fibonacci sphere for even distribution
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n - len(points)):
            y = 1 - (i / max(1, n - len(points))) * 2
            radius_at_y = np.sqrt(1 - y**2)
            theta = golden_angle * i
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            p = np.array([x, y, z]) * R3
            points.append(p)
    
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
    # Distance between centers in polytetrahedral packing
    # If they share a face, the distance is related to the geometry.
    # Let's try spacing * 1.5
    second_center = face_center + face_normal * spacing * 1.5
    
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
        third_center = face_center2 + face_normal2 * spacing * 1.5
        
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
    
    # If we have time, try a few more random starts
    remaining_time = deadline - time.time()
    if remaining_time > 30 and best_positions is not None:
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            rng_trial = np.random.default_rng(size + 10000 + trial)
            points_trial = _generate_random_compact(size, size + 10000 + trial, 1.05)
            try:
                energy, positions = relax(points_trial, potential, parameter)
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
