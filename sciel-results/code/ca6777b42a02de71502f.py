"""Dzugutov cluster global minima search using basin hopping with diverse polytetrahedral starts."""
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c_true = 1.10
d_true = 0.27
m = 16.0

def _find_r_min():
    """Find the minimum energy distance r_min for the Dzugutov potential."""
    r = np.linspace(0.5, 2.0, 1000)
    energies = []
    for ri in r:
        if ri < a:
            e = A * (ri ** (-m) - B) * np.exp(c_true / (ri - a))
        elif ri < b:
            e = B * np.exp(d_true / (ri - b))
        else:
            e = 0.0
        energies.append(e)
    energies = np.array(energies)
    idx = np.argmin(energies)
    return r[idx]

R_MIN = _find_r_min()

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

def _build_mackay_icosahedron(n, rng, r_min=R_MIN):
    """
    Build a Mackay icosahedral structure.
    For n=13: center + 12 vertices
    For n=55: 1 + 12 + 42 (Mackay icosahedron)
    For other n: start with appropriate Mackay shell and add/remove atoms.
    """
    points = [np.zeros(3)]
    
    # Shell 1: 12 icosahedron vertices
    ico = _icosahedron_vertices(r_min)
    points.extend(list(ico))
    
    if n <= 13:
        if n < 13:
            # For small n, just take random subset of the 13
            idx = rng.choice(13, n, replace=False)
            return np.array([points[i] for i in idx])
        return np.array(points)
    
    # Shell 2: Mackay icosahedron has 42 atoms on second shell
    # The 42 positions come from:
    # - 12 face centers
    # - 30 edge midpoints
    # We generate these explicitly to ensure correct geometry
    
    phi = (1 + np.sqrt(5)) / 2
    # The radius of the second shell in a Mackay icosahedron
    # The distance from center to face center is R1 * (1/3) * (1 + 1/phi) ? 
    # Actually, let's use the vector approach from the icosahedron vertices.
    
    ico_verts_unit = _icosahedron_vertices(1.0)
    
    # Find faces and edges of the unit icosahedron
    faces = []
    edges = set()
    n_ico = len(ico_verts_unit)
    
    # Precompute distances to find faces/edges
    dists = np.zeros((n_ico, n_ico))
    for i in range(n_ico):
        for j in range(i+1, n_ico):
            d = np.linalg.norm(ico_verts_unit[i] - ico_verts_unit[j])
            dists[i,j] = d
            dists[j,i] = d
            
    # The edge length of a unit circumradius icosahedron is 2/sqrt(1+1/phi^2) ? 
    # Let's just find the smallest distances as edges.
    edge_length = np.min(dists[dists > 1e-6])
    
    for i in range(n_ico):
        for j in range(i+1, n_ico):
            if abs(dists[i,j] - edge_length) < 1e-6:
                edges.add((i,j))
                
    # Find faces: triangles of edges
    for i in range(n_ico):
        for j in range(i+1, n_ico):
            if (i,j) not in edges: continue
            for k in range(j+1, n_ico):
                if (i,k) in edges and (j,k) in edges:
                    faces.append((i,j,k))
                    
    # Generate Shell 2 points
    # 12 Face Centers
    face_centers = []
    for f in faces:
        center = (ico_verts_unit[f[0]] + ico_verts_unit[f[1]] + ico_verts_unit[f[2]]) / 3.0
        # Normalize and scale to r_min? 
        # The distance from center to face center in a regular icosahedron of radius R is R * (1/3) * (1 + 1/phi) ??
        # Let's just normalize the vector and scale it to the correct radius.
        # The radius of the second shell R2 = R1 * sqrt(1 + 1/phi^2) is for the *vertices* of the dual?
        # Actually, for Mackay icosahedron, the shell 2 atoms are at the face centers and edge midpoints.
        # Let's calculate the distance of the face center from origin.
        d_fc = np.linalg.norm(center)
        # Scale this unit vector to R2.
        # What is R2? 
        # In a Mackay icosahedron, the shell 2 radius is such that the bond length between shell 1 and shell 2 is r_min.
        # Let's assume the "ideal" geometry has bond length r_min.
        # Distance from center to shell 1 vertex is R1.
        # Distance from center to shell 2 face center is R2.
        # Distance from shell 1 vertex to shell 2 face center is r_min.
        # This forms a triangle.
        # However, a simpler heuristic often used is to place them at the face centers of the icosahedron of radius R1, then relax.
        # Let's place them at the face centers of the icosahedron of radius R1, then scale the whole cluster?
        # No, let's just generate points at the face centers of the unit icosahedron, normalize them, and multiply by R1 * factor.
        # A common factor is that R2 ≈ 1.4 * R1.
        # Let's use the fact that the bond length is r_min.
        # If we place the face center at distance R2, and a vertex at R1, the distance between them is:
        # d^2 = R1^2 + R2^2 - 2 R1 R2 cos(theta)
        # Where theta is the angle between the vertex and the face center.
        
        # Let's just use the vector direction of the face center.
        dir_vec = center / d_fc
        face_centers.append(dir_vec)
        
    # 30 Edge Midpoints
    edge_mids = []
    for e in edges:
        mid = (ico_verts_unit[e[0]] + ico_verts_unit[e[1]]) / 2.0
        d_em = np.linalg.norm(mid)
        dir_vec = mid / d_em
        edge_mids.append(dir_vec)
        
    # We have 12 + 30 = 42 directions.
    # We need to determine the radius R2.
    # Let's assume the bond length between shell 1 and shell 2 is r_min.
    # Pick a vertex v and a face center f that are "connected" (adjacent).
    # In an icosahedron, each face center is adjacent to 3 vertices.
    # Let's pick the first face center and find its adjacent vertices.
    # Actually, we can just estimate R2.
    # For a regular icosahedron, the distance from center to vertex is R.
    # The distance from center to face center is R * (1/3) * (1 + 1/phi) is incorrect.
    # The distance from center to face center is R * (1/3) * (1 + 1/phi) * something?
    # Let's calculate it for the unit icosahedron.
    R1_unit = 1.0
    R2_unit = np.linalg.norm((ico_verts_unit[faces[0][0]] + ico_verts_unit[faces[0][1]] + ico_verts_unit[faces[0][2]]) / 3.0)
    
    # We want the distance between a shell 1 atom (radius R1) and a shell 2 atom (radius R2) to be r_min.
    # Let's assume R1 = r_min. Then we solve for R2.
    # The angle between a vertex and its adjacent face center in a unit icosahedron:
    v = ico_verts_unit[faces[0][0]]
    f = (ico_verts_unit[faces[0][0]] + ico_verts_unit[faces[0][1]] + ico_verts_unit[faces[0][2]]) / 3.0
    f_norm = f / np.linalg.norm(f)
    cos_theta = np.dot(v, f_norm)
    
    # Law of cosines: r_min^2 = R1^2 + R2^2 - 2 R1 R2 cos_theta
    # Let R1 = r_min.
    # r_min^2 = r_min^2 + R2^2 - 2 r_min R2 cos_theta
    # 0 = R2^2 - 2 r_min R2 cos_theta
    # R2 = 2 r_min cos_theta
    
    R2 = 2 * r_min * cos_theta
    
    # Generate shell 2 points
    shell2_points = []
    for dir_vec in face_centers:
        p = dir_vec * R2
        shell2_points.append(p)
    for dir_vec in edge_mids:
        p = dir_vec * R2
        shell2_points.append(p)
        
    # We have 42 points. We need `needed` points.
    needed = min(n - 13, 42)
    
    # Randomly select `needed` points from the 42 to break symmetry if n < 55
    if needed < 42:
        indices = rng.choice(42, needed, replace=False)
        selected_points = [shell2_points[i] for i in indices]
    else:
        selected_points = shell2_points
        
    points.extend(selected_points)
    
    # If we need more than 55, add random points on outer shells
    while len(points) < n:
        r_outer = R2 + 0.3 * rng.random()
        u = rng.random()
        v = rng.random()
        theta = 2 * np.pi * u
        phi_ang = np.arccos(2 * v - 1)
        p = np.array([
            r_outer * np.sin(phi_ang) * np.cos(theta),
            r_outer * np.sin(phi_ang) * np.sin(theta),
            r_outer * np.cos(phi_ang)
        ])
        points.append(p)
        
    # If we have too many (shouldn't happen with the logic above, but just in case)
    if len(points) > n:
        # Keep center and shell 1, remove from shell 2
        keep = list(range(13))
        for i in range(13, len(points)):
            if len(keep) < n:
                keep.append(i)
        points = [points[i] for i in keep]
        
    return np.array(points)

def _build_random_compact(n, rng, r_min=R_MIN):
    """Build a random compact cluster."""
    # Estimate radius
    radius = r_min * (n / 13) ** (1/3) * 1.2
    points = []
    attempts = 0
    min_dist = 0.5
    
    while len(points) < n and attempts < 10000:
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
        p = base + 0.2 * rng.standard_normal(3)
        points.append(p)
        
    return np.array(points)

def construct(instance):
    """Find low-energy Dzugutov clusters using basin hopping with diverse starts."""
    start_time = time.time()
    time_limit = 300.0
    safety_margin = 15.0
    n = instance["n"]
    potential = instance["potential"]
    param = None
    
    rng = np.random.default_rng(n + 42)
    
    best_energy = float('inf')
    best_positions = None
    
    def check_time():
        return (time.time() - start_time) < (time_limit - safety_margin)
    
    def try_relax(pos, label=""):
        nonlocal best_energy, best_positions
        try:
            energy, relaxed_pos = relax(pos, potential, param)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos.copy()
                return True
        except Exception:
            pass
        return False

    # Phase 1: Generate diverse initial guesses
    initial_guesses = []
    
    # 1. Mackay icosahedral guesses (primary)
    # We generate a few variations by perturbing the ideal Mackay structure
    for i in range(3):
        if not check_time(): break
        pos = _build_mackay_icosahedron(n, rng)
        # Add small random perturbation to break symmetry
        pos_noisy = pos + 0.05 * rng.standard_normal((n, 3))
        initial_guesses.append(pos_noisy)
        
    # 2. Random compact guesses
    for i in range(3):
        if not check_time(): break
        pos = _build_random_compact(n, rng)
        initial_guesses.append(pos)
        
    # 3. Defective Mackay: Remove random atoms from Mackay and let relaxation fill them
    if n > 13:
        for i in range(3):
            if not check_time(): break
            pos = _build_mackay_icosahedron(n, rng)
            # Remove 2-5 random atoms (not the center)
            n_remove = min(5, n - 1)
            remove_indices = rng.choice(np.arange(1, n), n_remove, replace=False)
            mask = np.ones(n, dtype=bool)
            mask[remove_indices] = False
            pos_defect = pos[mask]
            # Re-add them as random points near the surface? Or just relax the smaller cluster and add?
            # Better: Just relax the smaller cluster, then add atoms randomly near the surface and relax again.
            # For simplicity, just perturb the full Mackay structure more strongly.
            pos_strong_noisy = pos + 0.2 * rng.standard_normal((n, 3))
            initial_guesses.append(pos_strong_noisy)

    # Phase 2: Basin Hopping
    # We will perform basin hopping from each initial guess
    # To save time, we limit the number of kicks per start
    
    kicks_per_start = 10
    kick_sizes = np.linspace(0.1, 0.3, kicks_per_start)
    
    for start_idx, pos_guess in enumerate(initial_guesses):
        if not check_time(): break
        
        current_pos = pos_guess
        current_energy = float('inf')
        
        # First relax the initial guess
        try_relax(current_pos, f"init_{start_idx}")
        if best_positions is not None:
            current_pos = best_positions.copy()
            current_energy = best_energy
        
        # Basin Hopping loop
        for kick_idx in range(kicks_per_start):
            if not check_time(): break
            
            kick_size = kick_sizes[kick_idx]
            # Add noise
            noise = rng.standard_normal((n, 3))
            pos_kicked = current_pos + kick_size * noise
            
            # Relax the kicked structure
            improved = try_relax(pos_kicked, f"start{start_idx}_kick{kick_idx}")
            
            if improved:
                # Accept the new best
                current_pos = best_positions.copy()
                current_energy = best_energy
            else:
                # If not improved, we can still keep the current best as the reference for the next kick
                # Standard basin hopping accepts if energy is lower than the *previous* best in the chain,
                # but since we track global best, we just use the global best if it improved,
                # otherwise we might want to keep the local best of this chain to explore deeper.
                # However, given the time constraints, sticking to the global best is safer to avoid getting stuck.
                # Let's try a mix: if not improved, keep current_pos as is (which is the global best).
                pass

    # Phase 3: Final polish with more kicks if time permits
    if best_positions is not None and check_time():
        extra_kicks = 20
        for i in range(extra_kicks):
            if not check_time(): break
            kick_size = 0.1 + 0.02 * i
            pos_kicked = best_positions + kick_size * rng.standard_normal((n, 3))
            try_relax(pos_kicked, f"final_kick_{i}")

    # Fallback
    if best_positions is None:
        best_positions = _build_random_compact(n, rng)
        
    return best_positions
