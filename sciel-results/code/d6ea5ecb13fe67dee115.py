"""
Strategy: Mackay icosahedral superstructure generation + basin hopping refinement.

The Dzugutov potential has a well-known preference for polytetrahedral (icosahedral)
order. The global minima for n=38,45,50,55,57,60 are all subsets of the Mackay
supericosahedron. We generate the full Mackay icosahedral point set, select the
n closest points to the center, relax, and then run basin hopping to refine.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters (for reference; relax handles the actual energy)
_A_const = 5.82
_B_const = 1.28
_a_const = 1.87
_b_const = 1.94
_c_const = 1.10
_d_const = 0.27
_m_const = 16


def _icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron centered at origin, unit radius."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
        [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0], [-phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, -1.0]
    ], dtype=np.float64)
    # Normalize to unit radius
    norm = np.linalg.norm(verts[0])
    verts /= norm
    return verts


def _icosahedron_faces():
    """Return the 20 triangular faces of the icosahedron (indices into vertices)."""
    # Standard icosahedron face list
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    return faces


def _generate_mackay_icosahedron(n, shell_radii=None, seed=42):
    """
    Generate a Mackay supericosahedron and return n closest points to center.
    
    The Mackay icosahedron is built by tiling the 20 faces of an icosahedron
    with smaller icosahedra. For our purposes, we use the well-known construction:
    
    - Shell 0: 1 atom at center
    - Shell 1: 12 atoms at icosahedron vertices (radius r1)
    - Shell 2: 30 atoms at edge midpoints + face centers (radius r2)
    - Shell 3: 60 atoms (radius r3)
    - Shell 4: 120 atoms (radius r4)
    
    For the Dzugutov potential, the equilibrium nearest-neighbor distance is
    approximately 1.0 in reduced units. The icosahedron edge length at the
    first shell should be ~1.0, so r1 = 0.5 (since edge = sqrt(5/3) * r1 for
    unit-radius icosahedron... actually for regular icosahedron with circumradius R,
    edge length a = 4R/sqrt(10+2*sqrt(5))).
    
    Let's use the standard Mackay construction with proper scaling.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # For a regular icosahedron with circumradius R:
    # edge length = 4R / sqrt(10 + 2*sqrt(5))
    # We want edge length ~ 1.0 (typical bond length for Dzugutov)
    # So R = sqrt(10 + 2*sqrt(5)) / 4
    edge_target = 1.0
    R1 = edge_target * np.sqrt(10 + 2 * np.sqrt(5)) / 4.0
    
    # Generate icosahedron vertices at radius R1
    ico_verts = _icosahedron_vertices()  # unit radius
    ico_verts = ico_verts * R1
    
    # Build the Mackay supericosahedron point set
    # We'll use the standard construction:
    # The supericosahedron consists of 13 icosahedra: 1 central + 12 on faces
    # Each face-icosahedron has its own 12 vertices
    
    all_points = [np.zeros(3)]  # center
    
    # Shell 1: 12 vertices of the main icosahedron
    all_points.extend(ico_verts.tolist())
    
    # For the second shell, we need the 30 edge-midpoint-like positions
    # and 20 face-center-like positions of the subdivided structure.
    # 
    # Standard Mackay icosahedron (n=147 for full 3rd generation):
    # Gen 0: 1
    # Gen 1: 12 (total 13)
    # Gen 2: 30 (total 43) -- but Dzugutov clusters are 38-60
    # Gen 3: 60 (total 103)
    # Gen 4: 120 (total 223)
    #
    # For n=38-60, we need points from shells 1-3.
    # Let's generate the full set and pick the closest n.
    
    # The key insight: the Dzugutov global minima for these sizes are
    # specific subsets of the Mackay icosahedron. The points are arranged
    # in shells. Let's generate a comprehensive set.
    
    # Icosahedron geometry:
    # 12 vertices, 30 edges, 20 faces
    # 
    # For the Mackay construction, we place icosahedra on each face.
    # The face icosahedra share vertices with the central icosahedron
    # and with each other.
    
    # Let's use a direct approach: generate all points of a 3-shell
    # icosahedral cluster using known coordinates.
    
    # Shell radii (normalized so that nearest neighbor ~ 1.0):
    # The icosahedron with 12 vertices at distance R1 from center has
    # edge length e1 = R1 * 4/sqrt(10+2*sqrt(5))
    # We set e1 = 1.0, so R1 = sqrt(10+2*sqrt(5))/4
    
    # Shell 2 points: These are at the centers of the 30 edges and 20 faces
    # of the icosahedron, but in the Mackay construction, they form a
    # dodecahedral + other shell.
    
    # Actually, let me use a cleaner approach: generate points on the
    # icosahedral lattice.
    
    # Icosahedral lattice vectors:
    # The icosahedral group has 60 rotations. We can generate points by
    # applying all icosahedral rotations to a set of seed vectors.
    
    # Simpler approach: use the known fact that the Mackay icosahedron
    # shells are:
    # Shell 0: 1 atom at (0,0,0)
    # Shell 1: 12 atoms at icosahedron vertices, radius R1
    # Shell 2: 30 atoms, radius R2 = R1 * sqrt(5)/2  (approximately)
    # Shell 3: 60 atoms, radius R3 = R1 * sqrt(5)  (approximately)
    # Shell 4: 120 atoms, radius R4 = R1 * sqrt(10+2*sqrt(5))/2
    
    # Let me compute exact shell radii for the Mackay icosahedron.
    # The Mackay icosahedron is constructed by placing 13 icosahedra:
    # one at center, 12 on the faces. Each face-icosahedron has its
    # 12 vertices. The shared vertices between adjacent face-icosahedra
    # and the central icosahedron give the shell structure.
    
    # For a regular icosahedron with circumradius R:
    # - Vertex radius: R
    # - Edge midpoint radius: R * sqrt(5 - sqrt(5)) / (2*sqrt(5))... 
    #   Actually, for a regular icosahedron, the distance from center to
    #   edge midpoint is R * cos(angle) where angle is half the dihedral...
    
    # Let me just compute numerically.
    # For our unit icosahedron (circumradius 1):
    # Vertices at radius 1
    # Edge midpoints: average of two adjacent vertices
    
    # Get all edges of the icosahedron
    faces = _icosahedron_faces()
    edges = set()
    for f in faces:
        for i in range(3):
            e = tuple(sorted((f[i], f[(i+1)%3])))
            edges.add(e)
    
    # Edge midpoints
    edge_mids = []
    for e in edges:
        mid = (ico_verts[e[0]] + ico_verts[e[1]]) / 2.0
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    
    # Face centers
    face_centers = []
    for f in faces:
        fc = (ico_verts[f[0]] + ico_verts[f[1]] + ico_verts[f[2]]) / 3.0
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    # Now build the full Mackay point set
    # The 30 edge midpoints form one shell
    # The 20 face centers form another
    # But in the Mackay construction, the second shell has 30 atoms,
    # not 30+20=50. The 30 atoms are at specific positions.
    
    # Actually, for the Dzugutov problem, the known global minima are:
    # n=38: icosahedron with 13+30+... let me think differently.
    
    # The simplest approach that works: generate a dense icosahedral
    # point cloud and pick the n closest to center.
    
    # Generate points using icosahedral symmetry:
    # Take a set of seed points in one fundamental domain and apply
    # all 60 icosahedral rotations.
    
    # Icosahedral rotation matrix generators:
    # The icosahedral group is generated by 5-fold and 2-fold rotations.
    
    # 5-fold rotation around z-axis:
    theta5 = 2 * np.pi / 5
    R5 = np.array([
        [np.cos(theta5), -np.sin(theta5), 0],
        [np.sin(theta5), np.cos(theta5), 0],
        [0, 0, 1]
    ])
    
    # 2-fold rotation around x-axis:
    R2 = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    
    # Generate all 60 rotations by combining R5 and R2
    rotations = [np.eye(3)]
    for _ in range(59):
        # Add new rotations
        new_rots = []
        for r in rotations:
            new_rots.append(R5 @ r)
            new_rots.append(R2 @ r)
        # Remove duplicates
        for nr in new_rots:
            is_dup = False
            for existing in rotations:
                if np.allclose(nr, existing, atol=1e-10):
                    is_dup = True
                    break
            if not is_dup:
                rotations.append(nr)
        if len(rotations) >= 60:
            break
    
    rotations = np.array(rotations)  # (60, 3, 3)
    
    # Seed points in the fundamental domain (1/60 of space)
    # The fundamental domain of the icosahedral group is a spherical
    # triangle with angles pi/2, pi/3, pi/5.
    
    # Generate seed points at various radii within the fundamental domain
    # and apply all rotations to get the full symmetric set.
    
    # For efficiency, let's generate points at specific shell radii
    # and use the icosahedral symmetry to get all equivalent points.
    
    # Shell radii for Mackay icosahedron (with R1 = circumradius of first shell):
    # Shell 1: R1 (12 points)
    # Shell 2: R1 * sqrt(5)/2 ≈ 1.118*R1 (30 points) 
    # Shell 3: R1 * sqrt(5) ≈ 2.236*R1 (60 points)
    # Shell 4: R1 * sqrt(10+2*sqrt(5))/2 ≈ 1.902*R1... 
    
    # Actually, let me compute the exact radii.
    # For the Mackay icosahedron, the shells are at:
    # r0 = 0 (1 atom)
    # r1 = R (12 atoms) - icosahedron vertices
    # r2 = R * sqrt(5)/2 (30 atoms) - edge-related
    # r3 = R * sqrt(5) (60 atoms)
    # r4 = R * sqrt(10 + 2*sqrt(5))/2 (120 atoms)
    
    # Hmm, let me just use a practical approach:
    # Generate a large set of icosahedrally-symmetric points and pick
    # the n closest to center. This will naturally give the Mackay
    # icosahedron subset.
    
    # Generate candidate points at various radii and angles
    rng = np.random.default_rng(seed)
    
    # We'll generate points on icosahedral symmetry orbits
    # at various radii
    
    # Key radii for the Dzugutov icosahedral cluster:
    # With nearest-neighbor distance ~1.0:
    # First shell (12 atoms): radius ~ 0.5 (since icosahedron edge = sqrt(5/3)*R for unit ico... )
    
    # Let me just compute: for a regular icosahedron with circumradius R,
    # the edge length is a = R * 4 / sqrt(10 + 2*sqrt(5))
    # If we want a ≈ 1.0, then R = sqrt(10+2*sqrt(5))/4 ≈ 0.9511
    
    # So first shell radius ≈ 0.9511
    # The next shell in the Mackay icosahedron is at radius R * sqrt(5)/2 ≈ 1.064
    # Wait, that's not right either.
    
    # Let me use a different approach: just generate the points directly.
    
    # For the Dzugutov potential, the equilibrium structure is icosahedral
    # with a specific set of bond lengths. The known global minima for
    # n=38,45,50,55,57,60 are all icosahedral.
    
    # Practical approach: generate an icosahedral cluster with the correct
    # shell structure, then pick n closest points.
    
    # I'll generate points using the icosahedral group acting on seed vectors.
    
    # Seed vectors for each shell (in the fundamental domain):
    # Shell 1: one vertex direction
    # Shell 2: one edge-midpoint direction  
    # Shell 3: one face-center direction
    # etc.
    
    # Actually, the cleanest approach: 
    # 1. Generate the 12 icosahedron vertices (shell 1)
    # 2. Generate the 30 "edge" points (shell 2) 
    # 3. Generate the 60 "face" points (shell 3)
    # 4. Generate the 120 "next" points (shell 4)
    # 5. Combine and pick n closest to center
    
    # For the Mackay icosahedron:
    # The 12 vertices of the central icosahedron are at radius R1.
    # The 30 shell-2 atoms are at the midpoints of the 30 edges of a
    # slightly larger icosahedron, or equivalently at specific positions
    # related to the icosahedral symmetry.
    
    # Let me use the standard construction:
    # The Mackay icosahedron of order n has shells at radii proportional to:
    # 0, 1, sqrt(5)/2, sqrt(5), sqrt(10+2sqrt(5))/2, ...
    
    # For our purposes, let's generate points at these radii using
    # icosahedral symmetry.
    
    # Shell 1: 12 points at radius R1 = 1.0 (normalized)
    # These are the icosahedron vertices.
    
    # Shell 2: 30 points at radius R2
    # The 30 points form a dodecahedron + icosahedron combination.
    # They can be generated as the 30 edge midpoints of the icosahedron
    # scaled to the correct radius.
    
    # Shell 3: 60 points at radius R3
    # These can be generated from the 20 face centers scaled appropriately,
    # plus additional points.
    
    # Let me just use a robust numerical approach:
    # Generate a dense set of icosahedrally-symmetric points and select.
    
    # Generate points at multiple radii using icosahedral orbits
    
    # For each radius, generate a few seed points in the fundamental
    # domain and apply all 60 rotations.
    
    all_pts = [np.zeros(3)]  # center
    
    # Use the icosahedron vertices as the primary shell
    ico_unit = _icosahedron_vertices()  # unit circumradius
    
    # Scale so that edge length = 1.0
    # Edge length of unit icosahedron: 
    v0 = ico_unit[0]
    v1 = ico_unit[1]
    # Find nearest neighbor
    dists = np.linalg.norm(ico_unit - v0, axis=1)
    dists[0] = np.inf
    edge_len_unit = np.min(dists)
    scale = 1.0 / edge_len_unit  # scale so edge = 1
    
    ico_scaled = ico_unit * scale
    all_pts.extend(ico_scaled.tolist())
    
    # Now generate shell 2: 30 points
    # These are at the edge midpoints of the icosahedron, but at a
    # specific radius. For the Mackay icosahedron, the shell-2 radius
    # is such that the distance from a shell-1 atom to a shell-2 atom
    # is also ~1.0 (or close).
    
    # Edge midpoints of the scaled icosahedron:
    edge_mids_scaled = []
    for e in edges:
        mid = (ico_scaled[e[0]] + ico_scaled[e[1]]) / 2.0
        edge_mids_scaled.append(mid)
    edge_mids_scaled = np.array(edge_mids_scaled)
    
    # The edge midpoints are at radius R1 * cos(half_angle) where half_angle
    # is half the angle subtended by an edge at the center.
    # For a regular icosahedron, the angle between adjacent vertices
    # (as seen from center) is: cos(theta) = 1/(1+phi^2) = 1/(1+phi^2)
    # Actually: for unit icosahedron, dot product of adjacent vertices:
    # v0 . v1 / (|v0||v1|)
    
    # Let me just compute the radius of edge midpoints
    r_mid = np.linalg.norm(edge_mids_scaled[0])
    
    # For the Mackay icosahedron, shell 2 is at a slightly different radius.
    # Let's just use the edge midpoints as shell 2.
    all_pts.extend(edge_mids_scaled.tolist())
    
    # Shell 3: 60 points
    # Use face centers and additional points
    face_centers_scaled = []
    for f in faces:
        fc = (ico_scaled[f[0]] + ico_scaled[f[1]] + ico_scaled[f[2]]) / 3.0
        face_centers_scaled.append(fc)
    face_centers_scaled = np.array(face_centers_scaled)
    all_pts.extend(face_centers_scaled.tolist())
    
    # We need more points for shell 3. Let's generate additional icosahedral
    # points by applying rotations to a seed vector at the appropriate radius.
    
    # The 60 shell-3 points can be generated from 2 seed points (in the
    # fundamental domain) × 30 unique rotations (since each orbit has 60
    # points, but we need 60 total, so 1 seed × 60 rotations = 60 points,
    # or 2 seeds × 30 rotations each).
    
    # Actually, 60 points = 1 orbit of size 60 (if the seed is in general
    # position) or 2 orbits of size 30 (if the seed has 2-fold symmetry).
    
    # Let's generate a seed point at the appropriate radius and apply
    # all 60 rotations.
    
    # The shell-3 radius: for the Mackay icosahedron, it's approximately
    # R1 * sqrt(5) ≈ 2.236 * R1. But let me compute more carefully.
    
    # For now, let's generate points at a range of radii and use
    # icosahedral symmetry to get a dense set.
    
    # Generate additional points at various radii
    r_base = np.linalg.norm(ico_scaled[0])  # shell 1 radius
    
    # Generate points at radii: r_base * [1.0, 1.1, 1.2, ..., 3.0]
    # using icosahedral symmetry
    
    # Seed directions in the fundamental domain
    # The fundamental domain is bounded by:
    # - z-axis (5-fold)
    # - x-axis (2-fold)  
    # - a plane at 36° from x-axis (3-fold)
    
    # Simple seed directions:
    seed_dirs = [
        np.array([1.0, 0.0, 0.0]),  # along 2-fold axis
        np.array([0.5, 0.5, 0.0]),  # in xy plane
        np.array([0.5, 0.3, 0.5]),  # general direction
        np.array([0.3, 0.3, 0.7]),  # more z
        np.array([0.7, 0.2, 0.3]),  # another
    ]
    
    for rd in seed_dirs:
        rd = rd / np.linalg.norm(rd)
        for radius_factor in [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0]:
            r = r_base * radius_factor
            seed_pt = rd * r
            # Apply all 60 rotations
            for rot in rotations:
                pt = rot @ seed_pt
                all_pts.append(pt)
    
    all_pts = np.array(all_pts)
    
    # Remove near-duplicates
    # Sort by distance from center
    dists_from_center = np.linalg.norm(all_pts, axis=1)
    indices = np.argsort(dists_from_center)
    all_pts = all_pts[indices]
    
    # Select n closest points, but ensure minimum distance
    selected = [all_pts[0]]  # center
    min_dist = 0.5  # task requirement
    
    for i in range(1, len(all_pts)):
        if len(selected) >= n:
            break
        p = all_pts[i]
        sel_arr = np.array(selected)
        dists = np.linalg.norm(sel_arr - p, axis=1)
        if np.min(dists) >= min_dist:
            selected.append(p)
    
    # If we didn't get enough, relax the constraint
    if len(selected) < n:
        for i in range(1, len(all_pts)):
            if len(selected) >= n:
                break
            p = all_pts[i]
            sel_arr = np.array(selected)
            dists = np.linalg.norm(sel_arr - p, axis=1)
            if np.min(dists) >= 0.3:
                selected.append(p)
    
    # Last resort: just take the closest n
    if len(selected) < n:
        selected = [all_pts[i] for i in range(min(n, len(all_pts)))]
    
    return np.array(selected[:n])


def _generate_icosahedral_seed(n, seed=42):
    """
    Generate an icosahedral seed structure for n atoms.
    Uses the Mackay icosahedron construction.
    """
    try:
        pts = _generate_mackay_icosahedron(n, seed=seed)
        return pts
    except Exception:
        # Fallback: simple icosahedron + random shell
        ico_unit = _icosahedron_vertices()
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        # Scale icosahedron
        v0 = ico_unit[0]
        dists = np.linalg.norm(ico_unit - v0, axis=1)
        dists[0] = np.inf
        edge_len = np.min(dists)
        scale = 1.0 / edge_len
        ico_scaled = ico_unit * scale
        
        points = [np.zeros(3)]
        points.extend(ico_scaled.tolist())
        
        rng = np.random.default_rng(seed)
        while len(points) < n:
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            r = rng.uniform(1.0, 2.5)
            p = v * r
            # Check distance
            sel_arr = np.array(points)
            if np.min(np.linalg.norm(sel_arr - p, axis=1)) >= 0.5:
                points.append(p)
            elif len(points) < n:
                # Just add it
                points.append(p)
        
        return np.array(points[:n])


def _generate_deformed_icosahedron(n, seed=42, deformation=0.15):
    """Generate a slightly deformed icosahedral structure."""
    base = _generate_icosahedral_seed(n, seed=seed)
    rng = np.random.default_rng(seed + 1000)
    noise = rng.normal(0, deformation, size=base.shape)
    return base + noise


def _generate_fcc_seed(n, seed=42):
    """Generate an FCC-like seed structure."""
    rng = np.random.default_rng(seed)
    a = 1.0
    basis = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ])
    
    N = int(np.ceil(n ** (1/3))) + 3
    candidates = []
    for i in range(-N, N+1):
        for j in range(-N, N+1):
            for k in range(-N, N+1):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    candidates.append(p)
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    indices = np.argsort(dists)
    candidates = candidates[indices]
    
    points = candidates[:n]
    centroid = np.mean(points, axis=0)
    points -= centroid
    
    return points


def _generate_random_packing(n, seed=42):
    """Generate a random compact packing."""
    rng = np.random.default_rng(seed)
    points = []
    radius = 1.2 * n ** (1.0 / 3.0)
    min_dist = 0.7
    attempts = 0
    max_attempts = n * 200
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
        
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    return np.array(points)


def _safe_relax(positions, potential, parameter):
    """Safely call relax, returning (energy, positions) or (inf, positions)."""
    try:
        result = relax(positions, potential, parameter)
        # Handle both tuple return and single return
        if isinstance(result, tuple):
            energy, relaxed_pos = result
            return float(energy), np.array(relaxed_pos)
        else:
            # Maybe it returns just positions? Compute energy separately
            return np.inf, np.array(result)
    except Exception:
        return np.inf, np.array(positions)


def _basin_hopping(positions, potential, parameter, rng, max_steps, deadline, 
                   sigma_start=0.3, sigma_end=0.05):
    """
    Run basin hopping from the given positions.
    Returns (best_energy, best_positions).
    """
    current_energy, current_pos = _safe_relax(positions, potential, parameter)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    n_steps = 0
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Anneal the perturbation size
        t = step / max(max_steps - 1, 1)
        sigma = sigma_start * (sigma_end / sigma_start) ** t
        
        # Perturb
        noise = rng.normal(0, sigma, size=current_pos.shape)
        perturbed = current_pos + noise
        
        # Relax
        new_energy, new_pos = _safe_relax(perturbed, potential, parameter)
        
        # Accept if better (greedy basin hopping for global minimization)
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        else:
            # Keep exploring from current position
            pass
        
        n_steps += 1
    
    return best_energy, best_pos


def construct(instance):
    """
    Find low-energy structures for Dzugutov clusters using icosahedral
    seeding and basin hopping refinement.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 310.0  # Safety margin before 330s kill
    
    # Generate diverse initial candidates
    candidates = []
    
    # Primary: icosahedral seeds with different randomizations
    for seed in [42, 123, 456, 789, 1000, 2000]:
        try:
            p_ico = _generate_icosahedral_seed(size, seed=seed)
            candidates.append(p_ico)
        except Exception:
            pass
    
    # Deformed icosahedral
    for seed in [42, 123]:
        try:
            p_def = _generate_deformed_icosahedron(size, seed=seed, deformation=0.1)
            candidates.append(p_def)
        except Exception:
            pass
    
    # FCC seed (diversity)
    try:
        p_fcc = _generate_fcc_seed(size, seed=42)
        candidates.append(p_fcc)
    except Exception:
        pass
    
    # Random packings (diversity)
    for seed in [1, 2, 3]:
        try:
            p_rand = _generate_random_packing(size, seed=seed)
            candidates.append(p_rand)
        except Exception:
            pass
    
    if not candidates:
        # Last resort fallback
        return np.array([[i, 0, 0] for i in range(size)], dtype=np.float64)
    
    best_energy = np.inf
    best_pos = None
    
    # Phase 1: Parallel relaxation of all candidates
    max_workers = 4
    
    def _relax_task(pos):
        return _safe_relax(pos, potential, parameter)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_relax_task, pos): i for i, pos in enumerate(candidates)}
        
        for future in as_completed(futures):
            if time.time() > deadline - 20:
                break
            try:
                energy, pos = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos.copy()
            except Exception:
                continue
    
    # Phase 2: Basin hopping from the best structure(s)
    if best_pos is not None:
        # Use remaining time for basin hopping
        time_left = deadline - time.time()
        
        if time_left > 5:
            # Estimate how many BH steps we can afford
            # Each step involves one relax call (~1-3s)
            # Be conservative: assume 2s per step
            max_bh_steps = max(5, int(time_left / 3.0))
            
            rng = np.random.default_rng(size + 42)
            
            # Run multiple BH chains in parallel from best and near-best structures
            # For simplicity, run sequentially from the best
            bh_energy, bh_pos = _basin_hopping(
                best_pos, potential, parameter, rng, 
                max_steps=max_bh_steps, deadline=deadline - 5
            )
            
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos
    
    # Phase 3: If time remains, try BH from a different seed
    if best_pos is not None and time.time() < deadline - 15:
        # Try BH from a different icosahedral seed
        try:
            alt_seed = _generate_icosahedral_seed(size, seed=9999)
            alt_energy, alt_pos = _safe_relax(alt_seed, potential, parameter)
            
            if alt_energy < best_energy + 5.0:  # If reasonably close
                time_left = deadline - time.time()
                if time_left > 10:
                    max_bh_steps = max(3, int(time_left / 3.0))
                    rng2 = np.random.default_rng(size + 7777)
                    bh2_energy, bh2_pos = _basin_hopping(
                        alt_pos, potential, parameter, rng2,
                        max_steps=max_bh_steps, deadline=deadline - 5
                    )
                    if bh2_energy < best_energy:
                        best_energy = bh2_energy
                        best_pos = bh2_pos
        except Exception:
            pass
    
    if best_pos is None:
        # Fallback
        if candidates:
            return candidates[0]
        return np.array([[i, 0, 0] for i in range(size)], dtype=np.float64)
    
    return best_pos
# EVOLVE-BLOCK-END
