"""Lennard-Jones 55-atom cluster: construct Mackay icosahedron and polish with L-BFGS."""
N = 55
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _lj_energy(pos):
    """Compute total LJ energy for an (N,3) array."""
    n = pos.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r2 = np.dot(pos[i] - pos[j], pos[i] - pos[j])
            r6 = r2 ** 3
            total += 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return total


def _lj_grad(pos):
    """Compute gradient of total LJ energy."""
    n = pos.shape[0]
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            diff = pos[i] - pos[j]
            r2 = np.dot(diff, diff)
            r6 = r2 ** 3
            # dE/dr = 4*(-12*r^-13 + 6*r^-7)
            # dE/dxi = dE/dr * (xi - xj)/r = 4*(-12*r^-14 + 6*r^-8) * diff
            coeff = 4.0 * (-12.0 / (r2 * r6) + 6.0 / r6)
            grad[i] += coeff * diff
            grad[j] -= coeff * diff
    return grad


def _mackay_icosahedron_55():
    """
    Generate the two-shell Mackay icosahedron for 55 atoms.
    
    Shell 1: 12 vertices of a regular icosahedron (radius r1 = 1.12)
    Shell 2: 42 atoms - 12 face centers + 30 edge midpoints (radius r2 ≈ 2.06)
    Actually for Mackay icosahedron with 55 atoms:
      - 1 center atom
      - 12 first shell (icosahedron vertices)
      - 42 second shell: 12 face centers + 30 edge midpoints
    
    Let's use the standard construction:
    Center: 1 atom at origin
    First shell: 12 vertices of icosahedron at distance r1
    Second shell: 42 atoms at distance r2
    
    For Mackay icosahedron (2,2,2) or similar, the second shell consists of:
    - 12 atoms at face centers (distance from center = r2_face)
    - 30 atoms at edge midpoints (distance from center = r2_edge)
    
    Actually, the standard Mackay icosahedron for n=2 (two shells) has:
    - 1 center
    - 12 first shell vertices
    - 42 second shell: these are the icosahedron vertices of a larger icosahedron? No.
    
    Let me use the known coordinates. The Mackay icosahedron with 55 atoms:
    Shell 1 (12 atoms): icosahedron vertices
    Shell 2 (42 atoms): 12 face-centers + 30 edge-midpoints of the icosahedron
    
    For a regular icosahedron with vertices at distance R from center:
    - Face centers are at distance R * cos(acos(1/3)) ... let me compute properly.
    
    Icosahedron vertices (unit):
    (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1) where φ = (1+√5)/2
    
    Let's construct it properly.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 vertices of regular icosahedron (unit edge length ~ 2)
    verts = np.array([
        [0, 1, phi],
        [0, 1, -phi],
        [0, -1, phi],
        [0, -1, -phi],
        [1, phi, 0],
        [1, -phi, 0],
        [-1, phi, 0],
        [-1, -phi, 0],
        [phi, 0, 1],
        [phi, 0, -1],
        [-phi, 0, 1],
        [-phi, 0, -1],
    ], dtype=float)
    
    # Normalize to unit radius
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # First shell radius (LJ minimum distance is 2^(1/6) ≈ 1.1225)
    r1 = 2.0 ** (1.0 / 6.0)
    
    # Second shell: for Mackay icosahedron, the second shell radius is typically
    # r2 = r1 * sqrt(5) / 2 * something... Let me use the known ratio.
    # For a Mackay icosahedron, if first shell is at r1, second shell face centers
    # are at r1 * (1 + 1/phi) / something... 
    # 
    # Actually, let's think geometrically. The icosahedron has:
    # - 12 vertices
    # - 30 edges  
    # - 20 faces
    #
    # Face centers: 20 points
    # Edge midpoints: 30 points
    # Total second shell: 20 + 30 = 50? But we need 42.
    #
    # Wait, 1 + 12 + 42 = 55. So second shell is 42 atoms.
    # 42 = 12 + 30? That would be 12 face centers of a dodecahedron + 30 edge midpoints?
    # Or maybe it's a different decomposition.
    #
    # Let me reconsider. The Mackay icosahedron for n=2 has:
    # - 1 center
    # - 12 first shell
    # - 42 second shell
    #
    # The 42 second-shell atoms are arranged as: 12 at the "face center" positions 
    # (which correspond to the 12 vertices of the dual dodecahedron) and 30 at 
    # edge midpoint positions.
    #
    # Actually, I recall that for the 55-atom Mackay icosahedron:
    # - 1 center
    # - 12 first shell (icosahedron vertices)  
    # - 42 second shell: 30 edge midpoints + 12 "face center" type positions
    #
    # Let me compute face centers and edge midpoints of the icosahedron.
    
    # Find the 20 triangular faces of the icosahedron
    # Each face is a triangle of 3 vertices
    # For a regular icosahedron, each vertex connects to 5 others
    
    # Let's find faces by checking which triples of vertices form equilateral triangles
    # with the correct edge length
    edge_len = np.linalg.norm(verts_unit[0] - verts_unit[1])
    
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                d1 = np.linalg.norm(verts_unit[i] - verts_unit[j])
                d2 = np.linalg.norm(verts_unit[j] - verts_unit[k])
                d3 = np.linalg.norm(verts_unit[k] - verts_unit[i])
                if abs(d1 - edge_len) < 1e-10 and abs(d2 - edge_len) < 1e-10 and abs(d3 - edge_len) < 1e-10:
                    faces.append((i, j, k))
    
    # Face centers (20)
    face_centers = []
    for f in faces:
        fc = (verts_unit[f[0]] + verts_unit[f[1]] + verts_unit[f[2]]) / 3.0
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    # Normalize face centers to unit radius
    fc_norms = np.linalg.norm(face_centers, axis=1, keepdims=True)
    face_centers_unit = face_centers / fc_norms
    
    # Edge midpoints (30)
    edges = set()
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if abs(d - edge_len) < 1e-10:
                edges.add((i, j))
    
    edge_mids = []
    for e in edges:
        em = (verts_unit[e[0]] + verts_unit[e[1]]) / 2.0
        edge_mids.append(em)
    edge_mids = np.array(edge_mids)
    em_norms = np.linalg.norm(edge_mids, axis=1, keepdims=True)
    edge_mids_unit = edge_mids / em_norms
    
    # For Mackay icosahedron, the second shell radius r2 is chosen such that
    # the inter-shell distances are near LJ minimum.
    # The face centers are at distance r1 * (1/phi) from center? Let's check.
    # Actually face center distance from center:
    fc_dist = np.linalg.norm(face_centers_unit[0])
    em_dist = np.linalg.norm(edge_mids_unit[0])
    
    # For a unit-radius icosahedron, face center distance from center:
    # The face center is at distance cos(theta) where theta is the angle from center to face normal
    # For icosahedron, the face center (inradius) is at distance r_in = R * sqrt(5) * phi / (3*sqrt(3+sqrt(5)))... 
    # Let me just use the computed values.
    
    # The ratio of second shell to first shell for Mackay icosahedron:
    # r2_face / r1 and r2_edge / r1 should be chosen to minimize energy.
    # Typically r2 ≈ 2.0 * r1 for the outer shell.
    
    # Let's use the known optimal ratio. For Mackay icosahedron:
    # r2 = r1 * (1 + 2/phi) ≈ r1 * 2.236... no that's too large.
    
    # Actually, let me just set r2 based on geometry:
    # Face centers of unit icosahedron are at distance d_fc from center
    # Edge midpoints are at distance d_em from center
    # We scale the whole thing so first shell is at r1.
    # Then face centers are at r1 * (d_fc / d_vertex) where d_vertex = 1 (unit)
    # But we want to choose r2 independently to optimize.
    
    # Standard approach: place first shell at r1, second shell at r2 where r2 is a free parameter
    # to be optimized. Let's start with r2 = r1 * 2.0 and let relaxation adjust.
    
    r2 = r1 * 2.0  # initial guess for second shell radius
    
    # Build the 55-atom configuration
    # Center: 1 atom at origin
    center = np.array([0.0, 0.0, 0.0])
    
    # First shell: 12 atoms at r1
    shell1 = verts_unit * r1
    
    # Second shell: 42 atoms
    # Use 12 face-center-type positions and 30 edge-midpoint-type positions
    # But we have 20 face centers and 30 edge midpoints = 50, need only 42.
    # 
    # Reconsidering: maybe the 42 second-shell atoms are:
    # 12 positions corresponding to the 12 vertices of the dual dodecahedron (face centers of icosahedron)
    # 30 positions at edge midpoints
    # Total = 42. Yes!
    
    # So we use all 30 edge midpoints and 12 of the 20 face centers?
    # No, that breaks symmetry. 
    #
    # Let me reconsider the Mackay icosahedron structure.
    # Actually, I think the 42 second shell atoms are:
    # - 12 at the "pentagonal face center" positions (these are the 12 vertices of the dual dodecahedron, which correspond to the 20 triangular face centers... no)
    #
    # I think I'm overcomplicating this. Let me use a different approach:
    # The Mackay icosahedron can be generated by taking an icosahedral lattice.
    # 
    # Alternative: use the known fact that for n=2 Mackay icosahedron,
    # the 42 second-shell atoms are at the positions:
    # (±a, ±b, 0), (0, ±a, ±b), (±b, 0, ±a) and permutations, scaled appropriately
    # where a and b relate to the icosahedral symmetry.
    
    # Let me try a cleaner construction. The icosahedral group has orbits.
    # For the second shell of a Mackay icosahedron, the 42 points form:
    # - One orbit of 12 points (dodecahedral vertices = icosahedral face centers)
    # - One orbit of 30 points (edge midpoints)
    
    # Face centers of icosahedron = vertices of dual dodecahedron (12 points)
    # But we computed 20 face centers. The 12 dodecahedral vertices correspond to 
    # the 20 faces? No, dodecahedron has 12 vertices and 20 faces.
    # The dual of icosahedron (12 verts, 20 faces) is dodecahedron (20 verts, 12 faces).
    # So icosahedron face centers (20) = dodecahedron vertices (20).
    # Dodecahedron vertices (20) ≠ 12.
    
    # I'm confusing myself. Let me just use the 30 edge midpoints and 12 of the 
    # face centers chosen to maintain symmetry, or better yet, use a different 
    # well-known construction.
    
    # Actually, for the purpose of this problem, let me just generate a reasonable 
    # icosahedral-like structure and let L-BFGS relax it. The key insight is that
    # starting near the Mackay icosahedron will converge to it quickly.
    
    # Let me use a simpler approach: generate 55 points in icosahedral symmetry
    # by taking the 12 icosahedron vertices (first shell) and 42 points on a 
    # sphere of radius r2 with icosahedral symmetry.
    
    # The 42 points with icosahedral symmetry:
    # - 30 edge midpoints (orbit of 30)
    # - 12 face centers of the 20 faces... but that's 20, not 12.
    # 
    # Wait: 30 + 12 = 42. The 12 could be the vertices of the dual dodecahedron.
    # But dodecahedron has 20 vertices. 
    #
    # Let me just use 30 edge midpoints + 12 additional points. 
    # For icosahedral symmetry, orbits have sizes 1, 6, 12, 20, 30, 60.
    # 42 = 12 + 30. So one orbit of 12 and one orbit of 30.
    # Orbit of 12: dodecahedral vertices (but there are 20)... 
    # Actually, in icosahedral symmetry, a generic point has orbit size 120/2=60 (for full icosahedral group including reflections) or 60 (for chiral).
    # Points on special positions have smaller orbits.
    # - Center: orbit 1
    # - Vertex positions: orbit 12
    # - Edge midpoint positions: orbit 30  
    # - Face center positions: orbit 20
    # - Generic: orbit 60
    #
    # So 42 = 12 + 30 is perfect: 12 vertex-type + 30 edge-midpoint-type on the second shell.
    # But vertex-type on the second shell would be along the same directions as first shell vertices,
    # just at a larger radius. That gives 12 points at radius r2 in the same 12 directions.
    # Plus 30 edge midpoint directions at radius r2.
    # Total second shell: 12 + 30 = 42. 
    
    # So the construction is:
    # - 1 center
    # - 12 first shell at radius r1 in icosahedron vertex directions
    # - 12 second shell at radius r2 in icosahedron vertex directions (same directions, larger radius)
    # - 30 second shell at radius r2 in edge midpoint directions
    
    # This makes sense! The "12 face center" interpretation was wrong; it's actually
    # 12 points along the vertex axes at a larger radius, plus 30 edge midpoints.
    
    # Second shell: 12 points along vertex directions at r2, 30 edge midpoints at r2
    shell2_vertex = verts_unit * r2
    shell2_edge = edge_mids_unit * r2
    
    # Combine all 55 atoms
    all_points = np.vstack([
        center.reshape(1, 3),
        shell1,
        shell2_vertex,
        shell2_edge
    ])
    
    return all_points


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Generate initial Mackay icosahedron configuration
    pos = _mackay_icosahedron_55()
    
    # Ensure minimum distance constraint (0.5 apart)
    # Check and adjust if needed
    n = pos.shape[0]
    for _ in range(100):  # relaxation iterations to fix any constraint violations
        min_dist = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    min_dist = d
        if min_dist >= 0.5:
            break
        # Nudge atoms apart
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < 0.5 and d > 1e-10:
                    diff = (pos[i] - pos[j]) / d
                    pos[i] += diff * 0.01
                    pos[j] -= diff * 0.01
    
    # Compute initial energy
    best_pos = pos.copy()
    best_energy = _lj_energy(pos)
    
    # Try L-BFGS relaxation with time budget
    try:
        result = minimize(
            _lj_energy,
            pos,
            jac=_lj_grad,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        if result.x is not None:
            # Check minimum distance constraint
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    if np.linalg.norm(result.x[i] - result.x[j]) < 0.5:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                e = _lj_energy(result.x)
                if e < best_energy:
                    best_energy = e
                    best_pos = result.x.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() < deadline - 5.0:
        # Try a second relaxation with tighter settings
        try:
            result2 = minimize(
                _lj_energy,
                best_pos,
                jac=_lj_grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-18}
            )
            if result2.x is not None:
                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
                        if np.linalg.norm(result2.x[i] - result2.x[j]) < 0.5:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    e2 = _lj_energy(result2.x)
                    if e2 < best_energy:
                        best_energy = e2
                        best_pos = result2.x.copy()
        except Exception:
            pass
    
    return best_pos

# EVOLVE-BLOCK-END
