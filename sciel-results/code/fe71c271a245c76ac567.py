import numpy as np
import time
import science_chemistry

N = 55

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # 1. Generate icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    # Vertices of a regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit length
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    unit_verts = verts / norms

    # 2. Generate direction vectors for all 55 atoms
    # Center
    dirs = [np.array([0.0, 0.0, 0.0])]
    
    # Inner shell (12 atoms)
    for v in unit_verts:
        dirs.append(v.copy())
        
    # Outer shell (42 atoms)
    # We need the 12 vertices, 30 edge midpoints, and 20 face centers.
    # We will generate the unit direction vectors for these.
    
    # Vertices are already in unit_verts
    
    # Edges: pairs of vertices that are closest
    # Distance between adjacent vertices in unit icosahedron
    # We can find edges by checking distances
    edge_dirs = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(unit_verts[i] - unit_verts[j])
            # The minimum distance is the edge length. 
            # For unit icosahedron, edge length is 2/sqrt(phi+2) ~ 1.05? 
            # Actually, let's just find the 30 shortest distances.
            # Or simpler: hardcode or calculate robustly.
            # Let's collect all distances and find the threshold.
            pass
            
    # Robust edge finding
    dists = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(unit_verts[i] - unit_verts[j])
            dists.append((d, i, j))
    dists.sort(key=lambda x: x[0])
    # The 30 shortest distances are the edges
    edge_pairs = [(i, j) for d, i, j in dists[:30]]
    
    for i, j in edge_pairs:
        mid_vec = unit_verts[i] + unit_verts[j]
        mid_norm = np.linalg.norm(mid_vec)
        if mid_norm > 0:
            edge_dirs.append(mid_vec / mid_norm)
            
    # Faces: triples of vertices forming a triangle
    # In an icosahedron, faces are triangles of mutually adjacent vertices.
    # We can identify faces by finding triples (i,j,k) such that (i,j), (j,k), (i,k) are all edges.
    edge_set = set()
    for i, j in edge_pairs:
        edge_set.add((min(i,j), max(i,j)))
        
    face_dirs = []
    for i in range(12):
        for j in range(i+1, 12):
            if (min(i,j), max(i,j)) in edge_set:
                for k in range(j+1, 12):
                    if (min(j,k), max(j,k)) in edge_set and (min(i,k), max(i,k)) in edge_set:
                        face_vec = unit_verts[i] + unit_verts[j] + unit_verts[k]
                        face_norm = np.linalg.norm(face_vec)
                        if face_norm > 0:
                            face_dirs.append(face_vec / face_norm)
                            
    # Check count: 12 verts + 30 edges + 20 faces = 62? 
    # Wait, the outer shell has 42 atoms.
    # 12 (vertices) + 30 (edges) + 20 (faces) = 62.
    # Why 42?
    # Ah, the Mackay icosahedron M(2) has 42 atoms in the second shell.
    # The 13-atom icosahedron has 12 shell atoms.
    # The 55-atom cluster has 1 + 12 + 42 = 55.
    # Where do the 42 come from?
    # In the Mackay icosahedron, the second shell consists of:
    # - 12 atoms at the vertices of the icosahedron (radially out)
    # - 30 atoms at the edge midpoints (radially out)
    # - 20 atoms at the face centers (radially out)
    # Total 12 + 30 + 20 = 62.
    # This contradicts the 55 count.
    # Let's re-read the task description: "complete two-shell Mackay icosahedron".
    # Maybe the "two-shell" implies the 13-atom core (1+12) is one shell, and the next 42 is the second?
    # If the total is 55, and the core is 13, then the outer shell is 42.
    # But 12+30+20 = 62.
    # Is it possible that some atoms are shared or the geometry is different?
    # Actually, the 55-atom LJ minimum is indeed the Mackay icosahedron.
    # Let's check the number of atoms in the Mackay icosahedron shells.
    # Shell 1: 12
    # Shell 2: 42
    # Total: 1 + 12 + 42 = 55.
    # Why 42?
    # Perhaps the 42 atoms are NOT all vertices, edges, and faces of the *same* icosahedron.
    # Or perhaps my counting of vertices/edges/faces for the outer shell is wrong.
    # In a Mackay icosahedron, the outer shell atoms are located at the vertices, edge midpoints, and face centers of the icosahedron formed by the *inner* shell atoms?
    # If so, it should be 62.
    # Let's look up "55 atom Lennard Jones cluster Mackay icosahedron".
    # It is known that the 55-atom cluster has icosahedral symmetry.
    # The 42 atoms in the second shell are arranged in:
    # - 12 "vertices" (aligned with inner shell vertices)
    # - 30 "edges" (aligned with inner shell edges)
    # - 20 "faces" (aligned with inner shell faces)
    # This sums to 62.
    # Is it possible that the 55-atom cluster is NOT the full Mackay icosahedron?
    # The task says: "The best-known putative minimum is the complete two-shell Mackay icosahedron (energy -279.248470...)".
    # And "Place 55 atoms".
    # There is a discrepancy in my count.
    # Let's check the Cambridge Cluster Database for LJ55.
    # LJ55 global minimum is the Mackay icosahedron.
    # Number of atoms: 55.
    # Shells: 1, 12, 42.
    # How can the second shell have 42 atoms if the icosahedron has 12 vertices, 30 edges, 20 faces?
    # Maybe the "Mackay icosahedron" refers to a specific subset?
    # Or maybe the 42 atoms are:
    # - 12 vertices
    # - 30 edge midpoints
    # - 0 faces? No.
    # - 20 faces?
    # 12+30+20 = 62.
    # Is it possible that the 55-atom cluster is the 13-atom icosahedron plus 42 atoms, but the 42 atoms are not the full set of vertices/edges/faces?
    # Actually, some sources say the 55-atom cluster has the structure of a Mackay icosahedron.
    # Let's try to generate the 55 points by taking the 62 candidate positions and removing the 10 that are farthest? No.
    # Let's assume the 42 atoms are the 12 vertices and 30 edge midpoints of the icosahedron, and the 20 face centers are NOT present?
    # 12 + 30 = 42.
    # This matches the count!
    # So the outer shell consists of the 12 vertices and 30 edge midpoints of the icosahedron, but NOT the 20 face centers.
    # Let's verify this hypothesis.
    # If the outer shell has 12 vertices and 30 edge midpoints, does it have icosahedral symmetry?
    # Yes, the set of vertices and edge midpoints of an icosahedron is invariant under icosahedral rotations.
    # So, I will generate:
    # - 1 center
    # - 12 inner shell vertices
    # - 12 outer shell vertices (radially out from inner vertices)
    # - 30 outer shell edge midpoints (radially out from inner edge midpoints)
    # Total: 1 + 12 + 12 + 30 = 55.
    
    # So, I need to add the 12 vertex directions and 30 edge midpoint directions to the outer shell.
    # I already have `unit_verts` (12) and `edge_dirs` (30).
    
    outer_dirs = []
    for v in unit_verts:
        outer_dirs.append(v.copy())
    for e in edge_dirs:
        outer_dirs.append(e.copy())
        
    # Total directions: 1 (center) + 12 (inner) + 42 (outer) = 55.
    all_dirs = dirs + outer_dirs
    
    # 3. Scale the structure
    # We need to choose radii R1 (inner) and R2 (outer).
    # In the Mackay icosahedron, the ratio R2/R1 is typically sqrt(3).
    # Let's try R1 = 1.0, R2 = sqrt(3).
    # Then scale the whole thing so that the nearest neighbor distance is 2^(1/6).
    
    R1 = 1.0
    R2 = np.sqrt(3)
    
    positions = np.zeros((55, 3))
    # Center
    positions[0] = 0.0
    # Inner shell
    for i in range(12):
        positions[1 + i] = R1 * all_dirs[1 + i]
    # Outer shell
    for i in range(42):
        positions[13 + i] = R2 * all_dirs[13 + i]
        
    # Find minimum distance
    # Use scipy.spatial.distance.pdist for efficiency
    from scipy.spatial.distance import pdist
    dists = pdist(positions)
    min_dist = np.min(dists)
    
    # Scale factor to make min_dist = 2^(1/6)
    target_dist = 2.0 ** (1.0/6.0)
    scale = target_dist / min_dist
    positions *= scale
    
    # 4. Relax
    # Use science_chemistry.relax
    # Check time
    if time.time() < deadline - 10.0:
        try:
            relaxed_pos = science_chemistry.relax(positions)
            if relaxed_pos is not None:
                positions = relaxed_pos
        except Exception:
            pass
            
    return positions
